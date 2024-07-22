#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import cv2
import sys
from PIL import Image, ImageDraw
from collections import defaultdict
from typing import Any, Dict, List
import pyrealsense2 as rs
import clip
from sentence_transformers import util

import numpy as np
import torch
import tqdm

from habitat import Config
from habitat.core.logging import logger
from habitat.utils.visualizations.utils import observations_to_image, explored_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainerOracle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_env
from habitat_baselines.common.rollout_storage import RolloutStorageOracle
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    poll_checkpoint_folder
)
from habitat_baselines.rl.ppo import PPOOracle, ProposedPolicyOracle
from habitat.utils.visualizations import fog_of_war, maps

import kachaka_api
sys.path.append(f"/home/{os.environ['USER']}/Desktop/habitat2kachaka/kachaka-api/python/")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer



def to_grid(client, realworld_x, realworld_y):
    map = client.get_png_map()
    grid_resolution = map.resolution
        
    # マップの原点からエージェントまでの距離を算出
    dx = realworld_x - map.origin.x
    dy = realworld_y - map.origin.y
        
    # エージェントのグリッド座標を求める
    grid_x = dx / map.resolution
    grid_y = dy / map.resolution
    grid_y = map.height-grid_y
        
    # resizeの分、割る
    grid_x /= 3
    grid_y /= 3
        
    # 四捨五入する
    grid_x = int(grid_x)
    grid_y = int(grid_y)
        
    return grid_x, grid_y


@baseline_registry.register_trainer(name="oracle")
class PPOTrainerO(BaseRLTrainerOracle):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.env = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None
        
        self._num_picture = config.TASK_CONFIG.TASK.PICTURE.NUM_PICTURE
        self.save_picture_reward = 0.02
        #撮った写真のsaliencyとrange_mapを保存
        self._taken_picture_list = []
        
        self._observed_object_ci = []
        self._target_index_list = []
        self._taken_index_list = []
        
        self.TARGET_THRESHOLD = 250
        self._dis_pre = []
        
        # ストリームの設定
        realsense_config = rs.config()
        realsense_config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
        realsense_config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)

        # ストリーミング開始
        self.pipeline = rs.pipeline()
        #self.pipeline.stop()
        self.pipeline.start(realsense_config)
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)


    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = ProposedPolicyOracle(
            agent_type = self.config.TRAINER_NAME,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_size=ppo_cfg.hidden_size,
            device=self.device,
            previous_action_embedding_size=self.config.RL.PREVIOUS_ACTION_EMBEDDING_SIZE,
            use_previous_action=self.config.RL.PREVIOUS_ACTION
        )
        self.actor_critic.to(self.device)

        self.agent = PPOOracle(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "traj_metrics", "saliency"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue
                
            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results
    
    def _create_new_image_embedding(self, obs):
        image = Image.fromarray(obs)
        image = self.preprocess(image)
        image = torch.tensor(image).to(self.device).unsqueeze(0)
        embetting = self.clip_model.encode_image(image).float()
        return embetting

    def _calculate_pic_sim(self, picture_list):
        if len(picture_list) <= 1:
            return 0.0
            
        sim_list = [[-10 for _ in range(len(picture_list))] for _ in range(len(picture_list))]

        for i in range(len(picture_list)):
            emd = self._create_new_image_embedding(picture_list[i][1])
            for j in range(i, len(picture_list)):
                if i == j:
                    sim_list[i][j] = 0.0
                    continue
                    
                emd2 = self._create_new_image_embedding(picture_list[j][1])
                sim_list[i][j] = util.pytorch_cos_sim(emd, emd2).item()
                sim_list[j][i] = sim_list[i][j]
                
        total_sim = np.sum(sim_list)
        total_sim /= (len(picture_list)*(len(picture_list)-1))
        return total_sim
            
    def _select_pictures(self, taken_picture_list):
        results = []
        res_val = 0.0

        sorted_picture_list = sorted(taken_picture_list, key=lambda x: x[0], reverse=True)
        i = 0
        while True:
            if len(results) == self._num_picture:
                break
            if i == len(sorted_picture_list):
                break
            emd = self._create_new_image_embedding(sorted_picture_list[i][1])
            is_save = self._decide_save(emd, results)

            if is_save == True:
                results.append(sorted_picture_list[i])
                res_val += sorted_picture_list[i][0]
            i += 1

        result_len = max(len(results), 1)
        res_val /= result_len
        return results, res_val

    def _select_random_pictures(self, taken_picture_list):
        results = taken_picture_list
        num = len(taken_picture_list)
        if len(taken_picture_list) > self._num_picture:
            results = random.sample(taken_picture_list, self._num_picture)
            num = self._num_picture
        res_val = 0.0

        for i in range(num):
            res_val += results[i][0]

        return results, res_val


    def _decide_save(self, emd, results):
        for i in range(len(results)):
            check_emb = self._create_new_image_embedding(results[i][1])

            sim = util.pytorch_cos_sim(emd, check_emb).item()
            if sim >= self._select_threthould:
                return False
        return True


    def _create_results_image(self, picture_list, explored_picture):
        images = []
        x_list = []
        y_list = []
    
        if len(picture_list) == 0:
            return None

        for i in range(self._num_picture):
            idx = i%len(picture_list)
            images.append(Image.fromarray(picture_list[idx][1]))
            x_list.append(picture_list[idx][2])
            y_list.append(picture_list[idx][3])

        width, height = images[0].size
        result_width = width * 5
        result_height = height * 2
        result_image = Image.new("RGB", (result_width, result_height))

        for i, image in enumerate(images):
            x_offset = (i % 5) * width
            y_offset = (i // 5) * height
            result_image.paste(image, (x_offset, y_offset))
        
        draw = ImageDraw.Draw(result_image)
        for x in range(width, result_width, width):
            draw.line([(x, 0), (x, result_height)], fill="black", width=7)
        for y in range(height, result_height, height):
            draw.line([(0, y), (result_width, y)], fill="black", width=7)

        # explored_pictureの新しい横幅を計算
        if explored_picture.height != 0:
            aspect_ratio = explored_picture.width / explored_picture.height
        else:
            aspect_ratio = explored_picture.width
        new_explored_picture_width = int(result_height * aspect_ratio)

        # explored_pictureをリサイズ
        explored_picture = explored_picture.resize((new_explored_picture_width, result_height))

        # 最終画像の幅を計算
        final_width = result_width + new_explored_picture_width

        # 最終画像を作成
        final_image = Image.new('RGB', (final_width, result_height), color=(255, 255, 255))

        # result_imageを貼り付け
        final_image.paste(result_image, (0, 0))

        # リサイズしたexplored_pictureを貼り付け
        final_image.paste(explored_picture, (result_width, 0))

        return final_image, x_list, y_list


    def create_description_from_results_image(self, results_image):
        input_text = "You are an excellent property writer. This picture consists of 10 pictures arranged in one picture, 5 horizontally and 2 vertically on one building. In addition, a black line separates the pictures from each other. From each picture, you should understand the details of this building's environment and describe this building's environment in detail in the form of a summary of these pictures. At this point, do not describe each picture one at a time, but rather in a summarized form. Also note that each picture was taken in a separate location, so successive pictures are not positionally close. Additionally, do not mention which picture you are quoting from or the black line separating each picture."
        response = self.generate_response(results_image, input_text)
        response = response[4:-4]
        return response


    """
    def generate_response(self, image, input_text):
        if 'llama-2' in self.llava_model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.llava_model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.llava_model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        roles = conv.roles if "mpt" not in self.llava_model_name.lower() else ('user', 'assistant')

        image_tensor = self.llava_image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        inp = input_text
        if image is not None:
            if self.llava_model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.llava_model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=2048,
                streamer=streamer,
                use_cache=True,
            )

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        outputs = outputs.replace("\n\n", " ")
        return outputs
    """
    
    
    def get_explored_picture(self, explored, top_down_map):
        explored_map = explored["map"]
        fog_of_war_map = top_down_map["fog_not_clip"]
        start_position = explored["start_position"]
        y, x = explored_map.shape
        print(f"EXPLORED={explored_map.shape}, FOG={fog_of_war_map.shape}")

        for i in range(y):
            for j in range(x):
                if fog_of_war_map[i][j] == 1:
                    if explored_map[i][j] == maps.MAP_VALID_POINT:
                        explored_map[i][j] = maps.MAP_INVALID_POINT
                else:
                    if explored_map[i][j] in [maps.MAP_VALID_POINT, maps.MAP_INVALID_POINT]:
                        explored_map[i][j] = maps.MAP_BORDER_INDICATOR

        range_x = np.where(~np.all(explored_map == maps.MAP_BORDER_INDICATOR, axis=1))[0]
        range_y = np.where(~np.all(explored_map == maps.MAP_BORDER_INDICATOR, axis=0))[0]

        _ind_x_min = range_x[0]
        _ind_x_max = range_x[-1]
        _ind_y_min = range_y[0]
        _ind_y_max = range_y[-1]
        _grid_delta = 3

        explored_map = explored_map[
            _ind_x_min - _grid_delta : _ind_x_max + _grid_delta,
            _ind_y_min - _grid_delta : _ind_y_max + _grid_delta,
        ]
            
        return explored_map, start_position


    def _exec_kachaka(self, date, ip) -> None:
        max_step = 500
        #self.cap = cv2.VideoCapture(0)
        #_, frame = self.cap.read()
        #cv2.imshow('webカメラ', frame)
        #cv2.imwrite('photo.jpg', frame)

        for _ in range(20):
            # フレーム待ち
            frames = self.pipeline.wait_for_frames()
        
        frames = self.align.process(frames)    
        # RGB
        color_frame = frames.get_color_frame()
        # 深度
        depth_frame = frames.get_depth_frame()
            
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = Image.fromarray(color_image)
        
        depth_image = np.asanyarray(depth_frame.get_data())   
        # 2次元データをカラーマップに変換
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = Image.fromarray(depth_colormap)
            
        color_image.save("color.png")
        depth_colormap.save("depth.png")
        print("create")
        # ストリーミング停止
        self.pipeline.stop()
        
        client = kachaka_api.KachakaApiClient(ip)
        client.update_resolver()
        # カチャカにshelfをstartに連れていく
        print("Get the shelf and Go to the Start")
        #sclient.move_shelf("S01", "L01")
        client.move_shelf("S01", "start")
        client.set_auto_homing_enabled(False)
        
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"
            
            
        # evaluate multiple checkpoints in order
        checkpoint_index = 50
        print("checkpoint_index=" + str(checkpoint_index))
        while True:
            checkpoint_path = None
            while checkpoint_path is None:
                checkpoint_path = poll_checkpoint_folder(
                    self.config.EVAL_CKPT_PATH_DIR, checkpoint_index
                )
                print("checkpoint_path=" + str(checkpoint_path))
            print("checkpoint_path=" + str(checkpoint_path))
            logger.info(f"=======current_ckpt: {checkpoint_path}=======")
            checkpoint_index += 1
        
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            print("PATH")
            print(checkpoint_path)

            if self.config.EVAL.USE_CKPT_CONFIG:
                config = self._setup_eval_config(ckpt_dict["config"])
            else:
                config = self.config.clone()

            ppo_cfg = config.RL.PPO

            config.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
            config.freeze()

            if len(self.config.VIDEO_OPTION) > 0:
                config.defrost()
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
                config.freeze()

            logger.info(f"env config: {config}")
            self.env = construct_env(config, client)
            self._setup_actor_critic_agent(ppo_cfg)

            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.actor_critic = self.agent.actor_critic
        
            self._taken_picture_list = []
            self._taken_picture_list.append([])
            
            self.pre_agent_position = None
            self.pre_agent_rotation = None
            
            # LLaVA model
            """
            load_4bit = True
            load_8bit = not load_4bit
            disable_torch_init()
            model_path = "liuhaotian/llava-v1.5-13b"
            self.llava_model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.llava_model, self.llava_image_processor, _ = load_pretrained_model(model_path, None, self.llava_model_name, load_8bit, load_4bit)
            """
            # Load the clip model
            self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
            self._select_threthould = 0.9
        
            observations = self.env.reset()
            self.pre_observations = observations
            batch = batch_obs(observations, device=self.device)

            current_episode_reward = torch.zeros(1, 1, device=self.device)
            current_episode_exp_area = torch.zeros(1, 1, device=self.device)
            current_episode_picsim = torch.zeros(1, 1, device=self.device)
            current_episode_picture_value = torch.zeros(1, 1, device=self.device)
            
            test_recurrent_hidden_states = torch.zeros(
                self.actor_critic.net.num_recurrent_layers,
                self.config.NUM_PROCESSES,
                ppo_cfg.hidden_size,
                device=self.device,
            )
            prev_actions = torch.zeros(
                self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
            )
            not_done_masks = torch.zeros(
                self.config.NUM_PROCESSES, 1, device=self.device
            )
            stats_episodes = dict()  # dict of dicts that stores stats per episode
            raw_metrics_episodes = dict()

            rgb_frames = [[]]  # type: List[List[np.ndarray]]
            if len(self.config.VIDEO_OPTION) > 0:
                os.makedirs(self.config.VIDEO_DIR+"/"+date, exist_ok=True)

            pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
            episode_stats = dict()
            self.actor_critic.eval()
            
            self.step = 0
            while self.step < max_step:
                skip_flag = False
                if self.step % 10 == 0:
                    print("---------------------")
                    client.speak(str(self.step) + "ステップ終了しました。")
                
                self.step+=1
                print(str(self.step) + ": ", end="")

                with torch.no_grad():
                    (
                        _,
                        action,
                        _,
                        test_recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )

                    prev_actions.copy_(action)

                outputs = self.env.step(action[0].item())
    
                observations, rewards, done, infos = outputs
                observations, is_success = observations
                if is_success == False:
                    observations = self.pre_observations
                else:
                    self.pre_observations = observations
                batch = batch_obs(observations, device=self.device)
                
                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0]],
                    dtype=torch.float,
                    device=self.device,
                )
                
                reward = []
                pic_val = []
                picture_value = []
                similarity = []
                pic_sim = []
                exp_area = [] # 探索済みのエリア()
                
                reward.append(rewards[0])
                pic_val.append(rewards[1])
                exp_area.append(rewards[2])
                picture_value.append(0)
                similarity.append(0)
                pic_sim.append(0)
                
                self._taken_picture_list[0].append([pic_val[0], observations["rgb"], rewards[6], rewards[7]])

                reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
                exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
                picture_value = torch.tensor(picture_value, dtype=torch.float, device=self.device).unsqueeze(1)
                similarity = torch.tensor(similarity, dtype=torch.float, device=self.device).unsqueeze(1)
                    
                current_episode_reward += reward
                current_episode_exp_area += exp_area
                    
                # episode continues
                if len(self.config.VIDEO_OPTION) > 0:
                    agent_position = infos["top_down_map"]["agent_map_coord"]
                    agent_rotation=infos["top_down_map"]["agent_angle"],
                    
                    if is_success == False:
                        frame = rgb_frames[0][-1] 
                    else:
                        frame = observations_to_image(observations, infos, action.cpu().numpy())
                    for _ in range(5):
                        rgb_frames[0].append(frame)
                
                if self.step % 50 == 0:
                    # 探索済みの環境の写真を取得
                    explored_picture, start_position = self.get_explored_picture(infos["explored_map"], infos["top_down_map"])
                    explored_picture = explored_to_image(explored_picture, infos)
                    explored_picture = Image.fromarray(np.uint8(explored_picture))
                    
                    #写真の選別
                    self._taken_picture_list[0], picture_value[0] = self._select_pictures(self._taken_picture_list[0])
                    results_image, positions_x, positions_y = self._create_results_image(self._taken_picture_list[0], explored_picture)
                    
                    pic_sim[0] = self._calculate_pic_sim(self._taken_picture_list[0])
                    reward[0] += similarity[0]*10
 
                    # average of picture value par 1 picture
                    current_episode_picture_value[0] += picture_value[0]
                    current_episode_picsim[0] += pic_sim[0]
                            
                    # save description
                    """
                    out_path = os.path.join("output_descriptions/description.txt")
                    with open(out_path, 'a') as f:
                        # print関数でファイルに出力する
                        print(self.step,file=f)
                        print(pred_description,file=f)
                    """
                    
                    # save picture position
                    position_path = os.path.join("position.txt")
                    with open(position_path, 'a') as f:
                        for i in range(len(positions_x)):
                            print(str(self.step) + "-" + str(i) + "," + str(start_position) + "," + str(positions_x[i]) + "," + str(positions_y[i]), file=f)
                            
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[0].item()
                    episode_stats["exp_area"] = current_episode_exp_area[0].item()
                    episode_stats["picture_value"] = current_episode_picture_value[0].item()
                    episode_stats["pic_sim"] = current_episode_picsim[0].item()
                                
                    episode_stats.update(
                        self._extract_scalars_from_info(infos)
                    )
                    
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            "aaa", "aaa"
                        )
                    ] = episode_stats
                                
                    raw_metrics_episodes[
                        "aaa.aaa"
                    ] = infos["raw_metrics"]

                    if len(self.config.VIDEO_OPTION) > 0:
                        if len(rgb_frames[0]) == 0:
                            frame = observations_to_image(observations, infos, action.cpu().numpy())
                            rgb_frames[0].append(frame)
                        picture = rgb_frames[0][-1]
                        for j in range(10):
                            rgb_frames[0].append(picture) 
                        metrics=self._extract_scalars_from_info(infos)    
                        name_sim = -1
                        
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR+"/"+date,
                            images=rgb_frames[0],
                            episode_id="aaa",
                            checkpoint_idx=checkpoint_index,
                            metrics=metrics,
                            name_ci=name_sim,
                        )
                        client.speak("途中経過のビデオを作成しました。")
                        
                        # Save taken picture                     
                        for j in range(len(self._taken_picture_list[0])):
                            value = self._taken_picture_list[0][j][0]
                            picture_name = f"episode=aaa-{len(stats_episodes)}-{j}-{self.step}-{value}"
                            dir_name = "./taken_picture/" + date 
                            os.makedirs(dir_name, exist_ok=True)
                                
                            picture = Image.fromarray(np.uint8(self._taken_picture_list[0][j][1]))
                            file_path = dir_name + "/" + picture_name + ".png"
                            picture.save(file_path)
                            
                        if results_image is not None:
                            results_image.save(f"./taken_picture/{date}/episoede=aaa-{self.step}.png")    
            
            client.speak("エピソードが終了しました。")
                    
            logger.info("Exp Area: " + str(episode_stats["exp_area"]))   
            logger.info("Pic_Sim: " + str(episode_stats["pic_sim"]))
            logger.info("Picture Value: " + str(episode_stats["picture_value"]))

            self.env.close()
            client.return_shelf()
            client.return_home()
            break
            
