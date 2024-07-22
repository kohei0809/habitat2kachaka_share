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
sys.path.append(f"/home/{os.environ['USER']}/Desktop/habitat2kachaka_share/kachaka-api/python/")



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
        # RGB画像の取得
        color_frame = frames.get_color_frame()
        # Depth画像の取得
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
                exp_area = [] # 探索済みのエリア()
                
                reward.append(rewards[0])
                exp_area.append(rewards[1])

                reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
                exp_area = torch.tensor(exp_area, dtype=torch.float, device=self.device).unsqueeze(1)
                    
                current_episode_reward += reward
                current_episode_exp_area += exp_area
                    
                # episode continues
                if len(self.config.VIDEO_OPTION) > 0:
                    if is_success == False:
                        frame = rgb_frames[0][-1] 
                    else:
                        frame = observations_to_image(observations, infos, action.cpu().numpy())
                    for _ in range(5):
                        rgb_frames[0].append(frame)
                
                if self.step % 50 == 0:
                   # 途中経過のログ出力・ビデオ作成
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[0].item()
                    episode_stats["exp_area"] = current_episode_exp_area[0].item()
                                
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
            
            client.speak("エピソードが終了しました。")
                    
            logger.info("Exp Area: " + str(episode_stats["exp_area"]))   

            self.env.close()
            client.return_shelf()
            client.return_home()
            break
            
