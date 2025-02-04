#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union
import pickle
import torch
from scipy import ndimage, misc
import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict
from habitat import config

from utils.log_manager import LogManager
from utils.log_writer import LogWriter

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
from habitat.tasks.nav.maximum_info_task import MaximumInformationTask
from habitat_baselines.common.utils import quat_from_angle_axis
from habitat.sims.habitat_simulator.real_world import RealWorld
from habitat.utils.visualizations.maps import get_sem_map

import matplotlib.pyplot as plt

from PIL import Image
def display_sample(rgb_obs):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    arr = [rgb_img]
    plt.imshow(rgb_img)
    plt.show()



class Env:
    r"""Fundamental environment class for `habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied tasks with simulator
    is abstracted inside `Env`. Acts as a base for other derived environment
    classes. `Env` consists of three major components: ``dataset`` (`episodes`), ``simulator`` (`sim`) and `task` and connects all the three components
    together.
    """

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Type[Episode]]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None, client = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            
            "environment, use config.freeze()."
        )
        self._config = config
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        self._episodes = self._dataset.episodes if self._dataset else []
        self._current_episode = None
        self.client = client

        # load the first scene if dataset is present
        """
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._config.defrost()
            self._config.SIMULATOR.SCENE = self._dataset.episodes[0].scene_id
            self._config.freeze()
        """
        """
        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        """
        self._sim = RealWorld(config=self._config.SIMULATOR, client=client)
        self._task = MaximumInformationTask(self._config.TASK, self._sim, client)
        
        
        make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Type[Episode]) -> None:
        self._current_episode = episode

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False
    

    def conv_grid(self, realworld_x, realworld_y):
        map = self.client.get_png_map()
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

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()
    
        #assert len(self.episodes) > 0, "Episodes list is empty"

        #############################################
        # current_episodeについて
        #raise NotImplementedError
        #self._current_episode = next(self._episode_iterator)
        ############################################
            
        # Insert object here
        # ここは後で要変更
        """
        raise NotImplementedError
        object_to_datset_mapping = {'cylinder_red':0, 'cylinder_green':1, 'cylinder_blue':2,
            'cylinder_yellow':3, 'cylinder_white':4, 'cylinder_pink':5, 'cylinder_black':6, 'cylinder_cyan':7
        }
        for i in range(len(self.current_episode.goals)):
            current_goal = self.current_episode.goals[i].object_category
            dataset_index = object_to_datset_mapping[current_goal]
            
            raise NotImplementedError
            # オブジェクトの挿入
            ind = self._sim._sim.add_object(dataset_index)
            self._sim._sim.set_translation(np.array(self.current_episode.goals[i].position), ind)
        """

        #observations = self.task.reset(episode=self.current_episode)
        observations = self.task.reset()


        if self._config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            # mapの取得
            self.currMap  = get_sem_map(sim=self.sim, client=self.client)
            self.task.occMap = self.currMap
            self.task.sceneMap = self.currMap
            self.task.set_top_down_map(self.currMap)


        self._task.measurements.reset_measures(
            #episode=self.current_episode, task=self.task
            task=self.task, client=self.client
        )

        if self._config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            robot_pose = self.client.get_robot_pose()
            currPix = self.conv_grid(robot_pose.x, robot_pose.y)  ## Explored area marking

            self.expose = self.task.measurements.measures["fow_map"].get_metric()
                
            #print("EXPOSE: " + str(self.expose.shape))
            #print("currMap: " + str(self.currMap.shape))
            patch = self.currMap * self.expose

            patch = patch[currPix[1]-40:currPix[1]+40, currPix[0]-40:currPix[0]+40]
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"] * 180/np.pi) + 90, order=0, reshape=False)
            
            
            # padding
            
            #print("semMap: " + str(patch.shape))
            patch_ = np.zeros((80, 80))
            for i in range(patch.shape[0]): 
                for j in range(patch.shape[1]):
                    patch_[i][j] = patch[i][j]
            #print("semMap_after: " + str(patch_.shape))
            
            #center_x = int(patch_.shape[0]/2)
            #center_y = int(patch_.shape[1]/2)
            center_x = 40
            center_y = 40
            
            observations["semMap"] = patch_[center_y-25:center_y+25, center_x-25:center_x+25]
        return observations
    

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:

        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if isinstance(action, str) or isinstance(action, (int, np.integer)):
            action = {"action": action}

        observations, is_success = self.task.step(action=action)

        self._task.measurements.update_measures(
           action=action, task=self.task
        )

        if self._config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            robot_pose = self.client.get_robot_pose()
            currPix = self.conv_grid(robot_pose.x, robot_pose.y)  ## Explored area marking
            
            if self._config.TRAINER_NAME == "oracle-ego":
                self.expose = self.task.measurements.measures["fow_map"].get_metric()
                
                patch = self.currMap * self.expose
            elif self._config.TRAINER_NAME == "oracle":
                patch = self.currMap   
                
            patch = patch[currPix[1]-40:currPix[1]+40, currPix[0]-40:currPix[0]+40]
            #patch = ndimage.interpolation.rotate(patch, -(observations["heading"] * 180/np.pi) + 90, order=0, reshape=False)
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"] * 180/np.pi) - 90, order=0, reshape=False)
            
            """
            patch_ = np.zeros((80, 80))
            for i in range(patch.shape[0]): 
                for j in range(patch.shape[1]):
                    patch_[i][j] = patch[i][j]
            #print("semMap_after: " + str(patch_.shape))
            """
            #center_x = int(patch_.shape[0]/2)
            #center_y = int(patch_.shape[1]/2)
            center_x = 40
            center_y = 40
            
            #observations["semMap"] = patch_[center_x-25:center_x+25, center_y-25:center_y+25]
            observations["semMap"] = patch[40-25:40+25, 40-25:40+25]
            
            log_manager = LogManager()
            log_manager.setLogDirectory("semMap")
            log_writer = log_manager.createLogWriter("semMap")
            for i in range(observations["semMap"].shape[0]):
                for j in range(observations["semMap"].shape[1]):
                    log_writer.write(str(observations["semMap"][i][j]))
                log_writer.writeLine()     

        self._update_step_stats()
        return observations, is_success

    def seed(self, seed: int) -> None:
        self._sim.seed(seed)
        self._task.seed(seed)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        #raise NotImplementedError
        #self._sim.close()
        pass


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over `Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: `get_reward_range()`, `get_reward()`, `get_done()`, `get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None, client=None
    ) -> None:
        """Constructor

        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """
        self._config = config
        self._client = client
        self._env = Env(config, dataset, client)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._env.episodes

    @property
    def current_episode(self) -> Type[Episode]:
        return self._env.current_episode

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        self._env.episodes = episodes

    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the `step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations, is_success = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations, **kwargs)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return [observations, is_success], reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        #raise NotImplementedError
        #self._env.close()
        pass
