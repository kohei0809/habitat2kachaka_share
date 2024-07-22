#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import math
import time

from enum import Enum

from typing import Any, List, Optional, Union

import pyrealsense2 as rs
from typing import Any, List, Optional, Union

import attr
from gym import Space
from gym import spaces
import numpy as np

#import habitat_sim
from habitat.config import Config
from habitat.core.simulator import Simulator
from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    Observations,
    Sensor,
    SensorSuite,
    SensorTypes,
    Simulator,
)


RGBSENSOR_DIMENSION = 3


def overwrite_config(config_from: Config, config_to: Any) -> None:
    r"""Takes Habitat-API config and Habitat-Sim config structures. Overwrites
     Habitat-Sim config with Habitat-API values, where a field name is present
     in lowercase. Mostly used to avoid `sim_cfg.field = hapi_cfg.FIELD` code.

    Args:
        config_from: Habitat-API config node.
        config_to: Habitat-Sim config structure.
    """

    def if_config_to_lower(config):
        if isinstance(config, Config):
            return {key.lower(): val for key, val in config.items()}
        else:
            return config

    for attr, value in config_from.items():
        if hasattr(config_to, attr.lower()):
            setattr(config_to, attr.lower(), if_config_to_lower(value))


def check_sim_obs(obs, sensor):
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


@registry.register_sensor
class RealRGBSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:        
        # ストリームの設定
        realsense_config = rs.config()
        realsense_config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
        
        # ストリーミング開始
        self.pipeline = rs.pipeline()
        self.pipeline.start(realsense_config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.pre_frames = None
        #self.cap = cv2.VideoCapture(0)
        
        super().__init__(*args, **kwargs)
        
    def __exit__(self):
        cv2.destroyAllWindows()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=0,
            high=np.iinfo(np.int64).max,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self) -> Any:

        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                break
            except:
                print("rgb")
                ctx = rs.context()
                devices = ctx.query_devices()
                for dev in devices:
                    dev.hardware_reset()
                    print(dev)
                print("reset done")
            
        self.pre_frames = frames
        frames = self.align.process(frames)    
        # RGB
        color_frame = frames.get_color_frame()
        
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        size = self.observation_space.shape
        
        rgb_obs = cv2.resize(color_image, size[0:2])
          
        rgb_obs = rgb_obs.astype(np.uint8)
        return rgb_obs


@registry.register_sensor
class RealDepthSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        #self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        config = kwargs["config"]

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH
            
        #self.cap = cv2.VideoCapture(0)
        
        # ストリームの設定
        realsense_config = rs.config()
        realsense_config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)

        # ストリーミング開始
        self.pipeline = rs.pipeline()
        self.pipeline.start(realsense_config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.pre_frames = None
        super().__init__(*args, **kwargs)
        
    def __exit__(self):
        cv2.destroyAllWindows()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(self):
        """
        while True:
            flag, frames = self.pipeline.try_wait_for_frames()
            if flag == True: 
                break
        """
        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                break
            except:
                print("depth")
                ctx = rs.context()
                devices = ctx.query_devices()
                for dev in devices:
                    dev.hardware_reset()
                    print(dev)
                print("reset done")
            
        frames = self.align.process(frames)  
        # 深度
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())     
        depth_image = depth_image/1000
            
        size = self.observation_space.shape
        obs = cv2.resize(depth_image, size[0:2])
        obs = obs[:,:,np.newaxis]
          
        obs = obs.astype(np.float64)
        return obs

@registry.register_simulator(name="Real-v0")
class RealWorld(Simulator):
    # 実世界実験用のSimulatorクラス
    
    def __init__(self, config: Config, client) -> None:
        self.config = config
        self._client = client

        sim_sensors = []
        
        print(config)
        if "RGB_SENSOR" in config.AGENT_0.SENSORS:
            sim_sensors.append(RealRGBSensor(config=self.config.RGB_SENSOR))
        if "DEPTH_SENSOR" in config.AGENT_0.SENSORS:
            sim_sensors.append(RealDepthSensor(config=self.config.DEPTH_SENSOR))

        self._sensor_suite = SensorSuite(sim_sensors)
        
        self.is_reset_postion = False
        self.x = None
        self.y = None
        self.z = None
        self.theta_rad = None

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def reset(self) -> Observations:
        return self._sensor_suite.get_observations()

    def seed(self, seed: int) -> None:
        return

    def geodesic_distance(
        self,
        position_a: List[float],
        position_b: Union[List[float], List[List[float]]],
    ) -> float:
        r"""Calculates geodesic distance between two points.

        :param position_a: coordinates of first point.
        :param position_b: coordinates of second point or list of goal points
        coordinates.
        :param episode: The episode with these ends points.  This is used for shortest path computation caching
        :return:
            the geodesic distance in the cartesian space between points
            :p:`position_a` and :p:`position_b`, if no path is found between
            the points then `math.inf` is returned.
        """
        x_a, y_a = position_a
        x_b, y_b = position_b
        distance = math.sqrt((x_a-x_b)*(x_a-x_b) + (y_a-y_b)*(y_a-y_b))
        return distance

    def get_agent_state(self):
        if self.is_reset_postion == False:
            pos = self._client.get_robot_pose()
            self.x = pos.x
            self.y = pos.y
            self.z = 0.0
            self.theta_rad = pos.theta + math.pi/2
        
        return {"position":[self.x, self.z, self.y], "rotation": self.theta_rad}

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config
    
    
    def get_observations_at(self) -> Optional[Observations]:
        observations = self._sensor_suite.get_observations()
        return observations
    
    def reset_position(self) -> None:
        self.is_reset_postion = True
        #print("####### 自己位置をリセットしました ##########")
        
    def unreset_position(self) -> None:
        self.is_reset_postion = False
        #print("####### 自己位置を動かします ##########")
