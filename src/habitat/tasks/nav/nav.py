#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type, Union

import os
import attr
import numpy as np
from gym import spaces
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torchvision import transforms
from scipy import stats
from scipy.ndimage import label

import multiprocessing
import time
import threading
import json

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    Action,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps


cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 1250


def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


def move(client, x, y, theta, _sim):
    print(f"Moving to (x, y, theta)=({x}, {y}, {theta}) ...")
    
    while True:
        get_thread = threading.Thread(target=client.move_to_pose, args=(x, y, theta))
        get_thread.start()
        get_thread.join(10)  # get関数の終了を待つ（最大10秒）

        if get_thread.is_alive():
            print("move関数は10秒以上かかりました。強制終了します。")
            client.speak("move関数は10秒以上かかりました。強制終了します。")
            _sim.reset_position()
            return False
        
        #client.move_to_pose(x, y, theta)
        _sim.unreset_position()
        result = client.get_last_command_result()[0]
        if result.success:
            #print("Success!")
            return True
        else:
            with open(f"/home/{os.environ['USER']}/habitat2kachaka/kachaka-api/docs/KachakaErrorCode.json") as f:
                error_codes = json.load(f)
            for error_code in error_codes:
                if int(error_code["code"]) == result.error_code:
                    error_title = error_code["title"]
                    error_description = error_code["description"]
                    print(f"Failure: {error_title}\n{error_description}")
            client.speak("再実行します。")


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius.
    """

    room_id: str = attr.ib(default=None, validator=not_none_validator)
    room_name: Optional[str] = None

@attr.s(auto_attribs=True, kw_only=True)
class MaximumInformationEpisode(Episode):
    start_room: Optional[str] = None
    
    shortest_paths: Optional[List[ShortestPathPoint]] = None
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    object_category: Optional[List[str]] = None
    object_index: Optional[int] = None
    currGoalIndex: Optional[int] = 0 

@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.
    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    start_room: Optional[str] = None
    shortest_paths: Optional[List[ShortestPathPoint]] = None
    
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    object_category: Optional[List[str]] = None
    object_index: Optional[int] = None
    currGoalIndex: Optional[int] = 0 


@registry.register_sensor
class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.
            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]
    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim

        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pointgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        source_position = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            source_position, rotation_world_start, goal_position
        )


@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.
            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]
    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pointgoal_with_gps_compass"

    def get_observation(
        self, *args: Any, observations, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


@registry.register_sensor(name="PositionSensor")
class AgentPositionSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, **kwargs: Any
    ):
        return self._sim.get_agent_state()["position"]


@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state["rotation"]

        #return self._quat_to_xy_heading(rotation_world_agent.inverse())
        return rotation_world_agent


@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "compass"

    def get_observation(
        self, *args: Any, observations, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state["rotation"]
        rotation_world_start = rotation_world_agent

        """
        return self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )
        """
        return rotation_world_agent


@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state["position"]
        agent_rotation = agent_state["rotation"]
        
        origin = np.array(agent_position, dtype=np.float32)
        rotation_world_start = agent_rotation
        """
        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        """
        return agent_position[0], agent_position[2]


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "proximity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=float,
        )

    def get_observation(
        self, observations, *args: Any, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        return self._sim.distance_to_closest_obstacle(
            current_position, self._max_detection_radius
        )


@registry.register_measure
class RawMetrics(Measure):
    """All the raw metrics we might need
    """
    cls_uuid: str = "raw_metrics"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state()["position"]

        self._start_end_episode_distance = 0

        self._agent_episode_distance = 0.0
        self._metric = None

        self.update_metric(*args, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state()["position"]
        ###########################################
        # 距離について
        #raise NotImplementedError
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        #################################
        self._previous_position = current_position
        
        self._metric = {
            'agent_path_length': self._agent_episode_distance,
            'episode_lenth': task.measurements.measures[EpisodeLength.cls_uuid].get_metric()
        }


#@registry.register_measure
class STEPS(Measure):
    r"""Count for steps taken
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "wpl"

    def reset_metric(self, *args: Any, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        raise NotImplementedError
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self.update_metric(*args, task=task, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self, *args: Any, task: EmbodiedTask, **kwargs: Any
    ):
        #############################################
        # 現在位置の取得
        raise NotImplementedError
        current_position = self._sim.get_agent_state().position.tolist()
        #############################################
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ###########################################
        # 距離について
        """
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        """
        self._agent_episode_distance = 0.0
        ##########################################

        self._previous_position = current_position

        self._metric = 1 * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._shortest_path_points = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        self.line_thickness = int(
            np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map"

    def get_original_map(self, client=None):
        top_down_map, self._ind_x_min, self._ind_x_max, self._ind_y_min, self._ind_y_max = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            client,        
        )
        
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map
    
    def _clip_map(self, _map, fog=False):
        return _map[
            self._ind_y_min - self._grid_delta : self._ind_y_max + self._grid_delta,
            self._ind_x_min - self._grid_delta : self._ind_x_max + self._grid_delta,
        ]


    def reset_metric(self, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map(kwargs["client"])
        agent_position = self._sim.get_agent_state()["position"]
        a_x, a_y = maps.to_grid(self.client, agent_position[0], agent_position[2])
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target parts last to avoid overlap
        
        ##########################################
        # mapにオブジェクトの位置を記入する
        """
        raise NotImplementedError
        self._draw_goals_view_points(episode)
        self._draw_goals_aabb(episode)
        self._draw_goals_positions(episode)
        """
        ############################################

        ########################################
        # startの場所をmapに記入する
        
        if self._config.DRAW_SOURCE:
            self._draw_point(
                agent_position, maps.MAP_SOURCE_POINT_INDICATOR
            )
            
        self.update_metric(None, None)

    def update_metric(self, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state()["position"]
        )

        clipped_house_map = self._clip_map(house_map)
        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        

        clipped_fog_of_war_mask = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_mask = self._clip_map(self._fog_of_war_mask)

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_mask,
            "agent_map_coord": (
                map_agent_y - (self._ind_y_min - self._grid_delta),
                map_agent_x - (self._ind_x_min - self._grid_delta),
            ),
            "agent_angle": self._sim.get_agent_state()["rotation"],
            "fog_not_clip": self._fog_of_war_mask
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state["rotation"]

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip
    
    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(self.client, position[2], position[0])    
        
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(self.client, agent_position[0], agent_position[2])    
        print(f"#######({a_x}, {a_y})##############")
        """
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = int(
                np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
            )
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )
        """

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                -self._sim.get_agent_state()["rotation"]+math.pi/2,
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )
            

@registry.register_measure
class FowMap(Measure):
    r"""FOW map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._map_resolution = (1250, 1250)
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "fow_map"

    def reset_metric(self, *args: Any, task, **kwargs: Any):
        self._metric = None
        self._top_down_map = task.sceneMap
        self._fog_of_war_mask = np.zeros_like(self._top_down_map)
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        agent_position = self._sim.get_agent_state()["position"]
        a_x, a_y = maps.to_grid(self.client, agent_position[0], agent_position[2])
        agent_position = np.array([a_x, a_y])

        self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
            self._top_down_map,
            self._fog_of_war_mask,
            agent_position,
            -self._sim.get_agent_state()["rotation"]+math.pi/2,
            fov=self._config.FOV,
            max_line_len=self._config.VISIBILITY_DIST
            * max(self._map_resolution)
            / (self._coordinate_max - self._coordinate_min),
        )

        self._metric = self._fog_of_war_mask

@registry.register_measure
class DistanceToMultiGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_multi_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, task, *args: Any, **kwargs: Any):
        self._metric = None
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state()["position"]

        if self._config.DISTANCE_TO == "POINT":
            distance_to_target = []
            ###########################
            for goal_number in range(3):
                ########################################
                # ２点間の距離についての関数を定義する
                #raise NotImplementedError
                distance_to_target.append(0.0)
                """
                distance_to_target.append(self._sim.geodesic_distance(
                    current_position, episode.goals[goal_number].position
                ))
                """
                ######################################
            
        else:
            logger.error(
                f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
            )

        self._metric = distance_to_target


@registry.register_measure
class EpisodeLength(Measure):
    r"""Calculates the episode length
    """
    cls_uuid: str = "episode_length"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._episode_length = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, task, **kwargs: Any):
        self._episode_length = 0
        self._metric = self._episode_length

    def update_metric(
        self, *args: Any, task: EmbodiedTask, **kwargs: Any
    ):
        self._episode_length += 1
        self._metric = self._episode_length



@registry.register_task_action
class MoveForwardAction(Action):
    name: str = "MOVE_FORWARD"

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False 
        
        pos = self._client.get_robot_pose()
        x = pos.x
        y = pos.y
        theta_rad = pos.theta
        
        delta_x = self._meter * math.cos(theta_rad)
        delta_y = self._meter * math.sin(theta_rad)   
        print("MOVE_FORWARD " + str(self._meter) + "[m]")
        
        is_success = move(self._client, x+delta_x, y+delta_y, theta_rad, self._sim)
        return self._sim.get_observations_at(), is_success

    def set_client(self, client):
        self._client = client
        self._meter = 0.25

@registry.register_task_action
class TurnLeftAction(Action):
    def step(self, *args: Any,  task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False 
        
        pos = self._client.get_robot_pose()
        x = pos.x
        y = pos.y
        theta_rad = pos.theta
        
        angle = math.radians(self._angle)
        print("TURN_LEFT " + str(self._angle) + "[度]")
        
        is_success = move(self._client, x, y, theta_rad+angle, self._sim)
        return self._sim.get_observations_at(), is_success
    
    def set_client(self, client):
        self._client = client
        self._angle = 30


@registry.register_task_action
class TurnRightAction(Action):
    def step(self, *args: Any,  task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = False
        
        pos = self._client.get_robot_pose()
        x = pos.x
        y = pos.y
        theta_rad = pos.theta
        angle = math.radians(-self._angle)
        print("TURN_RIGHT " + str(-self._angle) + "[度]")
        
        is_success = move(self._client, x, y, theta_rad+angle, self._sim)
        return self._sim.get_observations_at(), is_success
    
    def set_client(self, client):
        self._client = client
        self._angle = 30


@registry.register_task_action
class TakePicture(Action):
    name: str = "TAKE_PICTURE"

    def reset(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.is_stop_called = False
        task.is_found_called = False ##C

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = True
        print("TAKE_PICTURE")
        self._client.speak("写真を撮りました。")
        return self._sim.get_observations_at()
    
    def set_client(self, client):
        self._client = client

    
@registry.register_task(name="Info-v0")
class InformationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, client=None, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, client=client, dataset=dataset)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
    
    
@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)