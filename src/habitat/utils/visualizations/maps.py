#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, Tuple

from PIL import Image
import imageio
import numpy as np
import scipy.ndimage
import io

from habitat.core.simulator import Simulator
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import utils

from utils.log_manager import LogManager
from utils.log_writer import LogWriter

cv2 = try_cv2_import()


AGENT_SPRITE = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_agent_sprite",
        "100x100.png",
    )
)
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))
COORDINATE_EPSILON = 1e-6
COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
#MAP_TARGET_POINT_INDICATOR = 6
MAP_TARGET_POINT_INDICATOR = 10
MAP_SHORTEST_PATH_COLOR = 7
MAP_VIEW_POINT_INDICATOR = 8
MAP_TARGET_BOUNDING_BOX = 9
TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[10:] = cv2.applyColorMap(
    np.arange(246, dtype=np.uint8), cv2.COLORMAP_JET
).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]  # White
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [150, 150, 150]  # Light Grey
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]  # Grey
TOP_DOWN_MAP_COLORS[MAP_SOURCE_POINT_INDICATOR] = [0, 0, 200]  # Blue
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR+1] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR+2] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_COLOR] = [0, 200, 0]  # Green
TOP_DOWN_MAP_COLORS[MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]  # Light Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_BOUNDING_BOX] = [0, 175, 0]  # Green


def draw_agent(
    image: np.ndarray,
    agent_center_coord: Tuple[int, int],
    agent_rotation: float,
    agent_radius_px: int = 5,
) -> np.ndarray:
    r"""Return an image with the agent image composited onto it.
    Args:
        image: the image onto which to put the agent.
        agent_center_coord: the image coordinates where to paste the agent.
        agent_rotation: the agent's current rotation in radians.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
    Returns:
        The modified background image. This operation is in place.
    """

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_SPRITE, agent_rotation * 180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = AGENT_SPRITE.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    utils.paste_overlapping_image(image, resized_agent, agent_center_coord)
    return image


def to_grid(client, realworld_x, realworld_y) -> Tuple[int, int]:
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


def from_grid(
    grid_x: int,
    grid_y: int,
    coordinate_min: float,
    coordinate_max: float,
    grid_resolution: Tuple[int, int],
) -> Tuple[float, float]:
    r"""Inverse of _to_grid function. Return real world coordinate from
    gridworld assuming top-left corner is the origin. The real world
    coordinates of lower left corner are (coordinate_min, coordinate_min) and
    of top right corner are (coordinate_max, coordinate_max)
    """
    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )
    realworld_x = coordinate_max - grid_x * grid_size[0]
    realworld_y = coordinate_min + grid_y * grid_size[1]
    return realworld_x, realworld_y

def resize_map(map):
    h = len(map)
    w = len(map[0])
    size = 3
    
    resized_map = np.zeros((int(h/size), int(w/size)))
    #print("h=" + str(h) + ", w=" + str(w) + ", h'=" + str(len(resized_map)) + ", w'=" + str(len(resized_map[0])))

    # mapのresize
    for i in range(len(resized_map)):
        for j in range(len(resized_map[0])):
            flag = False
            num_0 = 0
            num_1 = 0
            for k in range(size):
                if flag == True:
                    break
                
                if size*i+k >= h:
                    break
                
                for l in range(size):
                    if size*j+l >= w:
                        break
                    if map[size*i+k][size*j+l] == 2:
                        resized_map[i][j] = 2
                        flag = True
                    elif map[size*i+k][size*j+l] == 0:
                        num_0 += 1
                    elif map[size*i+k][size*j+l] == 1:
                        num_1 += 1
                        
            if flag == False:
                if num_0 > num_1:
                    resized_map[i][j] = 0
                else:
                    resized_map[i][j] = 1
            
    # borderをちゃんと作る
    for i in range(len(resized_map)):
        for j in range(len(resized_map[0])):
            flag = False
            if resized_map[i][j] == 1:
                for k in [-1, 1]:
                    if flag == True:
                        break
                    if i+k < 0 or i+k >= len(resized_map):
                        continue
                    for l in [-1, 1]:
                        if j+l < 0 or j+l >= len(resized_map[0]):
                            continue
                        if resized_map[i+k][j+l] == 0:
                            resized_map[i][j] = 2
                            flag = True
                            break
    return resized_map

def recreate_map(map_data):
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            # 村田研究室ゾーン
            if i > 175:
                map_data[i][j] = 0     
    
    # 境界線をちゃんと作る
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            flag = False
            if map_data[i][j] == 2:
                for k in [-1, 1]:
                    if flag == True:
                        break
                    if i+k < 0 or i+k >= map_data.shape[0]:
                        continue
                    for l in [-1, 1]:
                        if j+l < 0 or j+l >= map_data.shape[1]:
                            continue
                        if map_data[i+k][j+l] == 0:
                            map_data[i][j] = 2
                            flag = True
                            break
    """    
    map_manager = LogManager()
    map_manager.setLogDirectory("Map")
    map_logger = map_manager.createLogWriter("map_logger")
    
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            map_logger.write(map_data[i][j])
        map_logger.writeLine()
    """
                       
    return map_data
    
def clip_map(map):
    grid_delta = 3
    range_x = np.where(np.any(map != 0, axis=0))[0]
    range_y = np.where(np.any(map != 0, axis=1))[0]
    
    ind_x_min = range_x[0]
    ind_x_max = range_x[-1]
    ind_y_min = range_y[0]
    ind_y_max = range_y[-1]

    return ind_x_min, ind_x_max, ind_y_min, ind_y_max    
        
def create_map(map_data):
    h = len(map_data)
    w = len(map_data[0])
    map = np.zeros((h, w))
    
    for i in range(h):
        for j in range(w):
            # 不可侵領域
            if map_data[i][j][0] == 244:
                map[i][j] = 0
            elif map_data[i][j][0] == 191:
                map[i][j] = 2
            elif map_data[i][j][0] == 253:
                map[i][j] = 1
            else:
                map[i][j] = -1
                
    return map

def get_topdown_map(
    sim: Simulator,
    map_resolution: Tuple[int, int] = (1250, 1250),
    client = None
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.

    Args:
        sim: The simulator.
        map_resolution: The resolution of map which will be computed and
            returned.
        num_samples: The number of random navigable points which will be
            initially
            sampled. For large environments it may need to be increased.
        draw_border: Whether to outline the border of the occupied spaces.

    Returns:
        Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """
    
    map = client.get_png_map()
    map_img = Image.open(io.BytesIO(map.data))
    
    data = map_img.getdata()
    map_data = np.array(data) 
    
    # 分割サイズ
    chunk_size = map.width
    # 分割
    map_data = np.array(np.array_split(map_data, range(chunk_size, len(map_data), chunk_size), axis=0))
    
    map_data = create_map(map_data)
    map_data = resize_map(map_data)
    map_data = recreate_map(map_data)
    ind_x_min, ind_x_max, ind_y_min, ind_y_max = clip_map(map_data)

    top_down_map = map_data.astype(int)    
    
    return top_down_map, ind_x_min, ind_x_max, ind_y_min, ind_y_max


def get_sem_map(
    sim: Simulator,
    map_resolution: Tuple[int, int] = (1250, 1250),
    client = None
) -> np.ndarray:
    map = client.get_png_map()
    map_img = Image.open(io.BytesIO(map.data))
    
    data = map_img.getdata()
    map_data = np.array(data) 
    
    # 分割サイズ
    chunk_size = map.width
    # 分割
    map_data = np.array(np.array_split(map_data, range(chunk_size, len(map_data), chunk_size), axis=0))
    
    map_data = create_sem_map(map_data)
    sem_map = resize_sem_map(map_data)
    sem_map = recreate_sem_map(sem_map)
    return sem_map

def create_sem_map(map_data):
    h = len(map_data)
    w = len(map_data[0])
    map = np.zeros((h, w))
    
    for i in range(h):
        for j in range(w):
            # 不可侵領域
            if map_data[i][j][0] == 244:
                map[i][j] = 2
            elif map_data[i][j][0] == 191:
                map[i][j] = 3
            elif map_data[i][j][0] == 253:
                map[i][j] = 3
            else:
                map[i][j] = -1
                
    return map
    
def resize_sem_map(map):
    h = len(map)
    w = len(map[0])
    size = 3
    resized_map = np.zeros((int(h/size), int(w/size)))
    
    # mapのresize
    for i in range(len(resized_map)):
        for j in range(len(resized_map[0])):
            num_2 = 0
            num_3 = 0
            for k in range(size):
                if size*i+k >= h:
                    break
                for l in range(size):
                    if size*j+l >= w:
                        break   
                    if map[size*i+k][size*j+l] == 2:
                        num_2 += 1
                    elif map[size*i+k][size*j+l] == 3:
                        num_3 += 1

            if num_2 > num_3:
                resized_map[i][j] = 2
            else:
                resized_map[i][j] = 3            
              
    return resized_map

def recreate_sem_map(map_data):
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            # 村田研究室ゾーン
            if i>175:
                map_data[i][j] = 0
    return map_data


def colorize_topdown_map(
    top_down_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray] = None,
    fog_of_war_desat_amount: float = 0.5,
) -> np.ndarray:
    r"""Convert the top down map to RGB based on the indicator values.
        Args:
            top_down_map: A non-colored version of the map.
            fog_of_war_mask: A mask used to determine which parts of the
                top_down_map are visible
                Non-visible parts will be desaturated
            fog_of_war_desat_amount: Amount to desaturate the color of unexplored areas
                Decreasing this value will make unexplored areas darker
                Default: 0.5
        Returns:
            A colored version of the top-down map.
    """
    _map = TOP_DOWN_MAP_COLORS[top_down_map]

    if fog_of_war_mask is not None:
        fog_of_war_desat_values = np.array([[fog_of_war_desat_amount], [1.0]])
        # Only desaturate things that are valid points as only valid points get revealed
        desat_mask = top_down_map != MAP_INVALID_POINT

        _map[desat_mask] = (
            _map * fog_of_war_desat_values[fog_of_war_mask]
        ).astype(np.uint8)[desat_mask]

    return _map


def colorize_explored_map(
    explored_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray] = None,
    fog_of_war_desat_amount: float = 0.5,
) -> np.ndarray:
    _map = TOP_DOWN_MAP_COLORS[explored_map]
    return _map


def draw_path(
    top_down_map: np.ndarray,
    path_points: List[Tuple],
    color: int,
    thickness: int = 2,
) -> None:
    r"""Draw path on top_down_map (in place) with specified color.
        Args:
            top_down_map: A colored version of the map.
            color: color code of the path, from TOP_DOWN_MAP_COLORS.
            path_points: list of points that specify the path to be drawn
            thickness: thickness of the path.
    """
    for prev_pt, next_pt in zip(path_points[:-1], path_points[1:]):
        # Swapping x y
        cv2.line(
            top_down_map,
            prev_pt[::-1],
            next_pt[::-1],
            color,
            thickness=thickness,
        )
