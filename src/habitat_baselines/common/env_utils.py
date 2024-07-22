#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
from typing import Type, Union

import habitat
from habitat import Config, Env, RLEnv, make_dataset
from habitat_baselines.common.environments import InfoRLEnv


def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env

def construct_env(config: Config, client) -> InfoRLEnv:
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    """
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    """

    proc_config = config.clone()
    proc_config.defrost()

    task_config = proc_config.TASK_CONFIG
    task_config.SEED = task_config.SEED
    """
    if len(scenes) > 0:
        task_config.DATASET.CONTENT_SCENES = scene_splits[i]
    """

    task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
        config.SIMULATOR_GPU_ID
    )
                             
    task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

    proc_config.freeze()

    env = InfoRLEnv(proc_config, dataset, client)
    return env
