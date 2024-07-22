#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
sys.path.insert(0, "")
import argparse
import pathlib
import random
import datetime
import numpy as np
import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config    

def main():
    exp_config = "habitat_baselines/config/maximuminfo/ppo_maximuminfo.yaml"
    agent_type = "oracle-ego"
    run_type = "eval"
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    
    if run_type == "eval":
        # 学習済みモデルの日付
        datadate = "24-06-30 03-48-07"

    config = get_config(exp_config)
    
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    
    config.defrost()
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.CHECKPOINT_FOLDER = "cpt/" + start_date
    config.EVAL_CKPT_PATH_DIR = "cpt/" + datadate 
    config.VIDEO_OPTION = ["disk"]
    config.freeze()
    
    if agent_type in ["oracle", "oracle-ego", "no-map"]:
        trainer_init = baseline_registry.get_trainer("oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512 if agent_type=="no-map" else 768
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.5
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
        config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
        if agent_type == "oracle-ego":
            config.TASK_CONFIG.TASK.MEASUREMENTS.append('FOW_MAP')
        config.freeze()
    else:
        trainer_init = baseline_registry.get_trainer("non-oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512
        config.freeze()
        
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("-----------------------------------")
    print("device:" + str(device))
    print("-----------------------------------")

    
    # kachakaのipアドレス
    ip = "192.168.100.47:26400"
    trainer._exec_kachaka(start_date, ip)
       
    end_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    print("Start at " + start_date)
    print("End at " + end_date)

if __name__ == "__main__":
    main()
