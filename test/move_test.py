import sys
import os
import time

import numpy as np
import math

import kachaka_api
from utils import move
sys.path.append(f"/Users/{os.environ['USER']}/Desktop/habitat2kachaka/kachaka-api/python/")


if __name__ == "__main__":
    client = kachaka_api.KachakaApiClient(sys.argv[1])
    client.update_resolver()
    #print(client.get_locations())
    
    client.set_auto_homing_enabled(False)
    
    #client.move_shelf("S01", "L01")
    # startに移動する
    move(client, 1.5, -2.5, 0.15)
    time.sleep(2)
    
    
    # 手動で動かせるようにする
    client.set_manual_control_enabled(True)
    print("########## pose 1 ############")
    location1 = client.get_robot_pose()
    print(location1)
    x_1 = location1.x
    y_1 = location1.y
        
    start_time = time.time()
    while time.time() - start_time < 4:
        client.set_robot_velocity(linear=0.25/5, angular=0.0) # 0.3 [m/s], 0.0 [rad/s]で進む
        time.sleep(0.01)
    
    time.sleep(2)
    location2 = client.get_robot_pose()
    print("########## pose 2 ############")
    print(location2)
    x_2 = location2.x
    y_2 = location2.y
    theta_2_rad = location2.theta
    theta_2_deg = math.degrees(theta_2_rad)
    
    diff_meter = np.sqrt((x_2-x_1)*(x_2-x_1) + (y_2-y_1)*(y_2-y_1))
    
    start_time = time.time()
    while time.time() - start_time < 3:
        angle = math.radians(30)
        client.set_robot_velocity(linear=0.0, angular=-angle/10) # 0.3 [rad /s]で回転する
        time.sleep(0.01)
        
    time.sleep(2)
    
    location3 = client.get_robot_pose()
    print("########## pose 3 ############")
    print(location3)
    x_3 = location3.x
    y_3 = location3.y
    theta_3_rad = location3.theta
    theta_3_deg = math.degrees(theta_3_rad)
    
    
    diff_rad = theta_3_rad - theta_2_rad
    diff_deg = theta_3_deg - theta_2_deg
    
    print("DIFF_METER: " + str(diff_meter) + " [m]")
    print("DIFF_rad: " + str(diff_rad) + " [rad]")
    print("DIFF_DEG: " + str(diff_deg) + " [度]")
    