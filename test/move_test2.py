import sys
import os
import time

import numpy as np
import math

import kachaka_api
from utils import move
sys.path.append(f"/Users/{os.environ['USER']}/Desktop/habitat2kachaka/kachaka-api/python/")


# 座標を指定する(一番良い結果)
if __name__ == "__main__":
    client = kachaka_api.KachakaApiClient(sys.argv[1])
    client.update_resolver()
    #print(client.get_locations())
    
    client.move_shelf("S01", "L01")
    
    client.set_auto_homing_enabled(False)
    # startに移動する
    move(client, 1.5, -2.5, 0.15)
    time.sleep(2)
    
    print("########## pose 1 ############")
    location1 = client.get_robot_pose()
    print(location1)
    x_1 = location1.x
    y_1 = location1.y
    theta_1_rad = location1.theta
    theta_1_deg = math.degrees(theta_1_rad)
        
    x = 0.25 * math.cos(theta_1_rad)
    y =  0.25 * math.sin(theta_1_rad)   
    print("X: "+ str(x) + ", Y: " + str(y))
    move(client, x_1+x, y_1+y, theta_1_rad)
    
    time.sleep(2)
    location2 = client.get_robot_pose()
    print("########## pose 2 ############")
    print(location2)
    x_2 = location2.x
    y_2 = location2.y
    theta_2_rad = location2.theta
    theta_2_deg = math.degrees(theta_2_rad)
    
    diff_meter = np.sqrt((x_2-x_1)*(x_2-x_1) + (y_2-y_1)*(y_2-y_1))
    
    diff_rad = theta_2_rad - theta_1_rad
    diff_deg = theta_2_deg - theta_1_deg
    
    print("DIFF_X: " + str(x_2-x_1) + ", DIFF_Y: " + str(y_2-y_1))
    print("DIFF_METER: " + str(diff_meter) + " [m]")
    print("DIFF_rad: " + str(diff_rad) + " [rad]")
    print("DIFF_DEG: " + str(diff_deg) + " [度]")
    
    time.sleep(1)
    
    angle = math.radians(30)
    move(client, x_2, y_2, theta_2_rad+angle)

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
    