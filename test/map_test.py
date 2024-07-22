#from IPython.display import Image, display
from PIL import Image
import sys
import os
import io

import numpy as np

import kachaka_api
sys.path.append(f"/Users/{os.environ['USER']}/Desktop/habitat2kachaka/kachaka-api/python/")


class LogWriter:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        
        #ファイルを新規作成 or 上書き
        with open(self.file_path, "w") as f:
            pass
            
    #改行なし
    def write(self, log: str) -> None:
        #ファイルへ追記
        with open(self.file_path, 'a', newline='') as f:
            f.write(log + ",")
            
    #改行あり
    def writeLine(self, log: str="") -> None:
        #ファイルへ追記
        with open(self.file_path, "a") as f:
            f.write(log + "\n")
            
            
class LogManager:
    def __init__(self) -> None:
        self.writers = {}
    
    def setLogDirectory(self, dir_path: str) -> None:
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(self.dir_path)
            
        self.dir_path = "./" + dir_path + "/"
        
    def makeDir(self, path_dir: str) -> str:
        path_dir = self.dir_path + path_dir
        if not os.path.exists(path_dir):
            # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(path_dir)
            
        return path_dir
    
    def createLogWriter(self, key: str) -> LogWriter:
        if key in self.writers:
            return self.writers[key]
        
        writer = LogWriter(self.dir_path + key + ".csv")
        self.writers[key] = writer
        return writer
    
    #テスト用
    def printWriters(self) -> None:
        print(self.writers)
        
        
def resize_map(map):
    h = len(map)
    w = len(map[0])
    size = 3
    
    resized_map = np.zeros((int(h/size), int(w/size)))
    print("h=" + str(h) + ", w=" + str(w) + ", h'=" + str(len(resized_map)) + ", w'=" + str(len(resized_map[0])))

    # mapのresize
    for i in range(len(resized_map)):
        for j in range(len(resized_map[0])):
            flag = False
            num_0 = 0
            num_2 = 0
            for k in range(size):
                if flag == True:
                    break
                
                if size*i+k >= h:
                    break
                
                for l in range(size):
                    if size*j+l >= w:
                        break
                    if map[size*i+k][size*j+l] == 1:
                        resized_map[i][j] = 1
                        flag = True
                    elif map[size*i+k][size*j+l] == 0:
                        num_0 += 1
                    elif map[size*i+k][size*j+l] == 2:
                        num_2 += 1
                        
            if flag == False:
                if num_0 > num_2:
                    resized_map[i][j] = 0
                else:
                    resized_map[i][j] = 2
            
    # borderをちゃんと作る
    for i in range(len(resized_map)):
        for j in range(len(resized_map[0])):
            flag = False
            if resized_map[i][j] == 2:
                for k in [-1, 1]:
                    if flag == True:
                        break
                    if i+k < 0 or i+k >= len(resized_map):
                        continue
                    for l in [-1, 1]:
                        if j+l < 0 or j+l >= len(resized_map[0]):
                            continue
                        if resized_map[i+k][j+l] == 0:
                            resized_map[i][j] = 1
                            flag = True
                            break
    
    return resized_map

def clip_map(map):
    grid_delta = 3
    range_x = np.where(np.any(map != 0, axis=1))[0]
    range_y = np.where(np.any(map != 0, axis=0))[0]
    
    ind_x_min = range_x[0]
    ind_x_max = range_x[-1]
    ind_y_min = range_y[0]
    ind_y_max = range_y[-1]

    return map[
            ind_x_min - grid_delta : ind_x_max + grid_delta,
            ind_y_min - grid_delta : ind_y_max + grid_delta,
        ]
        
        
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
                map[i][j] = 1
            elif map_data[i][j][0] == 253:
                map[i][j] = 2
            else:
                map[i][j] = -1
                
    return map


def recreate_map(map_data):
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            # 棚
            if (i>=105 and i <= 154 and j>=26 and j <= 35):
                map_data[i][j] = 0
                
            # 机の下
            elif (i>=131 and i <= 193 and j>=53 and j <= 75):
                map_data[i][j] = 0
            elif (i>=105 and i <= 116 and j>=40 and j <= 51):
                map_data[i][j] = 0
            elif (i>=102 and i <= 112 and j>=56 and j <= 67):
                map_data[i][j] = 0
            elif (i>=101 and i <= 111 and j>=71 and j <= 83):
                map_data[i][j] = 0
            elif (i>=88 and i <= 106 and j>=83 and j <= 132):
                map_data[i][j] = 0
            elif (i>=107 and i <= 119 and j>=122 and j <= 130):
                map_data[i][j] = 0
                
            # 配線
            elif (i>=112 and i <= 130 and j>=64 and j <= 75):
                map_data[i][j] = 0
                
            # 机の下&ホワイトボード
            elif (i>=118 and i <= 137 and j>=89 and j <= 132):
                map_data[i][j] = 0
                
            # ホワイトボード~机
            elif (i>=119 and j>=108):
                map_data[i][j] = 0
            elif (i>=168 and i <= 201 and j>=94 and j <= 101):
                map_data[i][j] = 0
                
            # ソファーを大きく
            elif (i>=199 and i <= 201 and j>=97 and j <= 107):
                map_data[i][j] = 0
                
            # 出入り口
            elif (i>=186 and i <= 203 and j>=8 and j <= 31):
                map_data[i][j] = 0
            elif (i>=191 and i <= 203 and j>=6 and j <= 7):
                map_data[i][j] = 0
            elif (i>=189 and i <= 196 and j>=32 and j <= 41):
                map_data[i][j] = 0
            elif (i>=198 and i <= 200 and j>=32 and j <= 33):
                map_data[i][j] = 0
            elif (i>=224 and j>=24 and j <= 36):
                map_data[i][j] = 0
                
            # 村田研究室ゾーン
            elif i>=230:
                map_data[i][j] = 0
                
            # 右端
            elif j>=124:
                map_data[i][j] = 0    
            
            # 細かいところ
            elif (i>=103 and i <= 112 and j>=36 and j <= 53):
                map_data[i][j] = 0
            elif (i>=224 and i <= 230 and j>=44 and j <= 46):
                map_data[i][j] = 0
            elif (i <= 79):
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
                            map_data[i][j] = 1
                            flag = True
                            break
                        
    return map_data
                

if __name__ == "__main__":
    client = kachaka_api.KachakaApiClient("192.168.100.31:26400")
    client.update_resolver()
    
    logmanager = LogManager()
    logmanager.setLogDirectory("./")
    writer = logmanager.createLogWriter("map")
    writer_recreate = logmanager.createLogWriter("map_recreated")
    
    #client.set_auto_homing_enabled(False)
    
    #client.move_shelf("S01", "L02")
    
    map = client.get_png_map()
    print(map.name)
    print(map.resolution, map.width, map.height)
    print(map.origin)
    robot_pose = client.get_robot_pose()
    print(robot_pose)
    
    dx = robot_pose.x - map.origin.x
    dy = robot_pose.y - map.origin.y
    print("dx=" + str(dx) + ", dy=" + str(dy))
    
    grid_x = dx / map.resolution
    grid_y = dy / map.resolution
    grid_y = map.height-grid_y
    print("grid_x=" + str(grid_x) + ", grid_y=" + str(grid_y))
    

    map_img = Image.open(io.BytesIO(map.data))
    
    data = map_img.getdata()
    map_data = np.array(data) 
    
    # 分割サイズ
    chunk_size = map.width
    # 分割
    map_data = np.array(np.array_split(map_data, range(chunk_size, len(map_data), chunk_size), axis=0))
    
    print("map_data: " + str(len(map_data)) + "," + str(len(map_data[0])))
    map_data = create_map(map_data)
    map_data = resize_map(map_data)
    print("resize: " + str(len(map_data)) + "," + str(len(map_data[0])))
    
    map_recreated = recreate_map(map_data)
    
    #clipped_map = clip_map(map_data)
    clipped_map = map_data
    #print("clip: " + str(len(clipped_map)) + "," + str(len(clipped_map[0])))
    
    grid_x /= 3
    grid_x = round(grid_x)
    grid_y /= 3
    grid_y = round(grid_y)
    print("grid_x=" + str(grid_x) + ", grid_y=" + str(grid_y))

    """
    for i in range(len(clipped_map)):
        for j in range(len(clipped_map[0])):
            if i == grid_y and j == grid_x:
                writer.write(str(5))
            else:
                writer.write(str(clipped_map[i][j]))
        writer.writeLine()
    """
        
    for i in range(len(clipped_map)):
        for j in range(len(clipped_map[0])):
            if i == grid_y and j == grid_x:
                writer_recreate.write(str(5))
            else:
                writer_recreate.write(str(map_recreated[i][j]))
        writer_recreate.writeLine()
    
    map_img.save("test.png")
    
    
