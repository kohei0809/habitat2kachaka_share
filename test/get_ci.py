import os
import re
import numpy as np

def extract_ci(video_filename):
    # ファイル名からCIを抽出する正規表現パターン
    pattern = r'-([0-9]+\.[0-9]+)-'

    # 正規表現パターンに一致する部分を抽出
    match = re.search(pattern, video_filename)

    # 一致した部分があればその値を返し、なければNoneを返す
    if match:
        ci = match.group(1)
        return ci
    else:
        return None
    
def extract_episode_id(video_filename):
    # ファイル名からCIを抽出する正規表現パターン
    pattern = r'=([0-9]+)-'

    # 正規表現パターンに一致する部分を抽出
    match = re.search(pattern, video_filename)

    # 一致した部分があればその値を返し、なければNoneを返す
    if match:
        episode = match.group(1)
        return episode
    else:
        return None

def process_video_files(directory):
    # 指定されたディレクトリ内の全てのファイルに対して処理を行う
    num = 0
    data = []
    for filename in os.listdir(directory):
        # ファイルの絶対パスを取得
        filepath = os.path.join(directory, filename)

        # ファイルがディレクトリでなく、かつ拡張子が.mp4の場合に処理を行う
        if os.path.isfile(filepath) and filename.endswith(".mp4"):
            ci = extract_ci(filename)
            episode_id = extract_episode_id(filename)
            if ci:
                num += 1
                data.append([int(episode_id), ci])
            else:
                print(f"CI not found in {filename}")
                
    data = np.array(data)
    order = np.argsort(data[:,0])
    ordered_data = data[order,:] 
    
    for i in range(ordered_data.shape[0]):
        print(str(ordered_data[i, 0]) + "," + str(ordered_data[i, 1]))

# ビデオファイルが含まれているディレクトリのパスを指定
video_directory = "./../24-01-17 02-27-33"
#video_directory = "./../24-01-11 00-43-24"
#video_directory = "./../23-12-27 22-05-26"

# 処理を実行
process_video_files(video_directory)
