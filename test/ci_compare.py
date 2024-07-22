import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import numpy as np

def compare_data(data1, data2):
    win, draw, lose = 0, 0, 0
    for i in range(data1.shape[0]):
        if data1["ci"][i] > data2["ci"][i]:
            win += 1
        elif data1["ci"][i] == data2["ci"][i]:
            draw += 1
        else:
            lose += 1
            
    print("win: " + str(win) + ", draw: " + str(draw) + ", lose: " + str(lose))

date1 = "24-01-17 02-27-33"
date2 = "24-01-11 00-43-24"
date3 = "23-12-27 22-05-26"

df1 = pd.read_csv("./" + date1 + ".csv", names=['episode_id', 'ci'], header=None)
df2 = pd.read_csv("./" + date2 + ".csv", names=['episode_id', 'ci'], header=None)
df3 = pd.read_csv("./" + date3 + ".csv", names=['episode_id', 'ci'], header=None)

base = np.linspace(0, 200, 1000)

compare_data(df1, df2)
plt.scatter(df1["ci"], df2["ci"], color="blue", label="2-Scale Map vs Object Category Map")
plt.plot(base, base, color="green")
#ラベルの追加
plt.xlabel('2-Scale Map')
plt.ylabel('Object Category Map')
#表示範囲の指定
plt.xlim(0, 200)
plt.ylim(0, 200)
#グラフの保存
plt.savefig('./ci_compare_SCA&OBJ.png')

plt.clf()
compare_data(df1, df3)
plt.scatter(df1["ci"], df3["ci"], color="blue", label="2-Scale Mapp vs Individual")
plt.plot(base, base, color="green")
#ラベルの追加
plt.xlabel('2-Scale Map')
plt.ylabel('Individual')
#表示範囲の指定
plt.xlim(0, 200)
plt.ylim(0, 200)
#グラフの保存
plt.savefig('./ci_compare_SCA&IND.png')

plt.clf()
compare_data(df2, df3)
plt.scatter(df2["ci"], df3["ci"], color="blue", label="Object Category Map vs Individual")
plt.plot(base, base, color="green")
#ラベルの追加
plt.xlabel('Object Category Map')
plt.ylabel('Individual')
#表示範囲の指定
plt.xlim(0, 200)
plt.ylim(0, 200)
#グラフの保存
plt.savefig('./ci_compare_OBJ&IND.png')