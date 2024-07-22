import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import numpy as np

date = "24-01-17 02-27-33"
date = "24-01-11 00-43-24"
date = "23-12-27 22-05-26"

df = pd.read_csv("./" + date + ".csv")
print(df.max())
plt.hist(df, bins=range(0, 210, 10), )

#ラベルの追加
plt.xlabel('$CI$')
plt.ylabel('Number of Episodes')

#表示範囲の指定
#plt.xlim(0, 50000000)
plt.ylim(0, 80)

plt.xticks(np.arange(0, 210, 20))

#凡例の追加
#plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#グラフの保存
plt.savefig('./' + date + '.png')