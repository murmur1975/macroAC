# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 23:54:13 2019

SFVの6種の基本攻撃のSEの共分散を計算する

@author: murmur
"""

import wave
from scipy import int16
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import warnings

folder = './SFV_SE/'
filename = ('LP.wav', 'LK.wav', 'MP.wav', 'MK.wav', 'HP.wav', 'HK.wav')
# WAVファイルを開く
wr = [wave.open(folder + k, 'rb') for k in filename]
wr_w = wave.open(folder + 'whole.wav', 'rb')

# データの読み込み
data = [k.readframes(k.getnframes()) for k in wr]
data_w = wr_w.readframes(wr_w.getnframes())

# ファイルクローズ
[k.close() for k in wr]
wr_w.close()

# 文字型から数値型に変換
num_data = [np.frombuffer(k, dtype=int16) for k in data]
num_data[5] = num_data[5][0:-1]  # HKのデータが何故か1個多かったので揃える
num_data_w = np.fromiter(data_w, dtype='int16')  # 切り出し前の長いデータ

# 標準偏差が1となるよう正規化
v = [np.std(k) for k in num_data]
num_data = [k/j for k,j in zip(num_data, v)]

# 信号を行列形式に並べる（これもリスト内包表現にしたいがわかんない）
A = num_data[0]
for k in range(1,6):
    A = np.vstack([A,num_data[k]])

# 分散共分散行列を計算（標本分散）
S = np.cov(A, rowvar=1, bias=1)

# Sをプロット
label = ('LP', 'LK', 'MP', 'MK', 'HP', 'HK')
x = range(1,7)
y = range(1,7)
xx, yy = np.meshgrid(x, y)
sb.set()
fig, ax = plt.subplots(figsize=(6,6))
C = 2e3  # グラフ表示のための定数（見やすいように適当に決める）
#ax.scatter(xx, yy, s=abs(S)*C, alpha=0.5)  # 本来はこれ
warnings.simplefilter('ignore', RuntimeWarning)
ax.scatter(xx, yy, s= S*C, alpha=0.5)
ax.scatter(xx, yy, s=-S*C, alpha=0.5)
plt.xticks(x, label)
plt.yticks(y, label)
plt.show()

# 共分散のうちの最大値
print('max covariance in S :',np.max(S[S<0.999]))
