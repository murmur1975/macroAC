# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:38:32 2019

SFVの6種の基本攻撃のSEの共分散を計算する

@author: murmur
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import warnings
import soundfile as sf
from tqdm import tqdm


folder = './SFV_SE/'
filename = ('LP.wav', 'LK.wav', 'MP.wav', 'MK.wav', 'HP.wav', 'HK.wav')

data = [sf.read(folder + k) for k in filename]
data = [data[k][0] for k in range(6)]
data[5] = data[5][0:-1]  # HKのデータが何故か1個多かったので揃える
data_w, samplerate = sf.read(folder + 'whole.wav')


# 標準偏差が1となるよう正規化
v = [np.std(k) for k in data]
data = [k/j for k,j in zip(data, v)]
data_w = data_w / np.std(data_w)

#%% 
'''
引数 : 処理したい観測信号の先頭からのフレーム数
       そのフレームからヒット音声信号の長さを処理対象にする
'''

sfunc = []  # 分散共分散行列の要素の時系列成分
fr_unit = 88  # 44100/60 : 1フレームの長さ
L = len(data[0])  # 音声データの長さ
smp = 24182
fs = 44100

for k in tqdm(range(0,smp)):
    # 信号を行列形式に並べる（これもリスト内包表現にしたいがわかんない）
    tmp = data_w[fr_unit*k : fr_unit*k + L]
    tmp = tmp/np.std(tmp)

    A = tmp
    for j in range(6):
        A = np.vstack([A,data[j]])
    
    # 分散共分散行列を計算（標本分散）
    S = np.cov(A, rowvar=1, bias=1)
    if sfunc == []:
        sfunc = S[0,1:7]
    else:
        sfunc = np.vstack([sfunc, S[0,1:7]])

sb.set()
x = np.linspace(0, fr_unit*smp/fs, smp)
ax, fig = plt.subplots()
plt.plot(x, sfunc)
plt.legend(['LP', 'LK', 'MP', 'MK', 'HP', 'HK'])
