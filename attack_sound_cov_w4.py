# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:38:32 2019

SFVの6種の基本攻撃のSEの共分散を計算する
ダウンサンプリングして実際の音声との共分散を計算し、しきい値を決める。

@author: murmur
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import soundfile as sf
from tqdm import tqdm
from scipy import signal

folder = './SFV_SE/'
filename = ('LP.wav', 'LK.wav', 'MP.wav', 'MK.wav', 'HP.wav', 'HK.wav')

data = [sf.read(folder + k) for k in filename]
data = [data[k][0] for k in range(6)]
data[5] = data[5][0:-1]  # HKのデータが何故か1個多かったので揃える
data_w, fs = sf.read(folder + 'whole.wav')

# ダウンサンプリング
# decimate関数ヘルプより：ダウンサンプリングレートが13を超える場合は複数回に分ける
# 理由はおそらく、ダウンサンプリング用LPFを設計できないため
el = 10  # 1回あたりのダウンサンプリングレート
d_rate = el*el  # 2回で100になる
fs = int(fs/d_rate)  # サンプリング周波数を更新
# ダウンサンプリング用LPFはFIR型とする
# FIR型フィルタは位相特性が直線なので、波形が保存される
data = [signal.decimate(data[k], el, ftype='fir') for k in range(6)]
data = [signal.decimate(data[k], el, ftype='fir') for k in range(6)]
data_w = signal.decimate(data_w, el, ftype='fir')
data_w = signal.decimate(data_w, el, ftype='fir')

# 標準偏差が1となるよう正規化
v = [np.std(k) for k in data]
data = [k/j for k,j in zip(data, v)]
data_w = data_w / np.std(data_w)

# 変数準備
sfunc = []  # 分散共分散行列の要素の時系列成分
tr_func = []  # 主成分の要素の時系列成分
#fr_unit = int(fs/60/7)  # 44100/60 : 1フレームの長さ
fr_unit = 1
L = len(data[0])  # 音声データの長さ
Lw = len(data_w)
smp = int((Lw-L)/fr_unit)

# 共分散計算
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
        
#%% グラフプロット
sb.set()
x = np.linspace(0, fr_unit*smp/fs, smp)
th_dtct = [0.5]*len(x) # 識別スレッシュホールド
fig, ax = plt.subplots(figsize=(10,5))
plt.plot(x, sfunc, alpha=0.7)
plt.plot(x, data_w[range(Lw-smp,Lw)]/max(abs(data_w)), alpha=0.8)
plt.plot(x, th_dtct, color='slateblue', linestyle='dotted')
plt.legend(['LP', 'LK', 'MP', 'MK', 'HP', 'HK', 'sound'])
ax.annotate('threshold', (0,0.52), xycoords='data', color='slateblue')

#%% 矢印描画
def ann_arrow(ax, pos_a, pos_b, key, dr1=20, dr2=-80):  # dr はアングル矢印のコメントの位置

    def vdline(ax, pos_x):  # 垂直破線
        ax.annotate('', xy=(pos_x, -1), xycoords='data',
                xytext=(pos_x,  1), textcoords='data',
                arrowprops=dict(arrowstyle='-', linestyle="dashed", color='k', alpha=0.3))

    #両矢印
    ax.annotate('', xy=(pos_a[0], pos_a[1]), xycoords='data',
                xytext=(pos_b[0], pos_b[1]), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='coral'))
    # テキスト
    ax.annotate('detection lag\n0.47s (28frame)',
                ((pos_a[0]+pos_b[0])/2, (pos_a[1]+pos_b[1])/2-0.08),
                xycoords='data', ha='center', color='coral')
    # 垂直破線
    vdline(ax, pos_a[0])
    vdline(ax, pos_b[0])
    # アングル矢印（左）
    ax.annotate(key+' key triggered\n(probably)',
                xy=(pos_b[0], 1), xycoords='data',
                xytext=(dr1, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='dimgray',
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    # アングル矢印（右）
    ax.annotate('hit detected\n(cov > 0.5)',
                xy=(pos_a[0], 1), xycoords='data',
                xytext=(dr2, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='dimgray',
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"))
# グラフ描画
def plotg():
    plt.plot(x, sfunc, alpha=0.7)
    plt.plot(x, data_w[range(Lw-smp,Lw)]/max(abs(data_w)), alpha=0.8)
    plt.plot(x, th_dtct, color='slateblue', linestyle='dotted')
    plt.legend(['LP', 'LK', 'MP', 'MK', 'HP', 'HK', 'sound'], loc=4)
    plt.xlabel('time[s]')
    plt.ylabel('covariance (each vs sound)')

Ls = L/(fs/10/10)
h = 0.75

#%% LPの場合のグラフ
fig, ax = plt.subplots(figsize=(10,5))
plotg()
plt.xlim([7.25, 8.5])
p1 = 7.77; ann_arrow(ax, (p1, h), (p1-L, h), 'LP')
p2 = 8.37; ann_arrow(ax, (p2, h), (p2-L, h), 'LP')
ax.annotate('threshold', (p1-Ls/2,0.52), xycoords='data', ha='center', color='slateblue')

#%% HPの場合のグラフ
fig, ax = plt.subplots(figsize=(10,5))
plotg()
plt.xlim([31, 31+1.25]) #33.1
p1 = 31.63;  ann_arrow(ax, (p1, h), (p1-Ls, h), 'HP', dr2=20)
#p2 = 32.715; ann_arrow(ax, (p2, h), (p2-L, h), 'HP', dr2=20)
ax.annotate('threshold', (p1-0.12,0.52), xycoords='data', ha='center', color='slateblue')
