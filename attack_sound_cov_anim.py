# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:47:57 2019

SFVの6種の基本攻撃のSEの共分散の計算イメージのアニメ化

@author: murmur
"""
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import soundfile as sf
import matplotlib.animation as animation
from scipy import signal

# データ読み込み
folder = './SFV_SE/'
data_w, fs = sf.read(folder + 'whole.wav')

# 標準偏差が1となるよう正規化
data_w = data_w / np.std(data_w)

# 変数準備
fr_unit = 1
Lw = len(data_w)
        
el = 10
data_w = signal.decimate(data_w, el, ftype='fir')
data_w = signal.decimate(data_w, el, ftype='fir')
fs=fs/el/el
LP = data_w[2894:3038]
    
#%% グラフプロット
sb.set()
x = np.linspace(-0.5, 0.5, int(fs))
xx = x[221-len(LP):221]
k_stop = 2894-(221-len(LP))  # data_wの停止時刻
fig, ax = plt.subplots(figsize=(10,7))
k = 2700
i = 0
cv = np.empty(0)
def plot(frame):
    global k
    global i
    global cv
    global x_cv
    plt.cla()
    i = i + 1
    if k < k_stop:
        k = k + 1
    else:
        ax.annotate('match!\nmax covariance', xy=(-0.17,1.3), color='r')
    LP_w = data_w[k+77:k+77+len(LP)]
    temp = np.cov(LP/np.std(LP), LP_w/np.std(LP_w))
    cv = np.append(cv, temp[0,1])
    x_cv = np.linspace(x[220-i+1], x[220], i)

    plt.plot(x, data_w[k:k+len(x)], alpha=0.8)
    plt.plot(xx, LP, alpha=0.7)
    if k < k_stop:
        plt.plot(x_cv, cv-4)

    ax.set_ylim(-5, 3)
    plt.legend(['game sound','LP sound','covariance'], loc=1)
    plt.fill([xx[0],xx[0],0,0],[-5,3,3,-5], alpha=0.2, color='coral')
    ax.annotate('waveform of\nLP sound', xy=(-0.185,2), color='coral')
    ax.annotate('NOW', xy=(0.005,2.75), ha='center', color='k')
    ax.annotate('', xy=(0,3), xycoords='data',
                xytext=(-0.1,3), textcoords='data', 
                arrowprops=dict(arrowstyle='<-', color='k'))
    ax.annotate('past', xy=(-0.05,2.75), ha='right', color='k')
    ax.annotate('', xy=(0,3), xycoords='data',
                xytext=(0.1,3), textcoords='data', 
                arrowprops=dict(arrowstyle='<-', color='k'))
    ax.annotate('future', xy=(0.05,2.75), ha='left', color='k')
#    plt.plot(0, cv[-1]-4, 'o')

ani = animation.FuncAnimation(fig, plot, interval=30, frames=180)
ani.save("anim.mp4", writer = 'ffmpeg')
#plt.show()
