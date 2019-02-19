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

import matplotlib
import warnings

# データ読み込み
folder = './SFV_SE/'
data_w, fs = sf.read(folder + 'whole.wav')

# 標準偏差が1となるよう正規化
data_w = data_w / np.std(data_w)
       
# ダウンサンプリング
dsr1 = 10
dsr = dsr1*dsr1
data_w = signal.decimate(data_w, dsr1, ftype='fir')
data_w = signal.decimate(data_w, dsr1, ftype='fir')
fs = int(fs/dsr)
LP = data_w[2894:3038]
    
#%% グラフプロット
from IPython.display import HTML
sb.set()
x = np.linspace(-0.5, 0.5, int(fs))  # amp用
xx = x[221-len(LP):221]
k_stop = 2894-(221-len(LP))+1  # data_wの停止時刻

fig, ax = plt.subplots(figsize=(10,6))
i = 0
k = k_stop-400
cv = np.empty(0)

def plot(frame):
    # future warining抑制
    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

    global k
    global i
    global cv

    i = i + 1
    k = k + 1

    # [1] waveforms
    ax = plt.subplot(2,1,1)
    if k < k_stop:
        plt.cla()  # 図をクリア
        plt.title('Sound Waveforms')
        plt.plot(x, data_w[k:k+len(x)], alpha=0.8)  # 全体SEプロット
        plt.plot(xx, LP, alpha=0.7)  # LPSEプロット
        plt.xlim(-0.5, 0.5)
        plt.ylim(-3, 3)
        plt.tick_params(labelbottom=False)
        plt.legend(['game sound','LP sound'], loc=4)
        plt.fill([xx[0],xx[0],0,0],[-5,3,3,-5], alpha=0.1, color='coral')
        plt.vlines(0, -3, 3, colors='k', linewidth=1, linestyles='dotted', alpha=0.5)
        ax.annotate('waveform of\nLP sound', xy=(-0.185,2), color='coral')
        plt.ylabel('amplitude')
    if k == k_stop:
        ax.annotate('2 waveforms\nare matched! [A]', xy=(-0.17,-2), color='r', fontsize=15)
        plt.vlines(-144/441, -3, 3, colors='k', linewidth=1, linestyles='dotted', alpha=0.5)

    # [2] covariance
    ax = plt.subplot(2,1,2)
    if k < k_stop:
        LP_w = data_w[k+77:k+77+len(LP)]  # LPとのcov計算のためにwholeから切り出す
        temp = np.cov(LP/np.std(LP), LP_w/np.std(LP_w))  # cov計算
        cv = np.append(cv, temp[0,1])  # グラフのためにcov蓄積
        x_cv = np.linspace(-i*(1/fs), 0, i)  # covグラフ軸

        plt.cla()  # 図をクリア
        plt.title('Covariance and Threshold')
        plt.plot(x_cv, cv, color='yellowgreen')  # 過去のcov
        plt.plot(0, cv[-1], 'o', color='yellowgreen') # 現在のcov
        plt.hlines(0.6, -0.5, 0.5, colors='forestgreen', linestyles='dashed', alpha=0.5)
        plt.vlines(0, -1, 1.2, colors='k', linewidth=1, linestyles='dotted', alpha=0.5)
        plt.xlim(-0.5, 0.5)
        plt.ylim(-1, 1.2)
        plt.xlabel('time[s]')
        plt.ylabel('covariance')
        plt.legend(['past covariance','current covariance'], loc=4)

        pos_ty = -0.75
        ax.annotate('NOW', xy=(0,pos_ty-0.15), ha='center', color='gray')
        ax.annotate('', xy=(0,pos_ty), xycoords='data',
                    xytext=(-0.1,pos_ty), textcoords='data', 
                    arrowprops=dict(arrowstyle='<-', color='gray'))
        ax.annotate('past', xy=(-0.1,pos_ty-0.15), ha='center', color='gray')
        ax.annotate('', xy=(0,pos_ty), xycoords='data',
                    xytext=(0.1,pos_ty), textcoords='data', 
                    arrowprops=dict(arrowstyle='<-', color='gray'))
        ax.annotate('future', xy=(0.1,pos_ty-0.15), ha='center', color='gray')
        ax.annotate('threshold', xy=(0.48,0.62), ha='right', color='forestgreen')
    if i > 133:
        ax.annotate('LP hit\ndetected!', xy=((133-i)/fs,cv[133]), xycoords='data',
                    xytext=((133-i)/fs+0.1,0.8), textcoords='data', color='coral', fontsize=14, ha='center',
                    arrowprops=dict(arrowstyle='->', color='coral'))
    if k == k_stop:
        ax.annotate('covariance is max(1.0)!\n(this means [A])', xy=(0.17, 0.02), color='r', fontsize=15, ha='center')
        ax.annotate('LP detected!', xy=(0,1), xycoords='data',
                    xytext=(0.15,0.8), textcoords='data', color='coral', fontsize=15, ha='center',
                    arrowprops=dict(arrowstyle='->', color='coral'))
        ax.annotate('detection lag', xy=(-72/441,0.82), ha='center', color='coral')
        ax.annotate('', xy=(-144/441,0.8), xycoords='data',
                    xytext=(0,0.8), textcoords='data', color='coral',
                    arrowprops=dict(arrowstyle='<->', color='coral', linestyle='dotted'))
        plt.vlines(-144/441, 0.5, 1.2, colors='k', linewidth=1, linestyles='dotted', alpha=0.5)

    plt.tight_layout()
    return fig,
    
# アニメーション描画
ani = animation.FuncAnimation(fig, plot, interval=50, frames=500, blit=True)
#plt.show()

# アニメーション保存
ani.save("bscore.gif")
#ani.save('anim.mp4', writer = 'ffmpeg', bitrate=1500, codec='h264')
#HTML(ani.to_html5_video())
