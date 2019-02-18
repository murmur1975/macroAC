# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:09:10 2019

SFVの6種の基本攻撃のSEの共分散を計算する
信号長をどこまで短くできるかマトリクス的に調べる
信号長は識別ラグなのでなるべく短くしたい

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

#sb.set()
fig, ax = plt.subplots(figsize=(10,10))

mx = [0,2,4,6,8]
md = [5,6,7,13,19]
mgx, mgd = np.meshgrid(mx, md)
mgy = [k+j for k,j in zip(mgx,mgd)]

for idx,(kx,ky) in tqdm(enumerate(zip(np.hstack(mgx), np.hstack(mgy)), start=1)):
    data_ = [sf.read(folder + k) for k in filename]
    data = [data_[k][0][int(kx*44100/60):int(ky*44100/60)] for k in range(6)]  # 20672のデータのうち3000～6700のデータを使う
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
    for k in range(0,smp):
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
        
    # グラフプロット
    x = np.linspace(0, fr_unit*smp/fs, smp)
    ax = plt.subplot(5,5,idx)
    plt.plot(x, sfunc[:,1], alpha=0.7)
    #plt.plot(x, data_w[range(Lw-smp,Lw)]/max(abs(data_w)), alpha=0.8)
    #plt.legend(['LP', 'LK', 'MP', 'MK', 'HP', 'HK', 'sound'])
    ax.set_yticks(np.arange(0, 1.01, 0.1)) 
    ax.set_yticks(np.arange(0, 1, 0.02), minor=True) 
    ax.grid(which='minor', alpha=0.2) 
    ax.grid(which='major', alpha=0.5) 
    plt.ylim([0,1.01])
    plt.title('{}F :{}F-{}F'.format(ky-kx,kx,ky))
    if idx not in [21,22,23,24,25]:
        ax.set_xticklabels([])
    if idx not in [ 1, 6,11,16,21]:
        ax.set_yticklabels([])

