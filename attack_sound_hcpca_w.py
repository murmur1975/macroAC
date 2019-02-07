# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:42:31 2019

SFVの6種の基本攻撃のSEをフーリエ変換して主成分を計算する
 実際のゲーム音声に適用する

@author: murmur
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import soundfile as sf
from sklearn.decomposition import PCA
from tqdm import tqdm
from smooth import smooth

folder = './SFV_SE/'
filename = ('LP.wav', 'LK.wav', 'MP.wav', 'MK.wav', 'HP.wav', 'HK.wav')

data = [sf.read(folder + k) for k in filename]  # データ, サンプリング周波数を取得
data = [data[k][0] for k in range(6)]  # データのみを取得
data[5] = data[5][0:-1]  # HKのデータが何故か1個多かったので揃える
data_w, samplerate = sf.read(folder + 'whole.wav')  # ゲーム音声の読み込み

# 標準偏差が1となるよう正規化
data = [k/np.std(k) for k in data]
data_w = data_w / np.std(data_w)

# 音声行列生成
A = data[0]
for j in range(1,6):
    A = np.vstack([A,data[j]])

# フーリエ変換
n_fft = 1024
L_smth=15
X = [np.abs(np.fft.rfft(k, n_fft)) for k in A]
X = [smooth(k, window_len=L_smth) for k in X]  # スムージング
X = [k[int((L_smth-1)/2):-int((L_smth-1)/2)] for k in X]  # smoothで生じた冗長データを処理（両端を切る）
X = [k/np.std(k) for k in X]  # 正規化
A = X


# 主成分分析(PCA)
cr = []  # 寄与率格納用
N_pca = 2  # 主成分の次元（第何主成分までを使うか）
pca = PCA(n_components=N_pca)  # PCAオブジェクト生成
pca.fit(A)  # PCA実行（共分散行列の生成＞固有値分解）
for k in range(N_pca):
    cr.append(round(pca.explained_variance_ratio_[k]*100, 1))  # 第k主成分の寄与率
tr = pca.fit_transform(A)  # 行列AをN_pca次元固有空間に射影
print(cr, sum(cr))  # 寄与率, 累積寄与率の表示

# ゲーム音声の主成分を算出
tr_func = []  # 主成分の要素の時系列成分
fr_unit = 441  # 44100/60 : 1フレームの長さ
L = len(data[0])  # 音声データの長さ
Lw = len(data_w)
smp = int((Lw-L)/fr_unit)
fs = 44100
for k in tqdm(range(0,smp)):
    tmp = data_w[fr_unit*k : fr_unit*k + L]
    tmp = tmp/np.std(tmp)
    tmp = np.abs(np.fft.rfft(tmp, n_fft))  # フーリエ変換
    tmp = smooth(tmp, window_len=L_smth)  # スムージング
    tmp = tmp[int((L_smth-1)/2):-int((L_smth-1)/2)]  # smoothで生じた冗長データを処理（両端を切る）
    tmp = tmp / np.std(tmp)  # 正規化
    tmp = tmp[:,np.newaxis]
    tr_tmp = pca.transform(tmp.T)
    tr_func.append(tr_tmp[0])  # ゲーム音声をN_pca次元固有空間に射影

# プロット
sb.set()
ax, fig = plt.subplots()
plt.scatter(tr[:,0],tr[:,1])
label = ['LP', 'LK', 'MP', 'MK', 'HP', 'HK']
for i,k in enumerate(label):
    plt.annotate(k,tr[i])
tr_func = np.vstack(tr_func)
#plt.scatter(tr_func[:,0],tr_func[:,1])
plt.plot(tr_func[:,0],tr_func[:,1])
