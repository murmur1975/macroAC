# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:13:52 2019

SFVの6種の基本攻撃のSEの主成分を計算する

@author: murmur
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import soundfile as sf
from sklearn.decomposition import PCA

folder = './SFV_SE/'
filename = ('LP.wav', 'LK.wav', 'MP.wav', 'MK.wav', 'HP.wav', 'HK.wav')

data = [sf.read(folder + k) for k in filename]  # データ, サンプリング周波数を取得
data = [data[k][0] for k in range(6)]  # データのみを取得
data[5] = data[5][0:-1]  # HKのデータが何故か1個多かったので揃える

# 標準偏差が1となるよう正規化
v = [np.std(k) for k in data]
data = [k/j for k,j in zip(data, v)]

# 音声行列生成
A = data[0]
for j in range(1,6):
    A = np.vstack([A,data[j]])

# 主成分分析(PCA)
cr = []  # 寄与率格納用
N_pca = 2  # 主成分の次元（第何主成分までを使うか）
pca = PCA(n_components=N_pca)  # PCAオブジェクト生成
pca.fit(A)  # PCA実行（共分散行列の生成＞固有値分解）
for k in range(N_pca):
    cr.append(round(pca.explained_variance_ratio_[k]*100, 1))  # 第k主成分の寄与率
tr = pca.fit_transform(A)  # 行列AをN_pca次元固有空間に射影
print(cr, sum(cr))  # 寄与率, 累積寄与率の表示

# プロット
sb.set()
ax, fig = plt.subplots()
plt.scatter(tr[:,0],tr[:,1])
label = ['LP', 'LK', 'MP', 'MK', 'HP', 'HK']
for i,k in enumerate(label):
    plt.annotate(k,tr[i])
