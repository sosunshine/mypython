# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:51:12 2018

@author: ocean
"""

import re
import os
import pandas as pd
import matplotlib
import numpy as np
#import zipfile
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
#***********************************
def PCA_Scatter(pca):
    p1=plt.figure(figsize=(8,6),dpi=80)
    ax1=p1.add_subplot(1,1,1)
    ax1.set_title="distribution of pca1 and pca2"
    #plt.title=""
    plt.xlabel="pca1"
    plt.ylabel="pca2"
    #plt.scatter(pca[:,0],pca[:,1],color='r',marker='o')
    ax1.scatter(pca[:,0],pca[:,1],color='r',marker='o')
    plt.legend("ok")
    plt.show()
    
dir_files="G:\\ImageDetect\\TestPics\\"
os.chdir(dir_files)
files = os.listdir(dir_files)
num_f=len(files)
#y_all=np.arange(num_f)
y_all=np.ones((num_f,1),dtype=float)
for i in range(0,num_f):
    file_now=files[i]
    print(file_now)
    if file_now.find("b-")==0: #坏图片
        y_all[i]=-1
    #else:
    #    y_all[i]=1
    img=matplotlib.image.imread(file_now)
    if i==0:
        height,width=img.shape
        p=height*width
        #X_all=np.arange(num_f*p).reshape(num_f,p)
        X_all=np.zeros((num_f,p),dtype=float)
    img2=img.reshape(1,p,order="C") #按行转成向量,F:按列转成向量
    X_all[i,:]=img2

pca=PCA(n_components=2)
pca.fit(X_all)
x_pca=pca.transform(X_all)
PCA_Scatter(x_pca)

with open("y_X_all.csv","w",newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(y_all.T)
    #写入多行用writerows
    writer.writerows(X_all)
    

