import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.cluster import KMeans

def figshow(img,hsize=400,title='hoge'):
    """
    show a figure with defined size
    """
    shape = img.shape
    ratio=shape[0]/shape[1]
    img2= cv2.resize(img,(hsize,int(hsize*ratio)))
    cv2.imshow(title,img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def superposition(img_list,move_list):
    """
    ズレを表すdelta、引き算された画像、重なっている場所のみの画像、重なっている座標を出力する
    """

    rows, cols, col =img_list[0].shape

    #引き算される側に平行移動用の余白を付ける
    img1= np.zeros([rows*3,cols*3,col],np.int)
    img1[rows:rows*2,cols:cols*2]=img_list[1].astype(np.int)

    #move_listで指定された位置で引き算
    img1[rows+move_list[0]:rows*2+move_list[0], cols+move_list[1]:cols*2+move_list[1]] \
        = img1[rows+move_list[0]:rows*2+move_list[0], cols+move_list[1]:cols*2+move_list[1]] - img_list[0].astype(np.int)
    
    #縦横それぞれ次のリストの真ん中2つが取り出す範囲
    hlist=[rows,rows*2,rows+move_list[0],rows*2+move_list[0]]
    vlist=[cols,cols*2,cols+move_list[1],cols*2+move_list[1]]
    hlist.sort()
    vlist.sort()
    np.abs(img1)

    #取り出してズレに相当する量を計算
    trim = img1[hlist[1]:hlist[2],vlist[1]:vlist[2]]
    return img1.astype(np.uint8), trim.astype(np.uint8), [(hlist[1],hlist[2]),(vlist[1],vlist[2])]

def clastering(img,n_cluster=2):
    """
    指定した画像のクラスタリングを行い、ピクセルごとの判定結果を出力する
    """
    X=np.reshape(img,[-1,3]) #縦にピクセルを並べて横一列、深さRGBの2次元行列
    y_pred = KMeans(n_cluster).fit_predict(X) #cluster分類の結果を格納

    return np.reshape(y_pred,[img.shape[0],-1]).astype(np.uint8)

def binarization(img,collor=np.array([[0,0,0],[255,255,255]])):
    """
    指定した画像をクラスタリングにより2値化する
    """
    y_pred = clastering(img)
    #背景が黒になるように二値化
    X_pred = np.array([collor[i] for i in y_pred])
    if np.mean(y_pred==0) < 0.5:
        X_pred=1-X_pred
    
    return  np.reshape(X_pred,[img.shape[0],-1,3]).astype(np.uint8), y_pred


#def matching(diff,move_list):


img = cv2.imread('VHX_000032.jpg')
img_list=np.hsplit(img,2)
move_list = [100,-100]
shape = img_list[0].shape

"""
img_all, trim, trim_coord=superposition(img_list,move_list)

img_all=cv2.rectangle(img_all,trim_coord[0],trim_coord[1],(0,255,0),3)
delta = np.sqrt(np.square(trim).mean())/255
"""

#クラスタリングを用いた２値化
#clust_before, clustraw_before = clastering(cv2.resize(img_list[1] , (int(shape[1]*0.5), int(shape[0]*0.5))))
t=1
clust_after, clustraw_after = binarization(cv2.resize(img_list[0] , (int(shape[1]*t), int(shape[0]*t))))
dst = cv2.bitwise_and(clust_after, img_list[0])

#ラベリングとバウンディングボックス抽出
retval, labels, stats, centroids = cv2.connectedComponentsWithStats(clustraw_after)


"""
#輪郭抽出を用いた方法
label, contours= cv2.findContours(clustraw_after, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for i in range(0, len(contours)):
    if len(contours[i]) > 0:

        # remove small objects
        #if cv2.contourArea(contours[i]) < 500:
        #    continue

        cv2.polylines(img_list[0], contours[i], True, (255, 255, 255), 5)
"""

figshow(img_list[0])
#cv2.imwrite('./before.jpg', clust_before)
#cv2.imwrite('./after.jpg', clust_after)