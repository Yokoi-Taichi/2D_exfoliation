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
    引き算された画像、重なっている場所のみの画像、重なっている座標を出力する
    """

    rows, cols, col =img_list[0].shape

    #引き算される側に平行移動用の余白を付ける
    img1= np.zeros([rows*3,cols*3,col],np.int)
    img1[rows:rows*2,cols:cols*2]=img_list[1].astype(np.int)

    #move_listで指定された位置で引き算
    img1[rows+move_list[1]:rows*2+move_list[1], cols+move_list[0]:cols*2+move_list[0]] \
        = img1[rows+move_list[1]:rows*2+move_list[1], cols+move_list[0]:cols*2+move_list[0]] - img_list[0].astype(np.int)
    
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
    if np.mean(y_pred==0) < 0.5:
        y_pred=np.abs(1-y_pred)
    X_pred = np.array([collor[i] for i in y_pred])
    
    return  np.reshape(X_pred,[img.shape[0],-1,3]).astype(np.uint8), y_pred

def flakedetecting(clustraw):
    """
    connectedComponetsWithStatsの結果を面積によってソートして返す関数
    返り値：エリアの個数、ラベル、各エリアの統計、各エリアの重心
    """
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(clustraw)
    sort_indx=np.argsort(stats[:,cv2.CC_STAT_AREA])
    stats = stats[sort_indx[::-1]]
    centroids = centroids[sort_indx[::-1]]
    return retval, labels, stats, centroids

def detectedflakes(img, retval, labels, stats, centroids):
    """
    flakedetectingの関数から受け取った認識結果を画像に表示するプログラム
    """
    img_out=img.copy()
    bb=1 #何枚のフレークをハイライトするのか
    for i in range(1,1+bb):
        img_out = cv2.rectangle(img_out,\
            (stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP]),\
            (stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH],\
            stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]),(0,255,0),4)

    for i in range(retval):
        img_out = cv2.drawMarker(img_out,tuple(centroids[i].astype(np.int)),(0,0,255),markerType=cv2.MARKER_TILTED_CROSS, markerSize=30,thickness=2)

    return img_out

def tempmatch(before, after, stats,target):
    """
    画像を切り出し、テンプレートマッチングする。
    剥離前（検索対象）、剥離後(検索画像)、flakedetectingのstats、検索したいフレークの面積の順位
    """
    #templateの切り出し
    target_stats=stats[target]
    template = after[target_stats[cv2.CC_STAT_TOP]:target_stats[cv2.CC_STAT_TOP]+target_stats[cv2.CC_STAT_HEIGHT],\
        target_stats[cv2.CC_STAT_LEFT]:target_stats[cv2.CC_STAT_LEFT]+target_stats[cv2.CC_STAT_WIDTH]]
    
    #templatematching
    res = cv2.matchTemplate(before,template,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    #matchした領域を計算
    detect_topleft=(min_loc[0],min_loc[1])
    detect_buttomright=(detect_topleft[0]+template.shape[1],detect_topleft[1]+template.shape[0])


    return res, template, detect_topleft, detect_buttomright

if __name__ == '__main__':
    #画像を読み込み、transfer前後の2つに分割
    img = cv2.imread('VHX_000032.jpg')
    img_list=np.hsplit(img,2)
    shape = img_list[0].shape

    M = np.float32([[1,0,354],[0,1,-234]])
    img_list[1] = cv2.warpAffine(img_list[0],M,(shape[1],shape[0]))

    """
    img_all, trim, trim_coord=superposition(img_list,move_list)

    img_all=cv2.rectangle(img_all,trim_coord[0],trim_coord[1],(0,255,0),3)
    delta = np.sqrt(np.square(trim).mean())/255
    """

    #クラスタリングを用いた２値化
    #クラスタリングした画像でマッチング??
    #現在は1倍での特定にのみ対応
    t=1
    clust_after, clustraw_after = binarization(cv2.resize(img_list[0] , (int(shape[1]*t), int(shape[0]*t))))
    dst = cv2.bitwise_and(clust_after, img_list[0])

    #薄膜認識
    retval, labels, stats, centroids = flakedetecting(clustraw_after)

    #templateを切り出し、テンプレートマッチング
    target=2
    res, template, detect_topleft, detect_buttomright = tempmatch(img_list[1],img_list[0],stats,target)

    #マッチング結果を画像出力
    img_match=img_list[1].copy()
    img_match=cv2.rectangle(img_match,detect_topleft,detect_buttomright,(255,0,0),4)

    #どれだけ移動したかを計算
    hmove=detect_topleft[0]-stats[target,cv2.CC_STAT_LEFT]
    vmove=detect_topleft[1]-stats[target,cv2.CC_STAT_TOP]
    move_list=[hmove,vmove]

    #ずらして重ねる
    img_list[0] = detectedflakes(img_list[0], retval, labels, stats, centroids)
    img_all, trim, trim_coord=superposition(img_list,move_list)



    figshow(img_detect)

    #cv2.imwrite('./before.jpg', clust_before)
    #cv2.imwrite('./after.jpg', clust_after)