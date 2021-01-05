import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

def OneCH2TreeCH(image_1ch,labels=2):
    """
    ピクセルごとのラベル表示の1チャンネル画像から、BGR3チャンネルの画像への変換
    """

    step = step=255/(labels-1)
    color=np.array([[int(i*step),int(i*step),int(i*step)] for i in range(labels)]) #ラベルの階数のカラーパレット

    image_3ch=np.array([color[i] for i in image_1ch]) #BGR3チャンネルの行列に変換
    image_3ch=np.reshape(image_3ch,[image_1ch.shape[0],-1,3]).astype(np.uint8) #画像として成形
    return image_3ch

def highlightbinalimage(image, binalImage_3ch):
    """
    認識結果の2値化画像がどこを認識しているのかハイライトする
    """
    alpha = 0.7
    binalImage_3ch[np.where((binalImage_3ch==[255,255,255]).all(axis=2))]=[0,255,0]
    blended=cv2.addWeighted(image,alpha,binalImage_3ch,1-alpha,0)  
    return blended  

def figshow(img_original,labels=None,hsize=400,title='hoge'):
    """
    画像を指定したサイズで表示。各ピクセルごとのlabelを格納している場合にはラベルの総数も指定。
    """
    if labels != None: #1チャンネル画像の場合
        img=OneCH2TreeCH(img_original,labels)
    else: #3チャンネル画像の場合
        img=img_original

    shape = img.shape
    ratio=shape[0]/shape[1]
    img2= cv2.resize(img,(hsize,int(hsize*ratio))) #画像を指定サイズへリサイズ
    cv2.imshow(title,img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def savefigs():
    """
    必要な画像を適宜保存する。
    """
    cv2.imwrite(path+"before_scalebar.png", img_before_scales.bar_dictionally["binalTarget"])
    cv2.imwrite(path+"after_scalebar.png", img_after_scales.bar_dictionally["binalTarget"])
    

    cv2.imwrite(path+"after_clustImage.png", after_clust.clustImage_3ch)
    cv2.imwrite(path+"after_binalImage.png", after_clust.binalImage_3ch)
    cv2.imwrite(path+"after_binalhighlight.png", highlightbinalimage(img_after,after_clust.binalImage_3ch))

    cv2.imwrite(path+"after_flakedetecting.png", after_flakes.Detectedobjects(img_after))
    cv2.imwrite(path+"before_clustlImage.png", before_clust.clustImage_3ch)
    cv2.imwrite(path+"before_binalImage.png", before_clust.binalImage_3ch)
    cv2.imwrite(path+"before_binalhighlight.png", highlightbinalimage(img_before,before_clust.binalImage_3ch))

    cv2.imwrite(path+"match_matchingresult.png", matching.TM_image)
    cv2.imwrite(path+"match_template.png", matching.TM_dict["template"])
    cv2.imwrite(path+"match_trim.png", matching.SP_trim_3ch)
    cv2.imwrite(path+"match_highlight.png", matching.highlightresult(img_before))

def bgrExtraction(image, bgrLower, bgrUpper):
    """
    指定した画像の指定された範囲の色の領域を抽出する関数
    """
    img_mask = cv2.inRange(image, bgrLower, bgrUpper) # BGRからマスクを作成
    result = cv2.bitwise_and(image, image, mask=img_mask) # 元画像とマスクを合成
    return result

class Cluster():
    def __init__(self, image, n_cluster=2):
        self.image=image #クラスタリングする画像
        self.n_cluster=n_cluster #何段階にクラスタリングするか
        self.clustImage_1ch=self.clustering() #クラスターのラベルが各ピクセルに割り当てられた1ch画像
        self.clustImage_3ch=OneCH2TreeCH(self.clustImage_1ch,labels=self.n_cluster) #クラスタリング結果の3ch画像
        self.binalImage_1ch= self.binarization() #クラスタリングの結果を元に背景=0,それ以外=1の2値化した1ch画像
        self.binalImage_3ch=OneCH2TreeCH(self.binalImage_1ch) #3chの2値化画像

    def clustering(self):
        """
        指定した画像のクラスタリングを行い、ピクセルごとの判定結果を出力する
        """
        n_cluster=self.n_cluster
        X=np.reshape(self.image,[-1,3]) #縦にピクセルを並べて横一列、深さRGBの2次元行列
        y_pred = KMeans(n_cluster).fit_predict(X) #cluster分類の結果を格納

        return np.reshape(y_pred,[self.image.shape[0],-1]).astype(np.uint8)

    def binarization(self):
        """
        指定した画像をクラスタリングにより2値化する
        """
        y_pred = self.clustImage_1ch
        #背景が黒になるように二値化
        nlabels=[np.sum(y_pred == i) for i in range(self.n_cluster)]
        background_label = nlabels.index(max(nlabels)) #背景のラベルを特定(背景が最大面積だと仮定)
        
        y_binal = np.array([0 if i==background_label else 1 for i in np.reshape(y_pred,-1)]) #2値化
        y_binal = np.reshape(y_binal,[y_pred.shape[0],-1]).astype(np.uint8) #成型
        
        return  y_binal

class ObjectDetecting():
    """
    1チャンネルの2値画像を入力して薄膜の認識結果を返す。
    """
    def __init__(self, binalImage, Image=None):
        self.Image= Image #参考画像
        self.binalImage=binalImage #object認識をする2値化画像
        self.dictionally=self.Detecting() #認識結果をまとめた辞書

    def Detecting(self):
        """
        connectedComponetsWithStatsの結果を面積によってソートして返す関数
        返り値：エリアの個数、ラベル、各エリアの統計、各エリアの重心
        """
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.binalImage) #Object認識
        sort_indx=np.argsort(stats[:,cv2.CC_STAT_AREA])
        stats = stats[sort_indx[::-1]] #面積が大きい順に認識したオブジェクトをソート
        centroids = centroids[sort_indx[::-1]]
        result_dict={"nlabels":nlabels, #認識したオブジェクトの総数
                    "labels":labels, #ピクセルごとのオブジェクトの割り当て
                    "stats":stats,# オブジェクトごとの情報の配列
                    "centroids":centroids} #各オブジェクトの重心
        return result_dict

    def Detectedobjects(self,highlightedImage=None):
        """
        flakedetectingの関数から受け取った認識結果を画像に表示するプログラム
        """
        if highlightedImage.any() == None: #指定がない場合2値化画像に表示
            highlightedImage = OneCH2TreeCH(self.binalImage)

        img_out=highlightedImage.copy()
        stats=self.dictionally["stats"]
        nlabels=self.dictionally["nlabels"]
        centroids = self.dictionally["centroids"]
        bb=1 #何枚のフレークを箱でハイライトするのか
        for i in range(1,1+bb): #箱でハイライト
            img_out = cv2.rectangle(img_out,\
                (stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP]),\
                (stats[i,cv2.CC_STAT_LEFT]+stats[i,cv2.CC_STAT_WIDTH],\
                stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]),(0,255,0),4)

        for i in range(nlabels): #重心に×印
            img_out = cv2.drawMarker(img_out,tuple(centroids[i].astype(np.int)),(0,0,255),markerType=cv2.MARKER_TILTED_CROSS, markerSize=30,thickness=2)

        return img_out

class ImageScales():
    def __init__(self, image,nm_scale):
        self.image=image
        self.nmscale=nm_scale #画像のスケールbarの長さ

        self.TargetSizeRanking=1 #どの面積順位のオブジェクトをキャプションやバーだと思って抽出するか
        self.caption_COLOR= np.array([[240,240,240],   #scale のキャプション部分の色（下限）
                                [255,255,255]])  #上限
        self.bar_COLOR= np.array([[150,0,0],   #scale のキャプション部分の色（下限）
                                [255,50,50]])  #上限
        
        self.caption_dictionally = self.getRegionsByColor(self.caption_COLOR) #キャプション部分の抽出結果をまとめた辞書
        self.bar_dictionally = self.getRegionsByColor(self.bar_COLOR) #スケールバー部分の抽出結果をまとめた辞書
        self.bar_Pixelwidth = self.bar_dictionally["stats"][self.TargetSizeRanking,cv2.CC_STAT_WIDTH] #スケールバーが幅何ピクセルか
        self.nmParPixel = self.nmscale/self.bar_Pixelwidth #1ピクセルが何ナノメートルか

    def getRegionsByColor(self,color):
        """
        指定された色領域で抽出し、全体の2値化画像、オブジェクトが存在する部分の2値化画像を辞書で返す
        """
        binal = cv2.inRange(self.image, color[0],color[1]) #指定した色領域で抽出、2値化
        dictionally = ObjectDetecting(binal).dictionally #オブジェクト認識
        stats=dictionally["stats"][self.TargetSizeRanking]
        #ターゲットがある部分の2値化画像を切り出す
        target = binal[stats[cv2.CC_STAT_TOP]:stats[cv2.CC_STAT_TOP]+stats[cv2.CC_STAT_HEIGHT],\
            stats[cv2.CC_STAT_LEFT]:stats[cv2.CC_STAT_LEFT]+stats[cv2.CC_STAT_WIDTH]]
        
        dictionally["binalImage"]=binal
        dictionally["binalTarget"]=target

        return dictionally

class templatematching():
    """
    「消えた」薄膜を認識する。
    3チャンネル(BGR)のbefore(マッチング対象)、after(ターゲット元),ObjectDetectingのstats(ターゲットの情報)
    """
    def __init__(self,before_image,after_image, target_stats):
        self.before=before_image
        self.after = after_image
        self.target_stats=target_stats #ターゲットを取り出すためのターゲットについての情報

        #テンプレートマッチング(TM)の結果に関するインスタンス変数
        self.TM_dict=self.tempmatch() #マッチング結果をまとめた辞書
        self.TM_image = self.TM_dict["extended"] #マッチング対象のどこにターゲットが来るか？
        self.move_list =self.move() #2つの画像はどれだけ並進しているか   

        #画像の重ね合わせ(SP)の結果に関するインスタンス変数
        # allは全体、trimは重なった部分、
        self.SP_all_3ch, self.SP_trim_3ch, self.SP_dict = self.superposition()
        # trimの結果を2値化
        self.SP_trim_binal = np.array([0 if np.allclose(i, np.array([0,0,0])) else 1 for i in self.SP_trim_3ch])

    def tempmatch(self):
        """
        画像を切り出し、テンプレートマッチングする。
        剥離前（検索対象）、剥離後(検索画像)、flakedetectingのstats、検索したいフレークの面積の順位
        """
        #templateの切り出し
        target_stats=self.target_stats
        after=self.after
        template = after[target_stats[cv2.CC_STAT_TOP]:target_stats[cv2.CC_STAT_TOP]+target_stats[cv2.CC_STAT_HEIGHT],\
            target_stats[cv2.CC_STAT_LEFT]:target_stats[cv2.CC_STAT_LEFT]+target_stats[cv2.CC_STAT_WIDTH]]
        
        #templateの大きさまで検索対象に余白をつける
        img_ext = np.zeros([self.before.shape[0]+2*template.shape[0],self.before.shape[1]+2*template.shape[1],3],np.uint8)
        img_ext[template.shape[0]:template.shape[0]+self.before.shape[0],template.shape[1]:template.shape[1]+self.before.shape[1]]=self.before
        img_ext=img_ext.astype(np.uint8)

        #templatematching
        match = cv2.matchTemplate(img_ext,template,cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
        
        #matchした領域を計算(beforeの画像の右上頂点をゼロとした座標)
        detect_topleft=(min_loc[0]-template.shape[1],min_loc[1]-template.shape[0])
        detect_buttomright=min_loc
        #matchした場所に枠を表示
        img_ext=cv2.rectangle(img_ext,min_loc,(min_loc[0]+template.shape[1],min_loc[1]+template.shape[0]),(255,0,0),4)

        result_dict={"matching_result": match,
                        "extended":img_ext,
                        "template":template,
                        "detect_topleft":detect_topleft,
                        "detect_buttomright":detect_buttomright}

        return result_dict
    
    def move(self):
        detect_topleft=self.TM_dict["detect_topleft"]
        stats=self.target_stats
        #どれだけ移動したかを計算
        hmove=detect_topleft[0]-stats[cv2.CC_STAT_LEFT]
        vmove=detect_topleft[1]-stats[cv2.CC_STAT_TOP]
        return [hmove,vmove]

    def superposition(self):
        """
        引き算された画像、重なっている場所のみの画像、重なっている座標を出力する
        """
        img_before = self.before
        img_after = self.after

        b_rows, b_cols, b_col = img_before.shape
        a_rows, a_cols, a_col = img_after.shape
        
        #引き算される側に平行移動用の余白を付ける
        img_all= np.zeros([b_rows*3,b_cols*3,b_col],np.int)
        img_all[b_rows:b_rows*2,b_cols:b_cols*2]=img_before.astype(np.int)

        #move_listで指定された位置で引き算
        move_list=self.move_list
        img_all[b_rows+move_list[1]:b_rows+a_rows+move_list[1], b_cols+move_list[0]:b_cols+a_cols+move_list[0]] \
            = img_all[b_rows+move_list[1]:b_rows+a_rows+move_list[1], b_cols+move_list[0]:b_cols+a_cols+move_list[0]] - img_after.astype(np.int)
        np.abs(img_all)

        #縦横それぞれ次のリストの真ん中2つが取り出す範囲
        hlist=[b_rows,b_rows*2,b_rows+move_list[1],b_rows+a_rows+move_list[1]]
        vlist=[b_cols,b_cols*2,b_cols+move_list[0],b_cols+a_cols+move_list[0]]
        hlist.sort()
        vlist.sort()

        #重なった部分を切り出し
        trim = img_all[hlist[1]:hlist[2],vlist[1]:vlist[2]]
        return img_all.astype(np.uint8), trim.astype(np.uint8), \
            {"trim_topleft":(hlist[1]-b_rows,vlist[1]-b_cols),"trim_buttomright":(hlist[2]-b_rows,vlist[2]-b_cols)}

    def highlightresult(self, original_before):
        """
        重ね合わせによって得られた結果を元画像上に強調表示
        """
        trim_original= original_before[self.SP_dict["trim_topleft"][0]:self.SP_dict["trim_buttomright"][0],\
            self.SP_dict["trim_topleft"][1]:self.SP_dict["trim_buttomright"][1]]
        trim = self.SP_trim_3ch.copy()
        blended = highlightbinalimage(trim_original,trim)
        return blended

if __name__ == '__main__':
    """
    path="./test_1/"
    img = cv2.imread(path+'VHX_000032.jpg')
    img_list=np.hsplit(img,2)
    img_before=img_list[1]
    img_after=img_list[0]
    """
    path="./test_4/"
    img_before = cv2.imread(path+'before.jpg')
    img_after = cv2.imread(path+'after.jpg')
    
    shape_before = img_before.shape

    print("scale recognition")
    img_before_scales = ImageScales(img_before,10e3) #スケール認識
    img_after_scales = ImageScales(img_after,10e3) #スケール認識  
    expantion =img_before_scales.nmParPixel/img_after_scales.nmParPixel
    img_before = cv2.resize(img_before,(int(shape_before[1]*expantion),int(shape_before[0]*expantion)))
    print(f"expantion ratio = {expantion: .3f}")

    print("clustering")
    after_clust=Cluster(img_after,n_cluster=8) #クラスタリング
    before_clust=Cluster(img_before,n_cluster=8) #クラスタリング

    print("thin film detection")
    after_flakes=ObjectDetecting(after_clust.binalImage_1ch,img_after) #薄膜認識
    print("templatematching")
    matching = templatematching(before_clust.binalImage_3ch,after_clust.binalImage_3ch,after_flakes.dictionally["stats"][1]) # テンプレートマッチング
    
    print("saving result pictures")
    savefigs()
    
    
    
    
    
    
    