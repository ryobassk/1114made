import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv


#入力するファイルパス（作成したフレーズのピアノロールのパス）
makeimg_name = []
makeimg_name.append('./make_img/h100epo300_5_all.jpg')
makeimg_name.append('./make_img/h100epo300_5_all2.jpg')
makeimg_name.append('./make_img/h100epo300_5_all3.jpg')
#make_imgname.append()

#出力パス（csv）
path_csv = './result/0.5gakkai_result512.csv'


#フレーズ生成結果ピアノロール画像を読み込み
makeimg_num =[]
print('入力ファイル')
for i in range(len(makeimg_name)): 
    print(makeimg_name[i])
    #フレーズ生成結果ピアノロール画像を読み込み
    makeimg_import= cv2.imread(makeimg_name[i])
    #グレースケール変換
    makeimg_gray = cv2.cvtColor(makeimg_import, cv2.COLOR_BGR2GRAY)
    #画像の上下反転
    makeimg_gray = cv2.flip(makeimg_gray, 0)
    #保存
    makeimg_num.append(makeimg_gray)


#学習した曲のピアノロールを読み込む
temimg_num =[]
temimg_name = str('./template_img/mAAAk2_whole0.jpg')
print('比較ファイル')
for i in range(20):
    #入力画像を読み込み
    temimg_import = cv2.imread(temimg_name.replace('AAA', str(i)))
    print(temimg_name.replace('AAA', str(i)))
    #グレースケール変換
    temimg_gray = cv2.cvtColor(temimg_import, cv2.COLOR_BGR2GRAY)
    #画像の上下反転
    temimg_gray = cv2.flip(temimg_gray, 0)
    #保存
    temimg_num.append(temimg_gray)
    

#結果の入力配列
TM_graph_matome = []

#テンプレートマッチング
for i in range(len(makeimg_num)):
    print('作成フレーズ:',str(i+1)+'曲目')
    TM_graph = []
    
    for k in range(0,737): #0,513     256,513
        TM_maxScore = [0]*20
        template = makeimg_num[i][0 : 128, k : k+32]#(128,256)
        
        for n in range(len(temimg_num)):
            #templateと原曲でマッチング．相関係数の正規化指標を利用 
            temp_match = cv2.matchTemplate(temimg_num[n], template, cv2.TM_CCOEFF_NORMED)
            temp_match_max = np.amax(temp_match)
            TM_maxScore[n] = float(temp_match_max)
            
        matching_maxscore = max(TM_maxScore)
        TM_graph.append(matching_maxscore)
        
    TM_graph_matome.append(TM_graph)
    
    path_plt = './result/0.5gakkai_result512_' + str(i)+ '.jpg'
    # グラフの描画先の準備
    fig = plt.figure()
    # ここにグラフを描画する処理を記述する
    plt.plot(TM_graph_matome[i])
    #plt.xlim([0,513])
    plt.ylim([0,1])
    #plt.title("", fontname="MS Gothic")   
    plt.xlabel("マッチングウィンドウのスライド量[px]", fontname="MS Gothic")
    plt.ylabel("類似度", fontname="MS Gothic")
    # ファイルに保存
    fig.savefig(path_plt)
    
    with open(path_csv, 'w') as f0:
        for i in range(len(TM_graph_matome)):
            writer = csv.writer(f0)
            writer.writerow(['ruiji'+str(i)])
            writer.writerow(TM_graph_matome[i])
            writer.writerow([''])


print('解析終了') 


   
