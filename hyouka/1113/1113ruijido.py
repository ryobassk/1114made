#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
from PIL import Image
from pypianoroll import Multitrack, Track




if __name__ == '__main__':
    filename = []
    filename_midi = []
    filename_jpg = []
    
    
    filepath = './mididata/'
    outfilepath = './pianoroll/'
    midpath = '.mid'
    jpgpath = '.jpg'
    file_list = glob.glob('./mididata/*.mid')
    for i in range(len(file_list)):
        filename.append(os.path.splitext(os.path.basename(file_list[i]))[0])
    
    #print(filename)
 
    
    
    for i in range(len(filename)):       
        filename_midi.append(filepath  + filename[i] + midpath)
        print(filename_midi[i])  
        if os.path.exists(filename_midi[i]) == False :
            print("ファイル名.midは存在しません．")
            sys.exit()
        
        filename_jpg.append(outfilepath  + filename[i] + jpgpath)

    
    for i in range(len(filename)): 
        for nt in range(0,1):
            raw_mid = Multitrack(filename_midi[i], beat_resolution=16)
            raw_mid_track = raw_mid.tracks[nt]
            v_pianoroll = raw_mid_track.get_pianoroll_copy()
            #midiトラックを作成
            #Track()はシングル，Multitrack()はTrack()を複数組み合わせてマルチトラックにできる．
            track = Track(pianoroll=v_pianoroll, program=0, is_drum=False)
            track_save = Multitrack(tracks=[track,], tempo=183.0) #set tempo here
                
                
            #ピアノロール行列を転置(縦軸:音高，横軸:時間),画像用に正規化．
            h_pianoroll = v_pianoroll.T
            h_pianoroll = h_pianoroll * (float(255) / 80)
            #ピアノロール画像を保存
            pil_img_f = Image.fromarray(np.uint8(h_pianoroll))
            pil_img_f.save(filename_jpg[i])
                

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    import csv
    import glob
    
    #入力するファイルパス（作成したフレーズのピアノロールのパス）
    makeimg_name = []
    file_list = glob.glob('./pianoroll/*.jpg')
    for i in file_list:
        makeimg_name.append(i)
        
    #make_imgname.append()
    
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
    temimg_name = str('./tem/mAAAk2_whole0.jpg')
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
        
        for k in range(0,513): #0,513     256,513
            TM_maxScore = [0]*20
            template = makeimg_num[i][0 : 128, k : k+256]#(128,256)
            
            for n in range(len(temimg_num)):
                #templateと原曲でマッチング．相関係数の正規化指標を利用 
                temp_match = cv2.matchTemplate(temimg_num[n], template, cv2.TM_CCOEFF_NORMED)
                temp_match_max = np.amax(temp_match)
                TM_maxScore[n] = float(temp_match_max)
                
            matching_maxscore = max(TM_maxScore)
            TM_graph.append(matching_maxscore)
            
        TM_graph_matome.append(TM_graph)
    
    print('解析終了') 
    import os
    #################結果の出力start#####################################
    for i in range(len(TM_graph_matome)):
        #グラフの出力のパス
        path_plt = './result/result512_' + os.path.splitext(os.path.basename(file_list[i]))[0]+ '.jpg'
    
        # グラフの描画先の準備
        fig = plt.figure()
        # ここにグラフを描画する処理を記述する
        plt.plot(TM_graph_matome[i])
        plt.xlim([0,513])
        plt.ylim([0,1])
        #plt.title("", fontname="MS Gothic")   
        plt.xlabel("Slide[px]", fontname="MS Gothic")
        plt.ylabel("Similarity", fontname="MS Gothic")
        # ファイルに保存
        fig.savefig(path_plt)
    
    path_csv = './result/1116_result512.csv'
    with open(path_csv, 'w') as f0:
        for i in range(len(TM_graph_matome)):
            writer = csv.writer(f0)
            writer.writerow(['ruiji'+str(i)])
            writer.writerow(TM_graph_matome[i])
            writer.writerow([''])
    
    print('出力終了')
    #################結果の出力end##########################################   
      
                
