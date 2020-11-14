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
    
    
    filepath = './makemidi/'
    outfilepath = './result/'
    midpath = '.mid'
    jpgpath = '.jpg'
    file_list = glob.glob('C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/hyouka/makepianoroll/makemidi/*.mid')
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
                

  
            
