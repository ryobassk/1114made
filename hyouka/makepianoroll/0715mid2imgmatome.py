#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from PIL import Image
from pypianoroll import Multitrack, Track

filename=str('./midi/mXXk2.mid')
outfile = str('./sonota/mXXk2')

if __name__ == '__main__':

    for aaa in range(0,20):
        
        for bbb in range(0,9):
            filename_midi = filename.replace('XX', str(aaa)).replace('YY', str(bbb))
            print(filename_midi)

            if os.path.exists(filename_midi) == False :
                print("ファイル名.midは存在しません．")
                sys.exit()
            
            for nt in range(0,1):
                raw_mid = Multitrack(filename_midi, beat_resolution=16)
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
                filename_whole_pianoroll = outfile.replace('XX', str(aaa)).replace('YY', str(bbb)) + '_whole%d.jpg' % (nt)
                pil_img_f.save(filename_whole_pianoroll)
                

  
            
