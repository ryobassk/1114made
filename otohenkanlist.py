import pretty_midi
import datetime
now = datetime.datetime.now()

##パラメータ

#差分検出の値
setsabun = 10  # 8以上
#差分の割合検出比率
setsabunratio = 0.1 # 0.055以上


#使われている音を配列の変換テキスト/見る用
fp_look = './data/sakusei/datamake/'+str(now.month)+'_'+str(now.day)+'otohenkantext.txt'
#使われている音を配列の変換テキスト/使う用
fp_use = './data/sakusei/datamake/'+str(now.month)+'_'+str(now.day)+'otohenkan.txt'
#使われている音を配列の変換テキスト/見る用
fo_look = open(fp_look, "w")
#使われている音を配列の変換テキスト/使う用
fo_use = open(fp_use, "w")
print('音変換保存先/見る用',fp_look)
print('音変換保存先/使う用',fp_use)


# MIDIファイルのロード
musicmidi_data = []
filename_num = []
# MIDIファイルのロード/m0k2からm19k2のmidiを読み込む

 
import glob
file_list = glob.glob('C:/Users/Ryo Ogasawara/OneDrive/lab/楽譜スコア/0930sakusei/10_5midi/Bb/*.mid')
for i in file_list:
    musicmidi_data.append( pretty_midi.PrettyMIDI(i))
    filename_num.append(i)
"""
filename = str('./data/midi/Bb2/mAAA-Part_1.mid')#str('./midi/Bb/Bb_mAAA.mid')
for i in range(20):
 filename_num.append( filename.replace('AAA', str(i)) )
 musicmidi_data.append( pretty_midi.PrettyMIDI(filename_num[i]) )
# MIDIファイルのロード/その他のmidiを読み込む
sonota_midiname = './data/midi/jyunban10607.mid'
musicmidi_data.append( pretty_midi.PrettyMIDI(sonota_midiname))
filename_num.append(sonota_midiname)
"""




##関数start
#同じリストを消去する
def Listsamedelete(seq):
    samelist = []
    return [x for x in seq if x not in samelist and not samelist.append(x)]

#音の長さが同じもので差分が小さいもののみリストに残す
def Listhenkan_hosei(seq):
    seq2 = seq
    for i in range(len(seq)):
        for k in range(len(seq2)):
            if seq2[k][1] == seq[i][1]:
                if (seq2[k][2] > seq[i][2]) and (seq[i][2] > 0):
                    seq2[k][0] = seq[i][0] 
                    seq2[k][2] = seq[i][2]          
    seq3 = Listsamedelete(seq2)
    seq4 = sorted(seq3, key=lambda x:(x[1], x[2]))
    return seq4


# MIDIファイルの数
music_num = len(musicmidi_data)
#トラックの取得
musicmidi_tracks = []
for i in range(music_num):
    musicmidi_tracks.append(musicmidi_data[i].instruments)
#ノートの取得
music_notes = []
for i in range(music_num):
    music_notes.append(musicmidi_tracks[i][1].notes)
length_info = []
oto_all_info = []
for i in range(music_num):#range(1):#
    print(str(i+1) + '曲目(' +str(filename_num[i]) + ')のtick確認')
    for k in range(len(music_notes[i])):
        oto_info = []
        start0_now = musicmidi_data[i].time_to_tick(music_notes[i][k].start)
        now_syousetsu = int(start0_now / 1920) + 1
        start_endtick = musicmidi_data[i].time_to_tick(music_notes[i][k].end - music_notes[i][k].start)
        if k < len(music_notes[i]) - 1:  
            start_starttick = musicmidi_data[i].time_to_tick(music_notes[i][k+1].start - music_notes[i][k].start)
        elif k >= len(music_notes[i]) - 1:
            start_starttick = now_syousetsu*1920 - start0_now  
        oto_info = [start_starttick,#音と次の音の長さ　0
                    start_endtick,#音の長さ 1
                    start_starttick-start_endtick,#音が鳴っていない時間 2
                    i,#曲番号 3
                    now_syousetsu,#小節数 4
                    start0_now - (now_syousetsu-1)*1920#拍数 5
                    ]
        if length_info.count(oto_info[0:3])==0:
            length_info.append(oto_info[0:3])
            oto_all_info.append(oto_info)
            

#使われている音を配列の内、音の長さのうち差分が最も小さいもののみ取る
length_taiou = Listhenkan_hosei(length_info)
tango_num = len(length_taiou)
print('\n今回の単語数:'+str(tango_num)+'\n')
for i in range(len(length_taiou)):
    startstart = length_taiou[i][0]
    startend = length_taiou[i][1]
    sabun = length_taiou[i][2]
    sabunratio = sabun / startend
    #誤った変換の可能性があるものを通知
    """
    if (startstart==1200 and sabun==13) or (startstart==2640 and sabun==133):
        print('差分の割合:' + str(sabunratio) + '\n'
              +'差分が' + str(setsabun) + 'より大きく、' +'差分の割合が' 
              + str(setsabunratio) + 'より大きいものがあります。' + '\n'
              +'start-start:' + str(startstart) + '  start-end:' 
              + str(startend) + '  sabun:' + str(sabun))
        for k in range(len(oto_all_info)):
            #どこによくないやつがあるか確認するやつ
            if (oto_all_info[k][2] == sabun) and (oto_all_info[k][1] == startend):
                print('以下の曲を確認してください' + '\n'
                      + str(oto_all_info[k][3]+1)+'曲目:  '+str(filename_num[oto_all_info[k][3]])+ '\n'
                      +'小節:'+str(oto_all_info[k][4])+ '\n'
                      +'拍数（tick）:'+str(oto_all_info[k][5])+ '\n')
    """ 
    if sabun > setsabun and sabunratio > setsabunratio:
        print('差分の割合:' + str(sabunratio) + '\n'
              +'差分が' + str(setsabun) + 'より大きく、' +'差分の割合が' 
              + str(setsabunratio) + 'より大きいものがあります。' + '\n'
              +'start-start:' + str(startstart) + '  start-end:' 
              + str(startend) + '  sabun:' + str(sabun)) 
        for k in range(len(oto_all_info)):
            #どこによくないやつがあるか確認するやつ
            if (oto_all_info[k][2] == sabun) and (oto_all_info[k][1] == startend):
                print('以下の曲を確認してください' + '\n'
                      + str(oto_all_info[k][3]+1)+'曲目:  '+str(filename_num[oto_all_info[k][3]])+ '\n'
                      +'小節:'+str(oto_all_info[k][4])+ '\n'
                      +'拍数（tick）:'+str(oto_all_info[k][5])+ '\n')
                
    if sabun < 0:
        print('差分の割合:' + str(sabunratio) + '\n'
              +'差分が' + str(setsabun) + 'より大きく、' +'差分の割合が' 
              + str(setsabunratio) + 'より大きいものがあります。' + '\n'
              +'start-start:' + str(startstart) + '  start-end:' 
              + str(startend) + '  sabun:' + str(sabun)) 
        for k in range(len(oto_all_info)):
            #どこによくないやつがあるか確認するやつ
            if (oto_all_info[k][2] == sabun) and (oto_all_info[k][1] == startend):
                print('以下の曲を確認してください' + '\n'
                      + str(oto_all_info[k][3]+1)+'曲目:  '+str(filename_num[oto_all_info[k][3]])+ '\n'
                      +'小節:'+str(oto_all_info[k][4])+ '\n'
                      +'拍数（tick）:'+str(oto_all_info[k][5])+ '\n')
    #出力
    fo_look.write(' start-start:' + str(startstart) 
                  + '  start-end:' + str(startend) 
                  + '  sabun:' + str(sabun) + '\n' + '\n')
    fo_use.write(str(startstart)+'_'+str(startend)+'_'+str(sabun)+ '\n')
fo_look.close()
fo_use.close()


