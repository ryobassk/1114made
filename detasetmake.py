import pretty_midi
import datetime
now = datetime.datetime.now()
#for文の変数の順i,k,l,n,p

##パラメータ
# MIDIファイルのロード
musicmidi_data = []
filename_num = []
import glob
# MIDIファイルのロードパス/その他のmidiを読み込む
sonota_midiname = 'C:/Users/Ryo Ogasawara/OneDrive/lab/楽譜スコア/0930sakusei/bannsou/10_5midi/Bb/Another_Hairdo.mid'
musicmidi_data.append( pretty_midi.PrettyMIDI(sonota_midiname))
filename_num.append(sonota_midiname)

"""    
#file_list = glob.glob('C:/Users/Ryo Ogasawara/OneDrive/lab/楽譜スコア/0930sakusei/10_5midi/Bb/*.mid')
file_list = glob.glob('C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/data/midi/Bbmidi1019/10_2midi/*.mid')
for i in file_list:
    musicmidi_data.append( pretty_midi.PrettyMIDI(i))
    filename_num.append(i)


filename = str('./midi/Bb/Bb_mAAA.mid')
for i in range(20):
 filename_num.append(filename.replace('AAA', str(i)))
 musicmidi_data.append( pretty_midi.PrettyMIDI(filename_num[i])  )
# MIDIファイルのロードパス/その他のmidiを読み込む
sonota_midiname = 'C:/Users/Ryo Ogasawara/OneDrive/lab/楽譜スコア/0930sakusei/blues/10_2/Another_Hairdo.mid'
musicmidi_data.append( pretty_midi.PrettyMIDI(sonota_midiname))
filename_num.append(sonota_midiname)
"""



##関数
#同じリストを消去する
def Listsamedelete(seq):
    samelist = []
    return [x for x in seq if x not in samelist and not samelist.append(x)]
#タイを削除
def Delete_tai(seq, length_hosei):
    for i in range(len(seq)):
        k = 0
        while k<len(seq[i]):#range(len(seq[i])):#
            start_tick = seq[i][k][3]-(seq[i][k][2]-1)*1920
            if start_tick + seq[i][k][1] > int(start_tick/length_hosei +1)*length_hosei:
                separate1 = [seq[i][k][0],
                             int(start_tick/length_hosei +1)*length_hosei-start_tick,
                             seq[i][k][2],
                             seq[i][k][3]
                             ]
                length_save = int(start_tick/length_hosei +1)*length_hosei-start_tick
                separate2 = [seq[i][k][0],
                             start_tick + seq[i][k][1] - int(start_tick/length_hosei +1)*length_hosei,
                             int((seq[i][k][3]+length_save)/1920 +1),
                             seq[i][k][3]+length_save
                             ]                
                if len(seq[i][k]) > 4:#コードの構成音を入れる
                    separate1.append(seq[i][k][4])
                    separate2.append(seq[i][k][4])
                seq[i][k:k+1]=[separate1,separate2]
            else:
                k+=1    
    return seq


##MIDI読み込み
#MIDIファイルの数
music_num = len(musicmidi_data)
#トラックの取得
musicmidi_tracks = []
for i in range(music_num):
    musicmidi_tracks.append(musicmidi_data[i].instruments)
#ノートの取得
mero_notes = []
chord_notes = []
for i in range(music_num):
    mero_notes.append(musicmidi_tracks[i][0].notes)  
    chord_notes.append(musicmidi_tracks[i][1].notes)  


##使用コードの読み込み
dict_chord_info = [] #使用したコード
for i in range(music_num):
    start0_now = 0
    chord_oto = [] #一つのコード
    for k in range(len(chord_notes[i])):
        if start0_now == musicmidi_data[i].time_to_tick(chord_notes[i][k].start):
            chord_oto.append(chord_notes[i][k].pitch)       
        else:
            chord_henkan = []
            for l in range(len(chord_oto)):
                chord_henkan.append(chord_oto[l]-chord_oto[0])
            dict_chord_info.append(chord_henkan)
            chord_oto = [] #初期化
            chord_oto.append(chord_notes[i][k].pitch)           
        start0_now = musicmidi_data[i].time_to_tick(chord_notes[i][k].start)
dict_chord_info = Listsamedelete(dict_chord_info)
dict_chord_info = sorted(dict_chord_info, reverse=False, key=lambda x: x[0])

fdict_chord = open('C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/data/dataset/10_6/'+str(now.month)+'_'+str(now.day)+'dict_chord.txt', 'w')
for i in range(len(dict_chord_info)):
    fdict_chord.write(str(i)+'\t')
    for k in range(len(dict_chord_info[i])):
        fdict_chord.write(str(dict_chord_info[i][k])+'_')
    fdict_chord.write('\n')    
fdict_chord.close()

##音の長さ変換の辞書作成
fp = './data/siyou/0916otohenkan.txt'
f_read = open(fp, "r")
otohenkanlist = f_read.readlines()
for i in range(len(otohenkanlist)):
    otohenkanlist[i] = otohenkanlist[i].strip()
f_read.close()
otohenkanjisyo= {}

for i in range(len(otohenkanlist)):
    henkango, henkanmae, _ = otohenkanlist[i].split('_')
    otohenkanjisyo[int(henkanmae)] = int(henkango)


##メロディとコード情報を取得
mero_all = []
chord_all = []
for i in range(music_num):#(music_num):
    #メロディ情報を曲ごとに取得
    hakusuu_count = 0
    mero_oto = []
    print(str(i+1) + '曲目(' +str(filename_num[i]) + ')のtick確認')
    for k in range(len(mero_notes[i])):
        #休符追加
        if hakusuu_count != musicmidi_data[i].time_to_tick(mero_notes[i][k].start):#休符追加
            mero_oto.append([-1,#音階
                         musicmidi_data[i].time_to_tick(mero_notes[i][k].start) - hakusuu_count,#音の長さ
                         int(hakusuu_count/1920) + 1,#何小節
                         hakusuu_count#現在の音が鳴るまでの長さ
                         ])
            hakusuu_count += musicmidi_data[i].time_to_tick(mero_notes[i][k].start) - hakusuu_count
            
        start0_now = musicmidi_data[i].time_to_tick(mero_notes[i][k].start)#最初から現在の音が鳴るまでの長さ
        syousetsu = int(start0_now/1920) + 1
        length = musicmidi_data[i].time_to_tick(mero_notes[i][k].end - mero_notes[i][k].start)#変換前の音の長さ
        mero_oto.append([mero_notes[i][k].pitch,#音階
                         otohenkanjisyo[length],#音の長さ
                         syousetsu,#何小節
                         start0_now#現在の音が鳴るまでの長さ
                         ])
        hakusuu_count += otohenkanjisyo[length]
        #曲の終わりに休符があるときに休符情報を追加
        if k==len(mero_notes[i])-1 and (start0_now+otohenkanjisyo[length])%1920!=0:
            if syousetsu!=int((start0_now+otohenkanjisyo[length])/1920):
                mero_oto.append([-1,#音階
                                 syousetsu*1920 - (start0_now+otohenkanjisyo[length]),#音の長さ
                                 syousetsu,#何小節
                                 start0_now+otohenkanjisyo[length]#現在の音が鳴るまでの長さ
                                 ])
            else:
                syousetsu_hosei=int((start0_now+otohenkanjisyo[length])/1920)+1
                mero_oto.append([-1,#音階
                                 syousetsu_hosei*1920 - (start0_now+otohenkanjisyo[length]),#音の長さ
                                 syousetsu_hosei,#何小節
                                 start0_now+otohenkanjisyo[length]#現在の音が鳴るまでの長さ
                                 ])
    mero_all.append(mero_oto)


    
    #コード情報を曲ごとに取得
    start0_now = 0
    hakusuu_count = 0
    chord_oto = []
    chord_oto_one = [] #一つのコード
    for k in range(len(chord_notes[i])):
        if hakusuu_count != start0_now:#休符追加
            mero_oto.append([-1,#音階
                             musicmidi_data[i].time_to_tick(chord_notes[i][k].start) - hakusuu_count,#音の長さ
                             int(start0_now/1920) + 1,#何小節
                             hakusuu_count#現在の音が鳴るまでの長さ
                             -1,#構成音
                             ])
            hakusuu_count += musicmidi_data[i].time_to_tick(chord_notes[i][k].start) - hakusuu_count  
            
        if start0_now == musicmidi_data[i].time_to_tick(chord_notes[i][k].start):
            chord_oto_one.append(chord_notes[i][k].pitch)       
        else:
            syousetsu = int(start0_now/1920) + 1
            length = musicmidi_data[i].time_to_tick(chord_notes[i][k-1].end - chord_notes[i][k-1].start)#変換前の音の長さ
            chord_henkan = []#ルート音と何度離れているか
            for l in range(len(chord_oto_one)):
                chord_henkan.append(chord_oto_one[l] - chord_oto_one[0])
            
            chord_oto.append([chord_oto_one[0],#コード 0
                             otohenkanjisyo[length],#音の長さ 1
                             syousetsu,#何小節 2
                             start0_now,#現在の音が鳴るまでの長さ 3
                             dict_chord_info.index(chord_henkan)#コード構成のID 4
                             ])
            chord_oto_one = [] #初期化 
            chord_oto_one.append(chord_notes[i][k].pitch)
            hakusuu_count += otohenkanjisyo[length]
            
        start0_now = musicmidi_data[i].time_to_tick(chord_notes[i][k].start)
    syousetsu = int(start0_now/1920) + 1
    length = musicmidi_data[i].time_to_tick(chord_notes[i][k-1].end - chord_notes[i][k-1].start)#変換前の音の長さ
    chord_henkan = []#ルート音と何度離れているか
    for l in range(len(chord_oto_one)):
        chord_henkan.append(chord_oto_one[l] - chord_oto_one[0])
    chord_oto.append([chord_oto_one[0],#コード 0
                      otohenkanjisyo[length],#音の長さ 1
                      syousetsu,#何小節 2
                      start0_now,#現在の音が鳴るまでの長さ 3
                      dict_chord_info.index(chord_henkan)#コード構成のID 4
                      ])
    chord_all.append(chord_oto)
##タイを取り除く
mero_all = Delete_tai(mero_all, 960)
chord_all = Delete_tai(chord_all, 960)

          
##コードとメロディを合成
mero_chord_all=[]
mero_chord_info=[]
for i in range(len(chord_all)):#range(0,1):#
    l = 0
    mero_chord_info=[]
    for k in range(len(chord_all[i])):#range(0,20):#
        while chord_all[i][k][1]+chord_all[i][k][3] > mero_all[i][l][3]:
            mero_chord_info.append([mero_all[i][l][0],#音階
                                   mero_all[i][l][1],#長さ
                                   mero_all[i][l][2],#小節
                                   mero_all[i][l][3],#初めからのTick数
                                   chord_all[i][k][0],#コードのルート音
                                   chord_all[i][k][4]#コードの構成音
                                   ])
            l+=1
            if l==len(mero_all[i]):
                break
    mero_chord_all.append(mero_chord_info)


##コーパスの作成
call_all = []
#音を一小節でまとめる
for i in range(len(mero_chord_all)):
    call_info = []
    k=0
    check = True
    while len(mero_chord_all[i]) > k:
        call_syousetsu=[]
        now_syousetsu=mero_chord_all[i][k][2]
        while now_syousetsu+1 > mero_chord_all[i][k][2]:
            call_syousetsu.append(mero_chord_all[i][k])
            #print(k)
            #print(mero_chord_all[i][k])
            k+=1
            if k ==len(mero_chord_all[i]):
                break
        if check:
            call_info.append(call_syousetsu)
    call_all.append(call_info)

syousetsu_num = 4 #まとめる小節数
slide_num = 4 #スライドする小節数
#[曲][小節の塊][音]
call_text_kyoku = []
resp_text_kyoku = []
#[小節の塊][音]
call_text = []
resp_text = []
#[音]
call_textinfo = []
resp_textinfo = []

for i in range(len(call_all)):
    k=0
    while k < len(call_all[i]):
        for l in range(0, syousetsu_num):
            if  len(call_all[i]) >= k + 2*syousetsu_num:
                call_textinfo.extend(call_all[i][k+l])
                resp_textinfo.extend(call_all[i][k+l+syousetsu_num])  
        if  len(call_all[i]) >= k + 2*syousetsu_num:
            call_text.append(call_textinfo)
            resp_text.append(resp_textinfo)
        call_textinfo = []
        resp_textinfo = []
        k+=slide_num
    call_text_kyoku.append(call_text)
    resp_text_kyoku.append(resp_text)
    call_text = []
    resp_text = []


#出力
f_call = 'C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/'+str(now.month)+'_'+str(now.day)+'kyokucall10_6test.txt'
f_resp = 'C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/'+str(now.month)+'_'+str(now.day)+'kyokuresp10_6test.txt'
fo_call = open(f_call, "w")
fo_resp = open(f_resp, "w")
 
for i in range(len(call_text_kyoku)):
    for k in range(len(call_text_kyoku[i])):
        for l in range(len(call_text_kyoku[i][k])):
            fo_call.write(str(call_text_kyoku[i][k][l][0])+'_'#音階
                          +str(call_text_kyoku[i][k][l][1])+'_'#長さ
                          +str(call_text_kyoku[i][k][l][4])+'_'#コードのルート音
                          +str(call_text_kyoku[i][k][l][5])+' '#コードの構成音
                          )  
        for n in range(len(resp_text_kyoku[i][k])):
            fo_resp.write(str(resp_text_kyoku[i][k][n][0])+'_'#音階
                          +str(resp_text_kyoku[i][k][n][1])+'_'#長さ
                          +str(resp_text_kyoku[i][k][n][4])+'_'#コードのルート音
                          +str(resp_text_kyoku[i][k][n][5])+' '#コードの構成音
                          )
        fo_call.write('\n')
        fo_resp.write('\n')
fo_call.close()
fo_resp.close()
