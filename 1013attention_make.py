import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from layers.torch import Attention
from utils.torch import DataLoader

import numpy as np
import datetime
now = datetime.datetime.now()


class Vocab:
    #イニシャライザ
    def __init__(self):
        self.w2i = {} #単語⇒IDの辞書
        self.i2w = {} #ID⇒単語の辞書

    #ファイルの読み込みと辞書の作成
    def fit(self, path):        
        #文章を保存(4	70_480_58_0)
        with open(path, 'r', encoding="utf-8") as f:
            sentences = f.read().splitlines()         
        #ID⇒単語の辞書の作成
        for sentence in sentences:
            key, value = sentence.split('\t')
            self.i2w[int(key)] =  value
            
        #単語⇒IDの辞書の作成  
        self.w2i = {i: w for w, i in self.i2w.items()}

    
    #ファイルの読み込み、全体をIDに変換
    def transform(self, path, bos=False, eos=False):
        output = []  
        #文章を保存（'彼 は 走 る の が とても 早 い 。', ）
        with open(path, 'r', encoding="utf-8") as f:
            sentences = f.read().splitlines()
        for sentence in sentences:
            sentence = sentence.split()
            if bos:
                sentence = [self.bos_char] + sentence
            if eos:
                sentence = sentence + [self.eos_char]
            output.append(self.encode(sentence))
        return output
        
    #単語をIDに変換
    def encode(self, sentence):
        output = []       
        #1文章を入力,wは単語
        for w in sentence:
            #辞書にない単語は未知語としてIDを振る
            if w not in self.w2i:
                print(self.w2i)
                idx = self.w2i['<unk>']
            #単語を辞書を用いてIDに変換
            else:
                idx = self.w2i[w]
            output.append(idx)       
        return output
    
    #IDを単語に変換
    def decode(self, sentence):
        return [self.i2w[id] for id in sentence]


class Chord_vocab:
    #イニシャライザ
    def __init__(self):
        self.list = [] #ID⇒コードの辞書

    #ファイルの読み込みと辞書の作成
    def fit(self, path):        
        #文章を保存(4	70_480_58_0)
        with open(path, 'r', encoding="utf-8") as f:
            sentences = f.read().splitlines() 

        #ID⇒単語の配列
        for sentence in sentences:
            key, value = sentence.split('\t')
            value = value.split('_')
            value_list = [int(a) for a in value if a != '']
            self.list.append(value_list)  
        
    #単語をIDに変換
    def encode(self, id_num):
        output = self.list[id_num]
        return output
    
    #IDを単語に変換
    def decode(self, list_chord):
        output = self.list.index(list_chord)
        return output  

#エンコーダー
class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_layers=1,
                 dropout=0,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        len_source_sequences = (x.t() > 0).sum(dim=-1)
        x = self.embedding(x)
        x = pack_padded_sequence(x, len_source_sequences)
        h, states = self.lstm(x)
        h, _ = pad_packed_sequence(h)

        return h, states

#デコーダー
class Decoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 num_layers=1,
                 dropout=0,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.attn = Attention(hidden_dim, hidden_dim, device=self.device)
        self.out = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x, hs, states, source=None):
        x = self.embedding(x)
        ht, states = self.lstm(x, states)
        ht = self.attn(ht, hs, source=source)
        y = self.out(ht)

        return y, states


if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    en_one_path = 'C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/data/dataset/10_5/10_5callnum1.txt'
    
    oto_dict_path = 'C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/data/dataset/10_5/10_5dict_id_10_5callresp.txt'
    chord_dict_path = 'C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/data/dataset/10_5/10_5dict_chord.txt'    
    
    oto_vocab = Vocab()
    chord_vocab = Chord_vocab()
    
    oto_vocab.fit(oto_dict_path)
    chord_vocab.fit(chord_dict_path)
    
    x_test = oto_vocab.transform(en_one_path)
    test_dataloader = DataLoader((x_test, x_test),
                                  batch_size=1,
                                  batch_first=False,
                                  device=device)
    

    '''
    2. モデルの構築
    '''
    #辞書の長さ
    depth_x = len(oto_vocab.i2w)
    depth_t = len(oto_vocab.i2w)
    
    input_dim = depth_x #入力層
    hidden_dim = 128 #中間層
    output_dim = depth_t #出力層
    maxlen = 65 #入力データの長さ
    #モデルの設定
    enc = Encoder(input_dim,
                  hidden_dim,
                  device=device,
                  num_layers=2).to(device)

    dec = Decoder(hidden_dim,
                  output_dim,
                  device=device,
                  num_layers=2).to(device)
    
    #学習時のモデルの読み込み
    
    path_dec = ('C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/data/sakusei/learning/'
                'Attention/1013Atttention_decoder_300')  
    path_enc = ('C:/Users/Ryo Ogasawara/OneDrive/lab/0924made/data/sakusei/learning/'
                'Attention/1013Atttention_encoder_300')
    enc.load_state_dict(torch.load(path_enc, map_location=torch.device('cpu')))
    dec.load_state_dict(torch.load(path_dec, map_location=torch.device('cpu')))
    
    day_str = '10_5'
    kazu = '2'


    '''
    3. モデルの出力
    '''   
    # 再帰関数の木探索
    # 再帰関数の木探索
    def Treesearch(create_sentence, now_sentence, 
                   now_score, states_h, chordset, max_score, hs):
        if len(create_sentence) >= 20:
            return
        
        # 現在の単語を取得する
        now_word = now_sentence[-1]
        # 文のスコア
        score = np.prod(now_score)
        # 現在の文の長さ
        treedeeply = len(now_sentence)
        sum_tick = 0
        select_oto = []
        kouho_oto = []
        

        # パッティング文字(ID:1) or 終端文字/-300_0_0(ID:23)が出た場合に一つ前の文字に戻る
        if (now_word == oto_vocab.encode(['</s>'])[0] 
            or now_word == oto_vocab.encode(['<pad>'])[0] 
            or now_word == oto_vocab.encode(['<unk>'])[0] ):
            now_sentence.pop()
            states_h.pop()
            now_score.pop()
            now_word = now_sentence[-1]
        #生成した文字から拍数を抜き出して今何TICK目かをsum_tickに入れる
        for i in range(len(now_sentence)):
            if (now_sentence[i] == oto_vocab.encode(['<s>'])[0]):
                sum_tick += 0
            else:
                _, now_tick, _, _  = oto_vocab.decode([now_sentence[i]])[0].split('_')
                sum_tick += int(now_tick)
                
           
        if sum_tick < 1920:
            chord_tick =  1920      
            chord_info = chordset[0]
        elif sum_tick < 3840:
            chord_tick =  3840        
            chord_info = chordset[1]
        elif sum_tick < 5760:  
            chord_tick =  5760
            chord_info = chordset[2]
        else:
            chord_tick =  7680
            chord_info = chordset[3]
            
        # 枝刈り - 単語数が5以上で最大スコアの6割以下なら、終わる
        if treedeeply > 5 and max_score * 0.6 > score:
            return
        
        
        
        # 最大の文の長さ以上なら、文を追加して終わる
        #４小節になったら確率とTick数とテキストをcreate_sentenceに格納
        if sum_tick == 7680: #7680
            # 文を追加
            data = np.array(now_sentence)
            data = data.tolist()
            create_sentence.append([score, data, sum_tick])
            seen = []
            create_sentence = [x for x in create_sentence if x not in seen and not seen.append(x)]
            # 最大スコアを更新
            if max_score < score:
                max_score = score
            print(len(create_sentence))
            return
         
        #拍数をこえたものは終了
        if sum_tick > 7680: #7680  
            return

        #デコーダに入れる最初の値/開始記号[[1]]をつくる
        dec_input = torch.tensor([[now_word]],
                                 dtype=torch.long,
                                 device=device)
        dec_output, dec_states = dec(dec_input, hs,states_h[-1])
        
        #確立が高い順に並べる/index_outmaxは確立が高いもののインデックスが格納される
        #[11,22,55,33,1]なら[2,3,1,0,4]が格納される
        index_outmax = torch.argsort(-dec_output[0][0])
        

        len_if = chord_tick - sum_tick
        for i in range(len(index_outmax)):
            if (index_outmax[i].item() != oto_vocab.encode(['</s>'])[0] 
                and index_outmax[i].item() != oto_vocab.encode(['<s>'])[0] 
                and index_outmax[i].item() != oto_vocab.encode(['<pad>'])[0] 
                and index_outmax[i].item() != oto_vocab.encode(['<unk>'])[0] ):
                
                _, now_len, now_chord_oto, now_kousei  = oto_vocab.decode([index_outmax[i].item()])[0].split('_')
                now_chord = [int(now_chord_oto), int(now_kousei)]
                
                if int(now_len) <= len_if and now_chord == chord_info:
                    select_oto.append(index_outmax[i].item())
                           
        if len(select_oto)==0:
            return
        #音の確率
        for k in range(len(select_oto)):
            kouho_oto.append(dec_output[0][0][select_oto[k]].item()) 
        #音の確率を正規化
        kouho_oto = torch.tensor(kouho_oto)
        Softmax = nn.Softmax()
        kouho_oto = np.array(Softmax(kouho_oto))
        #音の選択
        if len(select_oto)<3:
            samples = np.random.choice(select_oto, len(select_oto),  p=kouho_oto, replace=False)
        else:
            samples = np.random.choice(select_oto, 3,  p=kouho_oto, replace=False)
        index=0
        for p in samples:
            # 現在生成中の文に一文字追加する
            now_sentence.append(p)
            # 現在生成中の文のスコアに一つ追加する
            score_info = dec_output[0][0][p].item()
            now_score.append(score_info)
            states_h.append(dec_states)
            # 再帰呼び出し
            Treesearch(create_sentence, now_sentence, 
                       now_score, states_h, chordset, max_score, hs)
            # 現在生成中の文を一つ戻す          
            now_sentence.pop()
            # 現在生成中の文のスコアを一つ戻す        
            now_score.pop()
            states_h.pop()
            index+=1

        
    
    #テストデータの予測
    def test_step(xdata, index):
        enc.eval(), dec.eval()        
        #生成したテキスト
        create_sentence = []
        #生成したテキスト仮に入れる
        now_sentence = [1]
        #文の確率スコア
        now_score = []
        #各文章の最大スコア
        max_score = 0
        hs, states = enc(xdata)
        
        states_h = [states]
        chordkouho = [[[39, 1], [39, 1], [46, 0], [46, 0]],
                      [[36, 3],[39, 3], [46, 0], [46, 0]],
                      [[46, 0], [46, 0], [46, 0], [46, 0]]]
        
        """
        chordkouho = [[[51, 0], [51, 0], [46, 0], [46, 0]],
                      [[48, 1],[48, 1], [46, 0], [46, 0]],
                      [[46, 0], [46, 0], [46, 0], [46, 0]]]
        
        chordkouho = [[[39, 1], [39, 1], [46, 0], [46, 0]],
                      [[36, 3],[39, 3], [46, 0], [46, 0]],
                      [[46, 0], [46, 0], [46, 0], [46, 0]]]
        
        """
        chordset = chordkouho[index%3]
        Treesearch(create_sentence, 
                   now_sentence, 
                   now_score, 
                   states_h, 
                   chordset,
                   max_score,
                   hs)
        create_sentence = sorted(create_sentence, key=lambda x: x[0])[::-1]
        if len(create_sentence) == 0:
            test_step(xdata, index)
        score_result, text_result, tick_result = create_sentence[0]
        for i in range(len(create_sentence)):
            score_result, text_result, tick_result = create_sentence[i]
        return text_result
    
    
    #出力
    fp_music = ('./data/sakusei/make_music/'
                +str(now.month)+'_'+str(now.day)+'Amusicmake'
                +str(day_str)+'_'+str(kazu)+'.txt')
    fo_music = open(fp_music, "w")
    for x, _ in test_dataloader:
        outmusic = [x.reshape(-1).tolist()]
        for index in range(0,2):
            print("start")
            out = test_step(x, index)
            out.pop(0)
            outmusic.append(out)
            x_yobi = DataLoader(([out], [out]),
                                batch_size=1,
                                batch_first=False,
                                device=device)
            for index2 in x_yobi:
                x = index2[0]
        
        
    for index in range(len(outmusic)):
        outprint = ' '.join(oto_vocab.decode(outmusic[index]))
        print(index, '>\n', outprint, '\n')
        fo_music.write(outprint+'\n')
    fo_music.close()

    '''
    4. MIDIの出力
    '''
    import pretty_midi 
    outmusic_list = []
    for i in range(len(outmusic)):
        out = oto_vocab.decode(outmusic[i])
        out3=[]
        for k in range(len(out)):
            out2 = out[k].split('_')
            out2 = [int(n) for n in out2]
            out3.append(out2)
        outmusic_list.append(out3)
        
    #書き出すファイル
    foutmidi = ('./data/sakusei/make_music/'
                +str(now.month)+'_'+str(now.day)+'Amusicmake'
                +str(day_str)+'_'+str(kazu)+'.mid')
    
    #midiの作成、４分音符を480tick、BPM=160と設定
    musicdata = pretty_midi.PrettyMIDI(resolution=480, 
                                       initial_tempo=160)
    #楽器を作成/0はピアノ　ここではメロディのsaxと伴奏のbackを作成
    instrument_sax = pretty_midi.Instrument(56) 
    instrument_back = pretty_midi.Instrument(26)
    #音量
    onryo_sax = 80
    onryo_back = 30
    
    preonkai=outmusic_list[0][0][2]
    prekousei = outmusic_list[0][0][3]
    chord_info=[]
    nowtick = 0
    for i in range(len(outmusic_list)):
        for k in range(len(outmusic_list[i])):
            onkai = outmusic_list[i][k][2]
            kousei = outmusic_list[i][k][3]
            starttime = musicdata.tick_to_time(nowtick)
            endtime = musicdata.tick_to_time(nowtick 
                                             + outmusic_list[i][k][1])
            if ((onkai != preonkai) 
                or (kousei != prekousei) 
                or (nowtick % 1920 ==0 and nowtick!=0 )): 
                chord_info.append([preonkai, 
                                   prekousei, 
                                   nowtick ]) 
                
            nowtick += outmusic_list[i][k][1]
            preonkai = onkai
            prekousei = kousei
    chord_info.append([onkai, kousei, nowtick ])
    nowtick = 0
    for i in range(len(chord_info)):
        onkai = chord_info[i][0]
        kousei = chord_vocab.encode(chord_info[i][1])
        onkai_list=[]
        for k in kousei:
            onkai_list.append(onkai+k)
        starttime = musicdata.tick_to_time(nowtick)
        endtime = musicdata.tick_to_time(chord_info[i][2])
        if onkai != -1:
            for l in onkai_list:               
                note = pretty_midi.Note(velocity=onryo_sax, 
                                        pitch=l, 
                                        start=starttime, 
                                        end=endtime)
                instrument_back.notes.append(note)
        nowtick = chord_info[i][2]

    #メロディライン情報の読み込みと情報を入れ込む
    nowtick = 0
    for i in range(len(outmusic_list)):
        for k in range(len(outmusic_list[i])):
            onkai = outmusic_list[i][k][0]
            starttime = musicdata.tick_to_time(nowtick)
            endtime = musicdata.tick_to_time(nowtick 
                                             + outmusic_list[i][k][1])
            if onkai != -1: 
                note = pretty_midi.Note(velocity=onryo_sax, 
                                        pitch=onkai, 
                                        start=starttime, 
                                        end=endtime)
                instrument_sax.notes.append(note)
            nowtick += outmusic_list[i][k][1]
            
    #各楽器の情報をMIDIに格納
    musicdata.instruments.append(instrument_sax)
    musicdata.instruments.append(instrument_back)
    musicdata.write(foutmidi)
    


"""
#楽器と数字の対応表
#1-8	Piano
#9-16	Chromatic Percussion
#17-24	Organ
#25-32	Guitar
#33-40	Bass
#41-48	Strings
#49-56	Ensemble
#57-64	Brass
#65-72	Reed
#73-80	Pipe
#81-88	Synth Lead
#89-96	Synth Pad
#97-104	Synth Effects
#105-112	Ethnic
#113-120	Percussive
#121-128	Sound Effects

#1-8	ピアノ
#9-16	クロマチックパーカッション
#17〜24	器官
#25〜32	ギター
#33〜40	ベース
#41〜48	文字列
#49〜56	アンサンブル
#57〜64	真鍮
#65〜72	リード
#73-80	パイプ
#81〜88	シンセリード
#89〜96	シンセパッド
#97-104	シンセ効果
#105〜112	エスニック
#113-120	パーカッシブ
#121〜128	音響効果
#1。	アコースティックグランドピアノ
#2。	明るいアコースティックピアノ
#3。	エレクトリックグランドピアノ
#4。	ホンキートンクピアノ
#5。	エレクトリックピアノ1
#6。	エレクトリックピアノ2
#7。	チェンバロ
#8。	クラヴィ
#9。	チェレスタ
#10。	グロッケンシュピール
#11。	オルゴール
#12。	ビブラフォン
#13。	マリンバ
#14。	木琴
#15。	管状の鐘
#16。	ダルシマー
#17。	ドローバーオルガン
#18。	打楽器オルガン
#19。	ロックオルガン
#20。	教会オルガン
#21。	リードオルガン
#22。	アコーディオン
#23。	ハーモニカ
#24。	タンゴアコーディオン
#25。	アコースティックギター（ナイロン）
#26。	アコースティックギター（スチール）
#27。	エレキギター（ジャズ）
#28。	エレクトリックギター（クリーン）
#29。	エレキギター（ミュート）
#30。	オーバードライブギター
#31。	ディストーションギター
#32。	ギターハーモニクス
#33。	アコースティックベース
#34。	エレクトリックベース（指）
#35。	エレクトリックベース（ピック）
#36。	フレットレスベース
#37。	スラップベース1
#38。	スラップベース2
#39。	シンセベース1
#40。	シンセベース2
#41。	バイオリン
#42。	ビオラ
#43。	チェロ
#44。	コントラバス
#57。	トランペット
#58。	トロンボーン
#59。	チューバ
#60。	ミュートトランペット
#61。	フレンチホルン
#62。	真鍮セクション
#63。	シンセブラス1
#64。	SynthBrass 2
#65。	ソプラノサックス
#66。	アルトサックス
#67。	テナーサックス
#68。	バリトンサックス
#69。	オーボエ
##71。	ファゴット
#72。	クラリネット
#73。	ピッコロ
#74。	フルート
#75。	レコーダー
#76。	パンフルート
#77。	ブローボトル
#78。	尺八
#79。	ホイッスル
#80。	オカリナ
"""