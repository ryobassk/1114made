import torch
import torch.nn as nn


from layers.torch import PositionalEncoding
from layers.torch import MultiHeadAttention
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



class Transformer(nn.Module):
    def __init__(self,
                 depth_source,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(depth_source,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               maxlen=maxlen,
                               device=device)
        self.decoder = Decoder(depth_target,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               maxlen=maxlen,
                               device=device)
        self.out = nn.Linear(d_model, depth_target)
        nn.init.xavier_normal_(self.out.weight)

        self.maxlen = maxlen

    def forward(self, source, target=None, tree=None, index=None):
        mask_source = self.sequence_mask(source)

        hs = self.encoder(source, mask=mask_source)
        if target is not None:
            target = target[:, :-1]
            len_target_sequences = target.size(1)
            mask_target = self.sequence_mask(target).unsqueeze(1)
            subsequent_mask = self.subsequence_mask(target)
            mask_target = torch.gt(mask_target + subsequent_mask, 0)

            y = self.decoder(target, hs,
                             mask=mask_target,
                             mask_source=mask_source)
            output = self.out(y)
        else:
            if tree is  None:
                batch_size = source.size(0)
                len_target_sequences = self.maxlen

                output = torch.ones((batch_size, 1),
                                    dtype=torch.long,
                                    device=self.device)

                for t in range(len_target_sequences - 1):
                    mask_target = self.subsequence_mask(output)
                    out = self.decoder(output, hs,
                                       mask=mask_target,
                                       mask_source=mask_source)
                    out = self.out(out)[:, -1:, :]
                    out = out.max(-1)[1]
                    output = torch.cat((output, out), dim=1)

            else:
                #生成したテキスト
                create_sentence = []
                #生成したテキスト仮に入れる
                now_sentence = [1]
                #文の確率スコア
                now_score = []
                #各文章の最大スコア
                max_score = 0
                
                chordkouho = [[[39, 1], [39, 1], [46, 0], [46, 0]],
                              [[36, 3],[39, 3], [46, 0], [46, 0]],
                              [[46, 0], [46, 0], [46, 0], [46, 0]]]                

                chordset = chordkouho[index%3]
                self.Treesearch(create_sentence, 
                                now_sentence, 
                                now_score, 
                                chordset, 
                                max_score, 
                                hs,
                                mask_source)
                #print(create_sentence)
                create_sentence = sorted(create_sentence, key=lambda x: x[0])[::-1]
                #print(create_sentence)
                score_result, text_result, tick_result = create_sentence[0]
                for i in range(len(create_sentence)):
                    score_result, text_result, tick_result = create_sentence[i]
                #print(text_result)
                return text_result

        return output

    def sequence_mask(self, x):
        return x.eq(0)

    def subsequence_mask(self, x):
        shape = (x.size(1), x.size(1))
        mask = torch.triu(torch.ones(shape, dtype=torch.uint8),
                          diagonal=1)
        return mask.unsqueeze(0).repeat(x.size(0), 1, 1).to(self.device)
    
    def Treesearch(self,create_sentence, now_sentence, 
                   now_score, chordset, max_score, hs, mask_source):
        #print(now_sentence)
        if len(create_sentence) >= 2:
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
            data = data
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
        dec_input = torch.tensor([now_sentence],
                                 dtype=torch.long,
                                 device=device)
        mask_target = self.subsequence_mask(dec_input)
        dec_output = self.decoder(dec_input, hs,
                                  mask=mask_target,
                                  mask_source=mask_source)
        dec_output = self.out(dec_output)[:, -1:, :]
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
        if len(select_oto)<2:
            samples = np.random.choice(select_oto, len(select_oto),  p=kouho_oto, replace=False)
        else:
            samples = np.random.choice(select_oto, 2,  p=kouho_oto, replace=False)
        index=0
        for p in samples:
            # 現在生成中の文に一文字追加する
            now_sentence.append(p)
            # 現在生成中の文のスコアに一つ追加する
            score_info = dec_output[0][0][p].item()
            now_score.append(score_info)
            # 再帰呼び出し
            self.Treesearch(create_sentence, now_sentence, 
                       now_score, chordset, max_score, hs, mask_source)
            # 現在生成中の文を一つ戻す          
            now_sentence.pop()
            # 現在生成中の文のスコアを一つ戻す        
            now_score.pop()
            index+=1

        

class Encoder(nn.Module):
    def __init__(self,
                 depth_source,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(depth_source,
                                      d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         maxlen=maxlen,
                         device=device) for _ in range(N)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        y = self.pe(x)
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y, mask=mask)

        return y


class EncoderLayer(nn.Module):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        h = self.attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)
        y = self.ff(h)
        y = self.dropout2(y)
        y = self.norm2(h + y)

        return y


class Decoder(nn.Module):
    def __init__(self,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(depth_target,
                                      d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         maxlen=maxlen,
                         device=device) for _ in range(N)
        ])

    def forward(self, x, hs,
                mask=None,
                mask_source=None):
        x = self.embedding(x)
        y = self.pe(x)
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, hs,
                              mask=mask,
                              mask_source=mask_source)

        return y


class DecoderLayer(nn.Module):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.src_tgt_attn = MultiHeadAttention(h, d_model)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff)
        self.dropout3 = nn.Dropout(p_dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, hs,
                mask=None,
                mask_source=None):
        h = self.self_attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)
        z = self.src_tgt_attn(h, hs, hs,
                              mask=mask_source)
        z = self.dropout2(z)
        z = self.norm2(h + z)

        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)

        return y


class FFN(nn.Module):
    def __init__(self, d_model, d_ff,
                 device='cpu'):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        h = self.l1(x)
        h = self.a1(h)
        y = self.l2(h)
        return y


if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1. データの準備
    '''
    en_one_path = './data/dataset/10_5/10_5callnum1.txt'
    
    oto_dict_path = './result/1114dict_id_10_5callresp.txt'
    chord_dict_path = './data/dataset/10_5/10_5dict_chord.txt'
    
    oto_vocab = Vocab()
    chord_vocab = Chord_vocab()

    oto_vocab.fit(oto_dict_path)
    chord_vocab.fit(chord_dict_path)

    x_test = oto_vocab.transform(en_one_path)
    test_dataloader = DataLoader((x_test, x_test),
                                  batch_size=1,
                                  batch_first=True,
                                  device=device)

    '''
    2. モデルの構築
    '''
    #辞書の長さ
    depth_x = len(oto_vocab.i2w)
    depth_t = len(oto_vocab.i2w)

    #モデルの設定
    model = Transformer(depth_x,
                        depth_t,
                        N=3,
                        h=4,
                        d_model=128,
                        d_ff=256,
                        maxlen=65,
                        device=device).to(device)

    path_model =('./result/transformer/transformer10_5_1114model_1')
    model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))

    day_str = '10_5'
    kazu = '2'


    '''
    3. モデルの学習・評価
    '''

    def test_step(x, index):
        model.eval()
        preds = model(x, tree=True, index=index)
        return preds


    #出力
    fp_music = ('./result/transformer/transformer'
                +str(now.month)+'_'+str(now.day)+'musicmake'
                +str(day_str)+'_'+str(kazu)+'.txt')
    fo_music = open(fp_music, "w")
    for x, _ in test_dataloader:
        outmusic = [x.reshape(-1).tolist()]

        for index in range(0,2):
            print("start")
            out = test_step(x, index)
            out = out.tolist()
            out.pop(0)
            outmusic.append(out)
            x_yobi = DataLoader(([out], [out]),
                                batch_size=1,
                                batch_first=True,
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
    foutmidi = ('./result/transformer/transformer'
                +str(now.month)+'_'+str(now.day)+'musicmake'
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

print('starttime',now)
print('endtime',datetime.datetime.now())
