import random
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


from utils import Vocab
from utils.torch import DataLoader
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()

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
        self.out = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x, states):
        x = self.embedding(x)
        h, states = self.lstm(x, states)
        y = self.out(h)

        return y, states


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    1. データの準備
    '''
    #単語変換関数
    dict_vocab = Vocab()
    
    #学習データのパス
    dict_path = './data/dataset/0924callresp.txt'
    
    en_train_path = './data/dataset/0924call.txt'
    en_val_path = './data/dataset/0924call_num9.txt'
    en_test_path = './data/dataset/0924call_num9.txt'

    de_train_path = './data/dataset/0924resp.txt'
    de_val_path = './data/dataset/0924resp_num9.txt'
    de_test_path = './data/dataset/0924resp_num9.txt'

    #データからID辞書を作成
    dict_vocab.fit(dict_path)
    
    #データをIDに変換
    x_train = dict_vocab.transform(en_train_path)
    x_val = dict_vocab.transform(en_val_path)
    x_test = dict_vocab.transform(en_test_path)

    t_train = dict_vocab.transform(de_train_path, eos=True)
    t_val = dict_vocab.transform(de_val_path, eos=True)
    t_test = dict_vocab.transform(de_test_path, eos=True)


    #データをバッチ化する（tensor）
    batch_size = 100
    train_dataloader = DataLoader((x_train, t_train),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  batch_first=False,
                                  device=device)
    val_dataloader = DataLoader((x_val, t_val),
                                batch_size=batch_size,
                                batch_first=False,
                                device=device)
    test_dataloader = DataLoader((x_test, t_test),
                                 batch_size=1,
                                 batch_first=False,
                                 device=device)

    '''
    2. モデルの構築
    '''
    #辞書の長さ
    depth_x = len(dict_vocab.i2w)
    depth_t = len(dict_vocab.i2w)
    
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

    '''
    3. モデルの学習
    '''
    #損失関数　ignore_index=0　マスク処理
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    #勾配法
    enc_optimizer = optimizers.Adam(enc.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)
    dec_optimizer = optimizers.Adam(dec.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)
    
    #seq2seqモデル（確率を出力）
    def Model_enc_dec(source, target=None, use_teacher_forcing=False):
        batch_size = source.size(1)
        if target is not None:
            len_target_sequences = target.size(0)
        else:
            len_target_sequences = maxlen

        _, states = enc(source)

        y = torch.ones((1, batch_size),
                       dtype=torch.long,
                       device=device)
        output = torch.zeros((len_target_sequences,
                              batch_size,
                              output_dim),
                              device=device)

        for t in range(len_target_sequences):
            out, states = dec(y, states)
            output[t] = out
            if use_teacher_forcing and target is not None:
                y = target[t].unsqueeze(0)
            else:
                y = out.max(-1)[1]
        return output

        
    
    #正解と予測値から誤差を求める
    def compute_loss(label, pred):
        return criterion(pred, label)
    
    #モデルの誤差を求め、逆伝搬する
    def train_step(xdata, tdata,
                   teacher_forcing_rate=0.5):
        use_teacher_forcing = (random.random() < teacher_forcing_rate)
        
        enc.train(), dec.train()
        preds = Model_enc_dec(xdata, tdata,
                              use_teacher_forcing=use_teacher_forcing)
        loss = compute_loss(tdata.reshape(-1),
                            preds.reshape(-1, preds.size(-1)))
        
        enc_optimizer.zero_grad(), dec_optimizer.zero_grad()
        loss.backward()
        enc_optimizer.step(), dec_optimizer.step()
        
        return loss, preds

    #1epochあたりのモデルの評価（検証データで）
    def val_step(xdata, tdata):
        enc.eval(), dec.eval()
        
        preds = Model_enc_dec(xdata, tdata ,
                              use_teacher_forcing=False) 
        loss = compute_loss(tdata.reshape(-1),
                            preds.reshape(-1, preds.size(-1)))

        return loss, preds
    
    #テストデータの予測
    def test_step(xdata):
        enc.eval(), dec.eval()
        
        preds = Model_enc_dec(xdata,
                              use_teacher_forcing=False) 
        return preds

    epochs = 300#30
    train_allloss=[]
    val_allloss=[]
    for epoch in range(epochs):
        print('-' * 20)
        print('epoch: {}'.format(epoch+1))

        train_loss = 0.
        val_loss = 0.
        
        #学習
        for (x, t) in train_dataloader:
            loss, _ = train_step(x, t)
            train_loss += loss.item()
            
        
        #lossの計算（訓練データ）
        train_loss /= len(train_dataloader)

        #lossの計算（検証データ）
        for (x, t) in val_dataloader:
            loss, _ = val_step(x, t)
            val_loss += loss.item()

        val_loss /= len(val_dataloader)
        
        train_allloss.append(train_loss)
        val_allloss.append(val_loss)
        print('loss: {:.3f}, val_loss: {:.3}'.format(
            train_loss,
            val_loss
        ), '\n')
        
        #モデルのセーブ
        if (epoch+1) % 50 == 0:
            torch.save(enc.state_dict(), './data/sakusei/learning/en/0'+str(now.month)+str(now.day)
                       +'encoder_'+str(epoch+1))
            torch.save(dec.state_dict(), './data/sakusei/learning/de/0'+str(now.month)+str(now.day)
                       +'decoder_'+str(epoch+1))

        #テストデータでの検証
        for idx, (x, t) in enumerate(test_dataloader):
            preds = test_step(x)

            source = x.reshape(-1).tolist()
            target = t.reshape(-1).tolist()
            out = preds.max(dim=-1)[1].reshape(-1).tolist()

            source = ' '.join(dict_vocab.decode(source))
            target = ' '.join(dict_vocab.decode(target))
            out = ' '.join(dict_vocab.decode(out))

            print('>', source, '\n')
            print('=', target, '\n')
            print('<', out, '\n')
            print()

            if idx >= 0:
                break

#result
plt.plot(train_allloss)
plt.savefig('./data/sakusei/learning/0'+str(now.month)+str(now.day)+'trainloss.png')
plt.plot(val_allloss)
plt.savefig('./data/sakusei/learning/0'+str(now.month)+str(now.day)+'valloss.png')

trainloss_txt = str(train_allloss)
valloss_txt = str(val_allloss)
with open('./data/sakusei/learning/0'+str(now.month)+str(now.day)+'loss.txt', mode='w') as f:
    f.write('trainloss\n')
    f.write(trainloss_txt )
    f.write('\nvalloss\n')
    f.write(valloss_txt )