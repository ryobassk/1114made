import torch
import torch.nn as nn
import torch.optim as optimizers


from layers.torch import PositionalEncoding
from layers.torch import MultiHeadAttention
from utils import Vocab
from utils.torch import DataLoader


import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()

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

    def forward(self, source, target=None, tree=None):
        #パッティング文字はTrueとなる
        mask_source = self.sequence_mask(source)
        
        hs = self.encoder(source, mask=mask_source)
        
        if target is not None:
            print("train")
            target = target[:, :-1]#正解ラベル
            len_target_sequences = target.size(1)#正解ラベルの長さ
            mask_target = self.sequence_mask(target).unsqueeze(1)#100,1,59 正解ラベルのパッティング文字がtrueに
            subsequent_mask = self.subsequence_mask(target)#100,59,59　59×59の上三角形行列が100個
            mask_target = torch.gt(mask_target + subsequent_mask, 0)#torch.Size([100, 59, 59])足してなところがFalseとなる

            y = self.decoder(target, hs,
                             mask=mask_target,
                             mask_source=mask_source)
            print(y.size())#torch.Size([100, 59, 128])
            output = self.out(y)
            #self.out = nn.Linear(d_model=128, depth_target=1289)
            #print(output.size())#torch.Size([100, 59, 1289])
            
        else:
            if tree is  None:
                print("test")
                batch_size = source.size(0)
                len_target_sequences = self.maxlen
                #print("len_target_sequences",len_target_sequences)#65
                output = torch.ones((batch_size, 1),
                                    dtype=torch.long,
                                    device=self.device)
                #print("output_size:",output.size())#torch.Size([1, 1])
                #print(output)#tensor([[1]])
                for t in range(len_target_sequences - 1):
                    mask_target = self.subsequence_mask(output)
                    #print('mask_target_size:',mask_target.size())#torch.Size([1, 1, 1]),torch.Size([1, 2, 2]),・・・torch.Size([1, 64, 64])
                    #print(mask_target)
                    out = self.decoder(output, hs,
                                       mask=mask_target,
                                       mask_source=mask_source)
                    #print("out1",out.size())#mask_target_size: torch.Size([1, 62, 62])のときout1 torch.Size([1, 62, 128])
                    out = self.out(out)[:, -1:, :]
                    #print("out2",out.size())#mask_target_size: torch.Size([1, 62, 62])のときout2 torch.Size([1, 1, 1289])
                    out = out.max(-1)[1]
                    #print("out3:",out.size())#torch.Size([1, 1])
                    #print(out)#tensor([[934]])次の単語
                    #print(output)#tensor([[  1, 934]])
                    output = torch.cat((output, out), dim=1)
                    #print("output_size:",output.size(),"\n")#torch.Size([1, 1])
                    #print(output)#tensor([[  1, 934]])
                    
            else:
                print("test_tree探索")
                batch_size = source.size(0)
                len_target_sequences = self.maxlen
                #print("len_target_sequences",len_target_sequences)#65
                output = torch.ones((batch_size, 1),
                                    dtype=torch.long,
                                    device=self.device)
                #print("output_size:",output.size())#torch.Size([1, 1])
                #print(output)#tensor([[1]])
                for t in range(len_target_sequences - 1):
                    mask_target = self.subsequence_mask(output)
                    #print('mask_target_size:',mask_target.size())#torch.Size([1, 1, 1]),torch.Size([1, 2, 2]),・・・torch.Size([1, 64, 64])
                    #print(mask_target)
                    
                    out = self.decoder(output, hs,
                                       mask=mask_target,
                                       mask_source=mask_source)
                    #print("out1",out.size())#mask_target_size: torch.Size([1, 62, 62])のときout1 torch.Size([1, 62, 128])
                    out = self.out(out)[:, -1:, :]
                    #print("out2",out.size())#mask_target_size: torch.Size([1, 62, 62])のときout2 torch.Size([1, 1, 1289])
                    out = out.max(-1)[1]
                    #print("out3:",out.size())#torch.Size([1, 1])
                    #print(out)#tensor([[934]])次の単語
                    
                    output = torch.cat((output, out), dim=1)
                    #print("output_size:",output.size(),"\n")#torch.Size([1, 1])
                    #print(output)#tensor([[  1, 934]])
             
        return output

    def sequence_mask(self, x):
        return x.eq(0)

    def subsequence_mask(self, x):
        shape = (x.size(1), x.size(1))        
        mask = torch.triu(torch.ones(shape, dtype=torch.uint8),
                          diagonal=1)#59*59
        '''
        triuに4×4を与えると
        [[0,1,1,1],
         [0,0,1,1],
         [0,0,0,1],
         [0,0,0,0]]が得られる
        '''
        return mask.unsqueeze(0).repeat(x.size(0), 1, 1).to(self.device)


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
        #print(x.size())#torch.Size([100, 48])
        x = self.embedding(x)#torch.Size([100, 48, 128])
        #print(x[0][0][100:-1])
        y = self.pe(x)#torch.Size([100, 48, 128])#位置情報を載せる
        #print(y[0][0][100:-1])
        #print(self.encoder_layers)
        """
        (0): EncoderLayer(
    (attn): MultiHeadAttention(
      (attn): ScaledDotProductAttention()
      (linear): Linear(in_features=128, out_features=128, bias=True)
    )
    (dropout1): Dropout(p=0.1, inplace=False)
    (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (ff): FFN(
      (l1): Linear(in_features=128, out_features=256, bias=True)
      (a1): ReLU()
      (l2): Linear(in_features=256, out_features=128, bias=True)
    )
    (dropout2): Dropout(p=0.1, inplace=False)
    (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
        """
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y, mask=mask)#エンコーダーの層の数回る（Nの値） 
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
        #print(x.size())#torch.Size([100, 48, 128])
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
        #print(y.size())#torch.Size([100, 59, 128])
        
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
        
        ###Decoderのみ############################
        #q:現在のラベル, k：入力のラベル, v：入力の応答
        #print(h.size(), hs.size(), mask_source.size())
        z = self.src_tgt_attn(h, hs, hs,
                              mask=mask_source)
        z = self.dropout2(z)
        z = self.norm2(h + z)
        ###############################
        
        
        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)
        #print(y.size())#torch.Size([100, 59, 128])

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
    #単語変換関数
    dict_vocab = Vocab()
    
    #学習データのパス
    dict_path = './data/dataset/10_2/10_2callresp.txt'    
    en_train_path = './data/dataset/10_2/10_2call.txt'
    de_train_path = './data/dataset/10_2/10_2resp.txt'


    #データからID辞書を作成
    dict_vocab.fit(dict_path)
    
    #データをIDに変換
    x_data = dict_vocab.transform(en_train_path)
    t_data = dict_vocab.transform(de_train_path, eos=True)



    #テストデータと訓練データに分ける
    x_train, x_val, t_train, t_val = train_test_split(x_data, t_data, test_size=0.1, shuffle=True)
    x_test, t_test = x_val, t_val
    
    #データをバッチ化する（tensor）
    batch_size = 100
    train_dataloader = DataLoader((x_train, t_train),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  batch_first=True,
                                  device=device)
    val_dataloader = DataLoader((x_val, t_val),
                                batch_size=batch_size,
                                batch_first=True,
                                device=device)
    test_dataloader = DataLoader((x_test, t_test),
                                 batch_size=1,
                                 batch_first=True,
                                 device=device)

    '''
    2. モデルの構築
    '''
    #辞書の長さ
    depth_x = len(dict_vocab.i2w)
    depth_t = len(dict_vocab.i2w)
    
    #モデルの設定
    model = Transformer(depth_x,
                        depth_t,
                        N=3,
                        h=4,
                        d_model=128,
                        d_ff=256,
                        maxlen=65,
                        device=device).to(device)

    '''
    3. モデルの学習・評価
    '''
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x, t):
        model.train()
        preds = model(x, t)
        loss = compute_loss(t[:, 1:].contiguous().view(-1),
                            preds.contiguous().view(-1, preds.size(-1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def val_step(x, t):
        model.eval()
        preds = model(x, t)
        loss = compute_loss(t[:, 1:].contiguous().view(-1),
                            preds.contiguous().view(-1, preds.size(-1)))

        return loss, preds

    def test_step(x):
        model.eval()
        preds = model(x)
        return preds

    epochs = 700
    train_allloss=[]
    val_allloss=[]
    for epoch in range(epochs):
        for idx, (x, t) in enumerate(test_dataloader):
            preds = test_step(x)

            source = x.view(-1).tolist()
            target = t.view(-1).tolist()
            out = preds.view(-1).tolist()

            source = ' '.join(dict_vocab.decode(source))
            target = ' '.join(dict_vocab.decode(target))
            out = ' '.join(dict_vocab.decode(out))

            print('>', source)
            print('=', target)
            print('<', out)
            print()

            if idx >= 0:
                break

#result
plt.plot(train_allloss)
plt.savefig('./Transformer/'+str(now.month)+str(now.day)+'Transformer_trainloss.png')
plt.plot(val_allloss)
plt.savefig('./Transformer/'+str(now.month)+str(now.day)+'Transformer_valloss.png')

trainloss_txt = str(train_allloss)
valloss_txt = str(val_allloss)
with open('./'+str(now.month)+str(now.day)+'loss.txt', mode='w') as f:
    f.write('trainloss\n')
    f.write(trainloss_txt )
    f.write('\nvalloss\n')
    f.write(valloss_txt )

print('starttime',now)
print('endtime',datetime.datetime.now())