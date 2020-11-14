'''
6.3.5.1 Scaled Dot-Product Attention - PyTorch
'''

import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 d_k,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.scaler = np.sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        '''
        # Argument
            q, k, v: (batch, sequence, out_features)
            mask:    (batch, sequence)
        '''
        
        score = torch.einsum('ijk,ilk->ijl', (q, k)) / self.scaler#内積
        #print(score.size())#torch.Size([400, 48, 48])
        #print(torch.max(score, dim=-1, keepdim=True)[0].size())#torch.Size([400, 48, 1])
        score = score - torch.max(score, dim=-1, keepdim=True)[0]#オーバーフロー対策
        #print(score.size())#torch.Size([400, 48, 48])
        
        score = torch.exp(score)
        #print(score.size())#torch.Size([400, 48, 48])
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1).repeat(1, score.size(1), 1)
                #print(mask.size())#torch.Size([400, 48, 48])
            score.data.masked_fill_(mask, 0)#パッティング文字のところの値を0にする
        #print(score.size())#torch.Size([400, 48, 48]) 
        a = score / torch.sum(score, dim=-1, keepdim=True)#重さを求める
        c = torch.einsum('ijk,ikl->ijl', (a, v))#valueの値を重さをかけて出力これが特徴量として使われる

        return c
