'''
6.3.5.2 Multi-Head Attention - PyTorch
'''

import torch
import torch.nn as nn
from .ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 h,
                 d_model,
                 device='cpu'):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k = d_model // h
        self.d_v = d_v = d_model // h
        self.device = device

        self.W_q = nn.Parameter(torch.Tensor(h,
                                             d_model,
                                             d_k))

        self.W_k = nn.Parameter(torch.Tensor(h,
                                             d_model,
                                             d_k))

        self.W_v = nn.Parameter(torch.Tensor(h,
                                             d_model,
                                             d_v))

        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        nn.init.xavier_normal_(self.W_v)

        self.attn = ScaledDotProductAttention(d_k)
        self.linear = nn.Linear((h * d_v), d_model)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, q, k, v, mask=None):
        '''
        # Argument
            q, k, v: (batch, sequence, out_features)
            mask:    (batch, sequence)
        '''
        batch_size = q.size(0)#100 qはtorch.Size([100, 48, 128])
        #print(self.h) 4
        q = torch.einsum('hijk,hkl->hijl',
                         (q.unsqueeze(0).repeat(self.h, 1, 1, 1),
                          self.W_q))#torch.Size([4, 100, 48, 32])
        k = torch.einsum('hijk,hkl->hijl',
                         (k.unsqueeze(0).repeat(self.h, 1, 1, 1),
                          self.W_k))#torch.Size([4, 100, 48, 32])
        v = torch.einsum('hijk,hkl->hijl',
                         (v.unsqueeze(0).repeat(self.h, 1, 1, 1),
                          self.W_v))#torch.Size([4, 100, 48, 32])

        q = q.view(-1, q.size(-2), q.size(-1))#view torch.Size([400, 48, 32])
        k = k.view(-1, k.size(-2), k.size(-1))#view torch.Size([400, 48, 32])
        v = v.view(-1, v.size(-2), v.size(-1))#view torch.Size([400, 48, 32])
        #print(mask.size())#torch.Size([100, 48])
        if mask is not None:
            multiples = [self.h] + [1] * (len(mask.size()) - 1)
            #print(multiples)#[4, 1]
            mask = mask.repeat(multiples)
            #print(mask.size())#torch.Size([400, 48])
        c = self.attn(q, k, v, mask=mask)
        #print(c)
        #print(c.size())#torch.Size([400, 48, 32])
        c = torch.split(c, batch_size, dim=0)#配列を分けるここでは[0]のところを100個に
        #print(c[0].size())#torch.Size([100, 48, 32])
        #print(len(c))#4
        c = torch.cat(c, dim=-1)
        #print(c.size())#torch.Size([100, 48, 128])<- 128 = 32 *4
        out = self.linear(c)
        #print(out.size())#torch.Size([100, 48, 128])
        return out
