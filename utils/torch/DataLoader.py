import numpy as np
from sklearn.utils import shuffle
import torch
from torch.nn.utils.rnn import pad_sequence
np.random.seed(123)
def sort(x, t):
        lens = [len(i) for i in x]        
        indices = sorted(range(len(lens)), key=lambda i: -lens[i])      
        x = [x[i] for i in indices]
        t = [t[i] for i in indices]
        return (x, t)

class DataLoader(object):
    def __init__(self, dataset,
                 batch_size=100,
                 shuffle=False,
                 batch_first=False,
                 device='cpu',
                 random_state=None):
        self.dataset = list(zip(dataset[0], dataset[1]))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_first = batch_first
        self.device = device
        if random_state is None:
            random_state = np.random.RandomState(123)
        self.random_state = random_state
        self._idx = 0
        self._reset()

    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()
        x, t = zip(*self.dataset[self._idx:(self._idx + self.batch_size)])
        (x, t) = sort(x, t)
        x = [torch.LongTensor(X) for X in x]
        t = [torch.LongTensor(T) for T in t]
        x = pad_sequence(x)
        t = pad_sequence(t)
        if self.batch_first:
            x = x.t()
            t = t.t()
        self._idx += self.batch_size     
        return x.to(self.device), t.to(self.device)

    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset,
                                   random_state=self.random_state)
        self._idx = 0
