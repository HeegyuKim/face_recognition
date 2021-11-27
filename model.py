import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd



# Generalizing Pooling
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)      
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class SimpleFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 128x128x3
            nn.Conv2d(3, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 64x64x12
            nn.Conv2d(12, 24, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32x32x24
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 16x16x48
            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 8x8x96
            GeM(p_trainable=True)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 96)

class FashionMnistNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 28x28x1
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 14x14x8
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 7x7x16
            nn.Conv2d(16, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            # 1x1x64
            GeM(p_trainable=True)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 10),
            nn.Softmax()
        )
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        emb = self.model(x).view(-1, 64)
        logits = self.head(emb)
        return {
            "embeddings": emb, 
            "logits": logits
        }

