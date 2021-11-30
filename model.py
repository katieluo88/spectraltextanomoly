import torch.nn as nn
import torch.nn.functional as F
import torch

from scipy.fftpack import dct, idct
import numpy as np


# classifier
class MLPClassifier(nn.Module):
    def __init__(self, embed_dim: int):
        super(MLPClassifier, self).__init__()

        # sequential: embed dim = 768
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 2))

    def forward(self, inputs, attention_mask):
        embeds = inputs.last_hidden_state
        embeds_mask = attention_mask.unsqueeze(dim=2).repeat(1, 1, embeds.shape[2])
        embeds[~embeds_mask] = 0
        # Remove CLS 
        embeds = inputs.last_hidden_state[:, 1:, :]
        vectors = torch.mean(embeds, dim=1)
        outputs = self.mlp(vectors)
        return F.softmax(outputs, dim=1)


'''
# tokens    DCT
1-2         



'''


# classifier
class SpectralClassifier(nn.Module):
    def __init__(self, embed_dim: int, filter: str):
        super(SpectralClassifier, self).__init__()

        # sequential
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 2))
        if filter == 'low':
            self.i, self.j = 0, 16
        elif filter == 'mid':
            self.i, self.j = 16, 256
        elif filter == 'high':
            self.i, self.j = 256, 768

    def _filter(self, embeds):
        dcts = dct(embeds, type=2, axis=1)
        dcts[:, :self.i, :] = 0
        dcts[:, self.j:, :] = 0
        idcts = idct(embeds, type=2, axis=1) / 256
        return idcts

    def forward(self, inputs, attention_mask):
        embeds = inputs.last_hidden_state
        embeds_mask = attention_mask.unsqueeze(dim=2).repeat(1, 1, embeds.shape[2])
        embeds[~embeds_mask] = 0
        embeds = inputs.last_hidden_state[:, 1:, :]
        # DCT
        embeds = self._filter(embeds)
        vectors = torch.mean(embeds.squeeze(dim=1))
        outputs = self.mlp(vectors)
        return F.softmax(outputs, dim=1)


# classifier
class DCTClassifier(nn.Module):
    def __init__(self, embed_dim: int):
        super(DCTClassifier, self).__init__()

        # sequential
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 2))

    def forward(self, inputs, attention_mask):
        embeds = inputs.last_hidden_state
        embeds_mask = attention_mask.unsqueeze(dim=2).repeat(1, 1, embeds.shape[2])
        embeds[~embeds_mask] = 0
        embeds = inputs.last_hidden_state[:, 1:, :]
        dcts = dct(embeds.cpu().detach().numpy(), type=2, axis=1)
        dcts = torch.tensor(dcts).to(attention_mask.device)
        vectors = torch.mean(dcts.squeeze(dim=1))
        outputs = self.mlp(vectors)
        return F.softmax(outputs, dim=1)
