from re import L
import torch.nn as nn
import torch.nn.functional as F
import torch

from scipy.fftpack import dct, idct


# classifier
class MLPClassifier(nn.Module):
    def __init__(self, embed_dim: int):
        super(MLPClassifier, self).__init__()

        # sequential: embed dim = 768
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 2))

    def forward(self, inputs, attention_mask, return_prob=False):
        embeds = inputs.last_hidden_state
        embeds_mask = attention_mask.unsqueeze(dim=2).repeat(1, 1, embeds.shape[2])
        embeds[~embeds_mask] = 0
        # Remove CLS
        embeds = inputs.last_hidden_state[:, 1:, :]
        # or torch.max(x,0).values
        vectors = torch.mean(embeds, dim=1)
        outputs = self.mlp(vectors)
        if return_prob:
            return F.softmax(outputs, dim=1)
        return outputs


'''
# tokens    DCT
1-2         



'''


# classifier
class SpectralClassifier(nn.Module):
    def __init__(self, embed_dim: int, filter: str, max_seq_len: int):
        super(SpectralClassifier, self).__init__()

        # sequential: embed dim = 768
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 2))
        x = max_seq_len - 1
        low = int(x / 8)
        mid = int(x / 2)
        if filter == 'low':
            self.i, self.j = 0, low
        elif filter == 'mid':
            self.i, self.j = low, mid
        elif filter == 'high':
            self.i, self.j = mid, x

    def _filter(self, embeds):
        dcts = dct(embeds.cpu().detach().numpy(), type=2, axis=1)
        dcts[:, :self.i, :] = 0
        dcts[:, self.j:, :] = 0
        idcts = idct(dcts, type=2, axis=1) / 254
        return idcts

    def forward(self, inputs, attention_mask, return_prob=False):
        embeds = inputs.last_hidden_state
        embeds_mask = attention_mask.unsqueeze(dim=2).repeat(1, 1, embeds.shape[2])
        embeds[~embeds_mask] = 0
        embeds = inputs.last_hidden_state[:, 1:, :]
        idcts = self._filter(embeds)
        # print(idcts)
        idcts = torch.tensor(idcts).to(attention_mask.device)
        vectors = torch.mean(idcts, dim=1)
        outputs = self.mlp(vectors)
        if return_prob:
            return F.softmax(outputs, dim=1)
        return outputs


class DCTClassifier(nn.Module):
    def __init__(self, embed_dim: int, max_seq_len: int):
        super(DCTClassifier, self).__init__()

        # sequential: embed dim = 768
        self.projection = nn.Linear(max_seq_len - 1, 1)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 2))

    def forward(self, inputs, attention_mask, return_prob=False):
        embeds = inputs.last_hidden_state
        embeds_mask = attention_mask.unsqueeze(dim=2).repeat(1, 1, embeds.shape[2])
        embeds[~embeds_mask] = 0
        embeds = inputs.last_hidden_state[:, 1:, :]
        dcts = dct(embeds.cpu().detach().numpy(), type=2, axis=1)
        dcts = torch.tensor(dcts).to(attention_mask.device)
        vectors = self.projection(torch.transpose(dcts, 1, 2)).squeeze(dim=2)
        outputs = self.mlp(vectors)
        if return_prob:
            return F.softmax(outputs, dim=1)
        return outputs
