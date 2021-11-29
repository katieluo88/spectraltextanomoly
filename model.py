import torch.nn as nn
import torch.nn.functional as F


# classifier
class MLPClassifier(nn.Module):
    def __init__(self, embed_dim: int):
        super(MLPClassifier, self).__init__()

        # sequential
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 2))

    def forward(self, inputs):
        vectors = inputs.last_hidden_state[:, 1, :].squeeze(dim=1)
        outputs = self.mlp(vectors)
        return F.softmax(outputs, dim=1)


# classifier
class SpectralClassifier(nn.Module):
    def __init__(self, embed_dim: int):
        super(SpectralClassifier, self).__init__()

        # sequential
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 64),
                                 nn.ReLU(), nn.Linear(64, 2))

    def forward(self, inputs):
        outputs = self.mlp(inputs)
        return F.softmax(outputs, dim=1)
