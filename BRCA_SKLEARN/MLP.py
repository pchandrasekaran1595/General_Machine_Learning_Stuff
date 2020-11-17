import torch
from torch import nn, optim
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(this, IL=None, HL=None, OL=None, DP1=0.2, DP2=0.5, use_dropout=False):
        super(MLP, this).__init__()

        this.use_dropout = use_dropout

        this.DP1 = nn.Dropout(p=DP1)
        this.DP2 = nn.Dropout(p=DP2)

        this.BN1 = nn.BatchNorm1d(num_features=IL)
        this.FC1 = nn.Linear(in_features=IL, out_features=HL[0])

        this.BN2 = nn.BatchNorm1d(num_features=HL[0])
        this.FC2 = WN(nn.Linear(in_features=HL[0], out_features=OL))

    def getOptimizer(this, lr=1e-3, wd=0):
        return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)

    def forward(this, x):
        if not this.use_dropout:
            x = this.BN1(x)
            x = F.relu(this.FC1(x))
            x = this.BN2(x)
            x = this.FC2(x)
            return x
        else:
            x = this.BN1(x)
            x = this.DP1(x)
            x = F.relu(this.FC1(x))
            x = this.BN2(x)
            x = this.DP2(x)
            x = this.FC2(x)
            return x
