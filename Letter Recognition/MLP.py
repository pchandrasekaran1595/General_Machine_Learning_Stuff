import torch
from torch import nn, optim
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(this, IL=None, HL=None, OL=None, use_dp=False, DP1=0.2, DP2=0.5):
        super().__init__()

        this.HL = HL
        this.use_dp = use_dp

        if len(HL) == 1:
            this.BN1 = nn.BatchNorm1d(num_features=IL)
            this.FC1 = WN(nn.Linear(in_features=IL, out_features=HL[0]))

            this.BN2 = nn.BatchNorm1d(num_features=HL[0])
            this.FC2 = WN(nn.Linear(in_features=HL[0], out_features=OL))

        elif len(HL) == 2:
            this.BN1 = nn.BatchNorm1d(num_features=IL)
            this.FC1 = WN(nn.Linear(in_features=IL, out_features=HL[0]))

            this.BN2 = nn.BatchNorm1d(num_features=HL[0])
            this.FC2 = WN(nn.Linear(in_features=HL[0], out_features=HL[1]))

            this.BN3 = nn.BatchNorm1d(num_features=HL[1])
            this.FC3 = WN(nn.Linear(in_features=HL[1], out_features=OL))

        elif len(HL) == 3:
            this.BN1 = nn.BatchNorm1d(num_features=IL)
            this.FC1 = WN(nn.Linear(in_features=IL, out_features=HL[0]))

            this.BN2 = nn.BatchNorm1d(num_features=HL[0])
            this.FC2 = WN(nn.Linear(in_features=HL[0], out_features=[1]))

            this.BN3 = nn.BatchNorm1d(num_features=HL[1])
            this.FC3 = WN(nn.Linear(in_features=HL[1], out_features=[2]))

            this.BN4 = nn.BatchNorm1d(num_features=HL[2])
            this.FC4 = WN(nn.Linear(in_features=HL[2], out_features=OL))

        else:
            raise NotImplementedError("Only Supports Networks of Length 1, 2 and 3")

    def getOptimizer(this, lr=1e-3, wd=0):
        return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)

    def getPlateauScheduler(this, optimizer=None, patience=4, eps=1e-6):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, eps=eps, verbose=True)

    def getStepScheduler(this, optimizer, step_size=5, gamma=0.1):
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def forward(this, x):
        if not this.use_dp:
            if len(this.HL) == 1:
                x = this.BN1(x)
                x = F.relu(this.FC1(x))
                x = this.BN2(x)
                x = F.log_softmax(this.FC2(x), dim=1)
                return x
            elif len(this.HL) == 2:
                x = this.BN1(x)
                x = F.relu(this.FC1(x))
                x = this.BN2(x)
                x = F.relu(this.FC2(x))
                x = this.BN3(x)
                x = F.log_softmax(this.FC3(x), dim=1)
                return x
            else:
                x = this.BN1(x)
                x = F.relu(this.FC1(x))
                x = this.BN2(x)
                x = F.relu(this.FC2(x))
                x = this.BN3(x)
                x = F.relu(this.FC3(x))
                x = this.BN4(x)
                x = F.log_softmax(this.FC4(x), dim=1)
                return x
        else:
            if len(this.HL) == 1:
                x = this.BN1(x)
                x = this.DP1(x)
                x = F.relu(this.FC1(x))
                x = this.BN2(x)
                x = this.DP2(x)
                x = F.log_softmax(this.FC2(x), dim=1)
                return x
            elif len(this.HL) == 2:
                x = this.BN1(x)
                x = this.DP1(x)
                x = F.relu(this.FC1(x))
                x = this.BN2(x)
                x = this.DP2(x)
                x = F.relu(this.FC2(x))
                x = this.BN3(x)
                x = this.DP2(x)
                x = F.log_softmax(this.FC3(x), dim=1)
                return x
            else:
                x = this.BN1(x)
                x = this.DP1(x)
                x = F.relu(this.FC1(x))
                x = this.BN2(x)
                x = this.DP2(x)
                x = F.relu(this.FC2(x))
                x = this.BN3(x)
                x = this.DP2(x)
                x = F.relu(this.FC3(x))
                x = this.BN4(x)
                x = this.DP2(x)
                x = F.log_softmax(this.FC4(x), dim=1)
                return x
