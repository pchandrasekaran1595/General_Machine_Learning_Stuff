########## Use this as a reference in creating templates ##########

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sklearn.datasets

import gc

datapath = "C:/Users/Ourself/Desktop/Machine Learning/Projects/BRCA/"

def breaker():
    print("\n" + 30 * "-" + "\n")

def head(x=None, no_of_ele=5):
    breaker()
    print(x[no_of_ele:])
    breaker()

def KFold_train_fn(X=None, y=None, epochs=None, n_folds=None,
                   IL=None, HL=None, OL=None, DP1=0.2, DP2=0.5,
                   use_dp=None, lr=None, wd=None,
                   tr_batch_size=None, va_batch_size=None,
                   device=None, criterion=None):

    breaker()
    print("Training ...")
    breaker()

    bestLoss = {"train" : np.inf, "valid" : np.inf}
    names = []
    LP = []
    fold = 0

    start_time = time()
    for tr_idx, va_idx in KFold(n_splits=n_folds, shuffle=True, random_state=0).split(X, y):
        print("Processing Fold {fold} ...".format(fold=fold+1))

        X_train, X_valid, y_train, y_valid = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

        tr_data_setup = DS(X_train, y_train.reshape(-1,1))
        va_data_setup = DS(X_valid, y_valid.reshape(-1,1))

        DLS = {"train" : DL(tr_data_setup, batch_size=tr_batch_size, shuffle=True, generator=torch.manual_seed(0)),
               "valid" : DL(va_data_setup, batch_size=va_batch_size, shuffle=False)
              }

        torch.manual_seed(0)
        model = MLP(IL, HL, OL, DP1, DP2, use_dp)
        model.to(device)

        optimizer = model.getOptimizer(lr=lr, wd=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, eps=1e-6, verbose=True)

        for e in range(epochs):
            epochLoss = {"train" : 0.0, "valid" : 0}
            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                lossPerPass = 0

                for f, l in DLS[phase]:
                    f, l = f.to(cfg.device), l.to(cfg.device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        output = model(f)
                        loss   = criterion(output, l)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    lossPerPass += (loss.item() / l.shape[0])
                epochLoss[phase] = lossPerPass
            LP.append(epochLoss)
            scheduler.step(epochLoss["valid"])
            if epochLoss["valid"] < bestLoss["valid"]:
                bestLoss = epochLoss
                name = "./Model_Fold_{fold}.pt".format(fold=fold)
                names.append(name)
                torch.save(model.state_dict(), datapath + name)
        fold += 1

    breaker()
    print("Time Taken to Train {f} folds for {e} epochs : {:.2f} minutes".format((time() - start_time)/60, f=n_folds, e=epochs))
    breaker()
    print("Training Complete")
    breaker()

    return LP, names, model

def eval_fn(model=None, names=None, dataloader=None, num_obs=None, ts_batch_size=None, device=None):
    y_pred = np.zeros((num_obs, 1))

    for name in names:
        Pred = torch.zeros(ts_batch_size, 1).to(device)

        model.load_state_dict(torch.load(datapath + name))
        model.eval()

        for X in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = torch.sigmoid(model(X))
            Pred = torch.cat((Pred, output), dim=0)
        Pred = Pred[ts_batch_size:].cpu().numpy()
        y_pred = np.add(y_pred, Pred)

    y_pred = np.divide(y_pred, len(names))

    y_pred[np.argwhere(y_pred <= 0.5)] = 0
    y_pred[np.argwhere(y_pred > 0.5)]  = 1
    return y_pred
        
class DS(Dataset):
    def __init__(this, X=None, y=None, mode="train"):
        this.mode = mode
        this.X = X
        if mode == "train":
            this.y = y

    def __len__(this):
        return this.X.shape[0]

    def __getitem__(this, idx):
        if this.mode == "train":
            return torch.FloatTensor(this.X[idx]), torch.FloatTensor(this.y[idx])
        else:
            return torch.FloatTensor(this.X[idx])


class MLP(nn.Module):
    def __init__(this, IL=None, HL=None, OL=None, DP1=None, DP2=None, use_dropout=False):
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
    
        
class CFG():
    tr_batch_size = 64
    va_batch_size = 64
    ts_batch_size = 64

    epochs  = 30
    n_folds = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    OL = 1

    def __init__(this, IL=None, HL=None, DP1=None, DP2=None):
        this.IL  = IL
        this.HL  = HL
        this.DP1 = DP1
        this.DP2 = DP2


while True:

    breaker()
    print("SKLEARN BREAST CANCER CLASSIFICATION")
    breaker()
    
    brca = sklearn.datasets.load_breast_cancer()

    brca_df = pd.DataFrame(brca["data"], columns=brca["feature_names"])

    features = brca_df.copy().values
    labels   = brca["target"].copy()

    del brca, brca_df

    X, X_test, y, y_test = train_test_split(features, labels, test_size=169, shuffle=True, random_state=0)

    hidden_size = int(input("Enter the number of hidden neurons : "))

    use_DP = input("\nUse Dropout {Y, N} : ")
    if use_DP == "Y":
        use_DP = True
        dropout_1   = float(input("\nEnter DP1 Probability : "))
        dropout_2   = float(input("\nEnter DP2 Probability : "))
        
        cfg = CFG(IL=X.shape[1], HL=[hidden_size], DP1=dropout_1, DP2=dropout_2)
        
        LP, Names, Model = KFold_train_fn(X=X, y=y, epochs=cfg.epochs, n_folds=cfg.n_folds,
                                      IL=cfg.IL, HL=cfg.HL, OL=cfg.OL, DP1=cfg.DP1, DP2=cfg.DP2,
                                      use_dp=use_DP, lr=1e-3, wd=0,
                                      tr_batch_size=cfg.tr_batch_size,
                                      va_batch_size=cfg.va_batch_size,
                                      device=cfg.device, criterion=nn.BCEWithLogitsLoss())
    else:
        use_DP = False
        
        cfg = CFG(IL=X.shape[1], HL=[hidden_size])
        
        LP, Names, Model = KFold_train_fn(X=X, y=y, epochs=cfg.epochs, n_folds=cfg.n_folds,
                                      IL=cfg.IL, HL=cfg.HL, OL=cfg.OL,
                                      use_dp=use_DP, lr=1e-3, wd=0,
                                      tr_batch_size=cfg.tr_batch_size,
                                      va_batch_size=cfg.va_batch_size,
                                      device=cfg.device, criterion=nn.BCEWithLogitsLoss())
    
    

    

    ts_data_setup = DS(X_test, None, "test")
    ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)
    y_pred = eval_fn(Model, Names, ts_data, ts_data_setup.__len__(), cfg.ts_batch_size, cfg.device)

    print("Accuracy Score : {:.5f} %".format(accuracy_score(y_test, y_pred) * 100))
    breaker()
    
    end_char = input("Enter 0 to stop : ")
    if end_char == "0":
        break    

breaker()
print("PROGRAM EXECUTION END")
breaker()
