from MLP import MLP
from DS import DS
import train_fn as t
import eval_fn as e

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader as DL

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sklearn.datasets

def breaker():
    print("\n" + 30 * "-" + "\n")

def head(x=None, no_of_ele=5):
    breaker()
    print(x[no_of_ele:])
    breaker()

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

if __name__ == "__main__":
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

            tr_data_setup = DS(X, y.reshape(-1,1))
            tr_data = DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=True, generator=torch.manual_seed(0))

            model = MLP(cfg.IL, cfg.HL, cfg.OL, cfg.DP1, cfg.DP2, use_dropout=use_DP)
            optimizer = model.getOptimizer()
            
            LP = t.singleModel_train_fn(model=model, optimizer=optimizer, dataloader=tr_data,
                                                      epochs=cfg.epochs, device=cfg.device, criterion=nn.BCEWithLogitsLoss())
        else:
            use_DP = False
            
            cfg = CFG(IL=X.shape[1], HL=[hidden_size])

            tr_data_setup = DS(X, y.reshape(-1,1))
            tr_data = DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=True, generator=torch.manual_seed(0))

            model = MLP(cfg.IL, cfg.HL, cfg.OL, use_dropout=use_DP)
            optimizer = model.getOptimizer()
            
            LP = t.singleModel_train_fn(model=model, optimizer=optimizer, dataloader=tr_data,
                                                      epochs=cfg.epochs, device=cfg.device, criterion=nn.BCEWithLogitsLoss())
        
        ts_data_setup = DS(X_test, None, "test")
        ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)
        y_pred = e.single_eval_fn(model=model, dataloader=ts_data, ts_batch_size=cfg.ts_batch_size, device=cfg.device)

        print("Accuracy Score : {:.5f} %".format(accuracy_score(y_test, y_pred) * 100))
        breaker()
        
        end_char = input("Enter 0 to stop : ")
        if end_char == "0":
            break        

    breaker()
    print("PROGRAM EXECUTION END")
    breaker()
