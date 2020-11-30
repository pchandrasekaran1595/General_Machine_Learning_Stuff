import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from Dataset import Dataset
from MLP import MLP
import fit_predict as fp

import torch
from torch import nn
from torch.utils.data import DataLoader as DL

def breaker():
    print("\n" + 50 * "-" + "\n")


def preprocess(x=None, *args):
    df = x.copy()
    df[args[0]] = df[args[0]].map({"b" : 0, "g" : 1})
    return df


class CFG():
    tr_batch_size = 64
    va_batch_size = 64
    ts_batch_size = 51

    OL = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(this, IL=None, HL=None, epochs=None, n_folds=None, use_dp=None, DP1=None, DP2=None):
        this.IL = IL
        this.HL = HL
        this.epochs = epochs
        this.n_folds = n_folds
        this.use_dp = use_dp
        this.DP1 = DP1
        this.DP2 = DP2


sc_X = StandardScaler()
root_dir = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Ionosphere Data/"
model_dir = root_dir + "Models/"

if __name__ == "__main__":
    data = pd.read_csv(root_dir + "ionosphere.csv")

    columns = [i+1 for i in range(data.shape[1])]

    data.columns = columns
    data = preprocess(data, columns[-1])

    X, X_test, y, y_test = train_test_split(data.iloc[:, :-1].copy().values,
                                            data.iloc[:, -1].copy().values,
                                            test_size=51,
                                            shuffle=True,
                                            random_state=0,
                                            stratify=data.iloc[:, -1].copy().values)

    X = sc_X.fit_transform(X)
    X_test = sc_X.transform(X_test)

    cfg = CFG(IL=X.shape[1], HL=[128, 64], epochs=50, n_folds=4, use_dp=True, DP1=0.2, DP2=0.5)

    LP, Names = fp.fit(X=X, y=y, epochs=cfg.epochs, n_folds=cfg.n_folds, use_all=False,
                       IL=cfg.IL, HL=cfg.HL, OL=cfg.OL, use_dp=cfg.use_dp, DP1=cfg.DP1, DP2=cfg.DP2,
                       lr=1e-3, wd=1e-5, patience=4, lr_eps=1e-6,
                       tr_batch_size=cfg.tr_batch_size,
                       va_batch_size=cfg.va_batch_size,
                       criterion=nn.BCEWithLogitsLoss(),
                       device=cfg.device,
                       path=model_dir,
                       verbose=True)

    LPV = []
    LPT = []
    for i in range(len(LP)):
        LPT.append(LP[i]["train"])
        LPV.append(LP[i]["valid"])

    xAxis = [i + 1 for i in range(cfg.epochs)]
    plt.figure(figsize=(10, 30))
    for fold in range(cfg.n_folds):
        plt.subplot(cfg.n_folds, 1, fold + 1)
        plt.plot(xAxis, LPT[fold * cfg.epochs:(fold + 1) * cfg.epochs], "b", label="Training Loss")
        plt.plot(xAxis, LPV[fold * cfg.epochs:(fold + 1) * cfg.epochs], "r--", label="Validation Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Fold {fold}".format(fold=fold + 1))
    plt.show(block=False)
    plt.pause(2.5)
    plt.close()

    torch.manual_seed(0)
    Model = MLP(cfg.IL, cfg.HL, cfg.OL, cfg.use_dp)

    ts_data_setup = Dataset(X_test, None, "test")
    ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

    y_pred = fp.predict(Model, set(Names), ts_data, ts_data_setup.__len__(), cfg.ts_batch_size,
                        cfg.device, model_dir)

    print("Accuracy, Log Loss : {:5f}, {:5f}".format(accuracy_score(y_test, y_pred),
                                                     log_loss(y_test, y_pred, labels=[0, 1], eps=1e-15)))
    breaker()
    print("PROGRAM EXECUTION COMPLETE")
    breaker()
