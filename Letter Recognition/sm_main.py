import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from Dataset import Dataset
from MLP import MLP
import fit_predict as fp

import torch
from torch import nn
from torch.utils.data import DataLoader as DL

import gc


def breaker():
    print("\n" + 50*"-" + "\n")


def head(x=None, no_of_ele=5):
    breaker()
    print(x[:no_of_ele])
    breaker()


class CFG():
    tr_batch_size = 128
    ts_batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OL = 26

    def __init__(this, IL=None, HL=None, epochs=None, use_dp=None, DP1=None, DP2=None):
        this.IL = IL
        this.HL = HL
        this.epochs = epochs
        this.use_dp = use_dp
        this.DP1 = DP1
        this.DP2 = DP2


root_dir = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Letter Recognition/"
sc_X = StandardScaler()

if __name__ == "__main__":
    data = pd.read_csv(root_dir + "data.csv", header=None)

    columns = ["Letter", "X_Box", "Y_Box", "Width", "Height", "Onpix",
               "Mean(X)", "Mean(Y)", "Mean(X_Var)", "Mean(Y_Bar)", "XY_Correlation",
               "Mean(X^2**Y)", "Mean(X*Y^2)", "Mean(Edge Count L->R)[X_ege]",
               "(X_ege)Y_Correlation", "Mean(Edge Count B->T)[Y_ege]",
               "X(Y_ege)_Correlation"
               ]

    data.columns = columns

    numpy_data = data.copy().values

    le = LabelEncoder()
    numpy_data[:, 0] = le.fit_transform(numpy_data[:, 0])

    X, X_test, y, y_test = train_test_split(numpy_data[:, 1:],
                                            numpy_data[:, 0],
                                            test_size=0.2,
                                            shuffle=True,
                                            random_state=0,
                                            stratify=numpy_data[:, 0])

    X = X.astype(float)
    X_test = X_test.astype(float)
    y = y.astype(int)
    y_test = y_test.astype(int)

    cfg = CFG(IL=X.shape[1], HL=[128, 64], epochs=50, use_dp=False)

    torch.manual_seed(0)
    tr_data_setup = Dataset(X, y.reshape(-1, 1))
    tr_data = DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=True)

    Model = MLP(cfg.IL, cfg.HL, cfg.OL, cfg.use_dp)
    optimizer = Model.getOptimizer(lr=1e-3, wd=0)

    LP = fp.fit_sm(model=Model, optimizer=optimizer, epochs=cfg.epochs, dataloader=tr_data,
                   criterion=nn.NLLLoss(),
                   device=cfg.device, verbose=True
                   )

    plt.figure(figsize=(8, 5))
    plt.plot([i+1 for i in range(len(LP))], LP, "r")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    ts_data_setup = Dataset(X_test, None, "test")
    ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

    y_pred = fp.predict_sm(model=Model, dataloader=ts_data, batch_size=cfg.tr_batch_size, device=cfg.device)

    print("Accuracy  : {:.5f}".format(accuracy_score(y_test, y_pred)))
    print("Precision : {:.5f}".format(precision_score(y_test, y_pred, labels=numpy_data[:, 0], average="macro")))
    print("Recall    : {:.5f}".format(recall_score(y_test, y_pred, labels=numpy_data[:, 0], average="macro")))

    breaker()
    print("Program Execution Complete")
    breaker()
