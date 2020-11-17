import torch
from torch import optim
from torch.utils.data import DataLoader as DL
from MLP import MLP
from DS import DS
from time import time
from sklearn.model_selection import KFold
import numpy as np

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
                   device=None, criterion=None, path=None):

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
                    f, l = f.to(device), l.to(device)

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
                torch.save(model.state_dict(), path + name)
        fold += 1

    breaker()
    print("Time Taken to Train {f} folds for {e} epochs : {:.2f} minutes".format((time() - start_time)/60, f=n_folds, e=epochs))
    breaker()
    print("Training Complete")
    breaker()

    return LP, names, model


def singleModel_train_fn(model=None, optimizer=None, dataloader=None, epochs=None, device=None, criterion=None):
    breaker()
    print("Training ...")

    LP = []

    start_time = time()
    for e in range(epochs):
        lossPerPass = 0
        for X, y in dataloader:
            X, y =  X.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss   = criterion(output, y)
            loss.backward()
            lossPerPass += (loss.item() / y.shape[0])
            optimizer.step()
        LP.append(lossPerPass)

    breaker()
    print("Time Taken to Train for {e} epochs : {:.2f} minutes".format((time() - start_time)/60, e=epochs))
    breaker()
    print("Training Complete")
    breaker()

    return LP
      


