import torch
import numpy as np

from torch.utils.data import DataLoader as DL
from sklearn.model_selection import StratifiedKFold
from time import time
from Dataset import Dataset
from MLP import MLP

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def breaker():
    print("\n" + 50*"-" + "\n")

def fit(X=None, y=None, epochs=None, n_folds=None, use_all=False,
        IL=None, HL=None, OL=None,
        use_dp=None, DP1=0.2, DP2=0.5,
        lr=None, wd=None, patience=None, lr_eps=None,
        tr_batch_size=None, va_batch_size=None,
        criterion=None, device=None, path=None, verbose=False):

    breaker()
    print("Training ...")
    breaker()

    LP = []
    names = []
    bestLoss = {"train" : np.inf, "valid" : np.inf}
    fold = 1

    start_time = time()
    for tr_idx, va_idx in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0).split(X, y):
        print("Processing Fold {}...".format(fold))

        X_train, X_valid, y_train, y_valid = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

        tr_data_setup = Dataset(X_train, y_train.reshape(-1,1))
        va_data_setup = Dataset(X_valid, y_valid.reshape(-1,1))

        DLS = {"train" : DL(tr_data_setup, batch_size=tr_batch_size, shuffle=True, generator=torch.manual_seed(0)),
               "valid" : DL(va_data_setup, batch_size=va_batch_size, shuffle=True)}

        torch.manual_seed(0)
        model = MLP(IL, HL, OL, use_dp, DP1, DP2)

        optimizer = model.getOptimizer(lr=lr, wd=wd)
        scheduler = model.getPlateauScheduler(optimizer, patience, lr_eps)

        for e in range(epochs):
            epochLoss = {"train" : 0.0, "valid" : 0.0}
            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                lossPerPass = 0

                for features, labels in DLS[phase]:
                    features, labels = features.to(device), labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        output = model(features)
                        loss = criterion(output, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    lossPerPass += (loss.item() / labels.shape[0])
                epochLoss[phase] = lossPerPass
            LP.append(epochLoss)
            scheduler.step(epochLoss["valid"])
            if not use_all:
                if epochLoss["valid"] < bestLoss["valid"]:
                    bestLoss = epochLoss
                    name = "Model_Fold_{}.pt".format(fold)
                    names.append(name)
                    torch.save(model.state_dict(), path + name)
            else:
                name = "Model_Fold_{}.pt".format(fold)
                names.append(name)
                torch.save(model.state_dict(), path + name)
                if epochLoss["valid"] < bestLoss["valid"]:
                    bestLoss = epochLoss
            if verbose:
                print("Fold {}, Epoch {} - TL : {}, VL : {}".format(fold, e, epochLoss["train"], epochLoss["valid"]))
        fold += 1

    breaker()
    print("Time Taken [{} Epochs, {}, Folds] : {:.2f} minutes".format(epochs, n_folds, (time()-start_time)/60))
    breaker()
    print("Training Complete")
    breaker()

    return LP, names

def predict(model=None, names=None, dataloader=None, num_obs=None, batch_size=None, device=None, path=None):
    y_pred = np.zeros((num_obs, 1))

    for name in names:
        model.load_state_dict(torch.load(path + name))
        model.eval()

        Pred = torch.zeros(batch_size, 1).to(device)
        for X in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = torch.exp(model(X))
            Pred = torch.cat((Pred, output), dim=0)
        Pred = Pred[batch_size:].cpu().numpy()
        y_pred = np.add(y_pred, Pred)
    y_pred = np.divide(y_pred, len(names))

    y_pred[np.argwhere(y_pred <= 0.5)] = 0
    y_pred[np.argwhere(y_pred > 0.5)] = 1
    return y_pred.reshape(-1)


def predict_proba(model=None, names=None, dataloader=None, num_obs=None, batch_size=None, device=None, path=None):
    y_pred = np.zeros((num_obs, 1))

    for name in names:
        model.load_state_dict(torch.load(path + name))
        model.eval()

        Pred = torch.zeros(batch_size, 1).to(device)
        for X in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = torch.exp(model(X))
            Pred = torch.cat((Pred, output), dim=0)
        Pred = Pred[batch_size:].cpu().numpy()
        y_pred = np.add(y_pred, Pred)
    y_pred = np.divide(y_pred, len(names))

    return y_pred.reshape(-1)
