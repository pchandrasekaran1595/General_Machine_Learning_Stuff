import numpy as np
import torch

from sklearn.metrics import accuracy_score
from time import time


def breaker():
    print("\n" + 50*"-" + "\n")


def getAccuracy(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred[np.argwhere(y_pred > 0.5)] = 1
    y_pred[np.argwhere(y_pred <= 0.5)] = 0

    return accuracy_score(y_true, y_pred)

def fit_sm(model=None, optimizer=None, scheduler=None, epochs=None,
           trainloader=None, validloader=None,
           criterion=None, device=None,
           verbose=False, path=None):

    breaker()
    print("Training ...")
    breaker()

    TRL = []
    TVL = []
    TRA = []
    TVA = []
    model.to(device)

    start_time = time()
    for e in range(epochs):
        e_trl = []
        e_tvl = []
        e_tra = []
        e_tva = []

        model.train()
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            e_trl.append(loss.item() / y.shape[0])
            e_tra.append(getAccuracy(y, output))

        tr_mean_loss = np.mean(np.array(e_trl))
        tr_mean_accs = np.mean(np.array(e_tra))

        TRL.append(tr_mean_loss)
        TRA.append(tr_mean_accs)

        if scheduler:
            scheduler.step()

        model.eval()
        for X, y in validloader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                output = model(X)
                loss = criterion(output, y)
                e_tvl.append(loss.item() / y.shape[0])
                e_tva.append(getAccuracy(y, output))
        va_mean_loss = np.mean(np.array(e_tvl))
        va_mean_accs = np.mean(np.array(e_tva))

        TVL.append(va_mean_loss)
        TVA.append(va_mean_accs)

        if verbose:
            print("Epoch : {} | Train Loss : {:.5f} | Valid Loss : {:.5f} \
| Train Accuracy : {:.5f} | Valid Accuracy : {:.5f}".format(e+1, tr_mean_loss, va_mean_loss, tr_mean_accs, va_mean_accs))

    breaker()
    print("Time Taken [{} Epochs] : {:.2f} seconds".format(epochs, (time()-start_time)/60))
    breaker()
    print("Training Complete")
    breaker()

    return TRL, TVL, TRA, TVA


def predict(model=None, dataloader=None, batch_size=None, device=None):
    model.eval()

    y_pred = torch.zeros(batch_size, 1).to(device)
    for X, y in dataloader:
        X = X.to(device)
        with torch.no_grad():
            output = model(X)
        y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)
    y_pred = y_pred[batch_size:].detach().cpu().numpy()

    y_pred[np.argwhere(y_pred > 0.5)] = 1
    y_pred[np.argwhere(y_pred <= 0.5)] = 0

    return y_pred.reshape(-1)
