import torch
from time import time
from sklearn.metrics import accuracy_score
import numpy as np

def breaker():
    print("\n" + 50*"-" + "\n")


def fit_sm(model=None, optimizer=None, scheduler=None, epochs=None,
           trainloader=None, validloader=None,
           criterion=None, device=None, verbose=False):

    breaker()
    print("Training ...")
    breaker()

    model.train()
    model.to(device)

    TRL = []
    TVL = []
    TRA = []
    TVA = []

    start_time = time()
    for e in range(epochs):
        e_trl = []
        e_tvl = []
        e_tra = []
        e_tva = []

        for X, y in trainloader:
            X, y = X.to(device), y.to(device).view(-1)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            e_trl.append(loss.item() / y.shape[0])
            e_tra.append(accuracy_score(torch.argmax(torch.exp(output), dim=1).detach().cpu().numpy(), y))

        tr_mean_loss = np.mean(np.array(e_trl))
        tr_mean_accs = np.mean(np.array(e_tra))

        TRL.append(tr_mean_loss)
        TRA.append(tr_mean_accs)

        if scheduler:
            scheduler.step()

        for X, y in validloader:
            X, y = X.to(device), y.to(device).view(-1)

            with torch.no_grad():
                output = model(X)
                loss = criterion(output, y)
                e_tvl.append(loss.item() / y.shape[0])
                e_tva.append(accuracy_score(torch.argmax(torch.exp(output), dim=1).detach().cpu().numpy(), y))

        va_mean_loss = np.mean(np.array(e_tvl))
        va_mean_accs = np.mean(np.array(e_tva))

        TVL.append(va_mean_loss)
        TVA.append(va_mean_accs)

        if verbose:
            print("Epoch : {} | Train Loss : {:.5f} | Valid Loss : {:.5f} \
| Train Accuracy : {:.5f} | Valid Accuracy : {:.5f}".format(e + 1, tr_mean_loss, va_mean_loss, tr_mean_accs,
                                                            va_mean_accs))

    breaker()
    print("Time Taken [{} Epochs] : {:.2f} seconds".format(epochs, time()-start_time))
    breaker()
    print("Training Complete")
    breaker()

    return TRL, TVL, TRA, TVA

def predict_sm(model=None, dataloader=None, batch_size=None, device=None, path=None):
    model.eval()
    model.to(device)

    y_pred = torch.zeros(batch_size, 1).to(device)

    for X, y in dataloader:
        X = X.to(device)
        with torch.no_grad():
            Prob = torch.exp(model(X))
        Pred = torch.argmax(Prob, dim=1)
        y_pred = torch.cat((y_pred, Pred.view(-1, 1)), dim=0)

    return y_pred[batch_size:].cpu().numpy().reshape(-1).astype(int)
