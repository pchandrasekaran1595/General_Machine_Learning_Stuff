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

    Losses = []
    Accuracies = []

    DLS = {"train": trainloader, "valid": validloader}

    start_time = time()
    for e in range(epochs):
        epochLoss = {"train": 0, "valid": 0}
        epochAccs = {"train": 0, "valid": 0}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            lossPerPass = []
            accuracy = []

            for X, y in DLS[phase]:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(X)
                    loss = criterion(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item() / y.shape[0])
                accuracy.append(getAccuracy(y, output))
            epochLoss[phase] = np.mean(np.array(lossPerPass))
            epochAccs[phase] = np.mean(np.array(accuracy))
        Losses.append(epochLoss)
        Accuracies.append(epochAccs)

        if scheduler:
            scheduler.step()

        if verbose:
            print("Epoch : {} | Train Loss : {:.5f} | Valid Loss : {:.5f} \
| Train Accuracy : {:.5f} | Valid Accuracy : {:.5f}".format(e + 1, epochLoss["train"], epochLoss["valid"],
                                                            epochAccs["train"],
                                                            epochAccs["valid"]))

    breaker()
    print("Time Taken [{} Epochs] : {:.2f} seconds".format(epochs, (time() - start_time) / 60))
    breaker()
    print("Training Complete")
    breaker()

    return Losses, Accuracies


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
