import torch
from time import time
from sklearn.metrics import accuracy_score
import numpy as np

def breaker():
    print("\n" + 50*"-" + "\n")


def fit_sm(model=None, optimizer=None, scheduler=None, epochs=None,
           trainloader=None, validloader=None,
           criterion=None, device=None,
           verbose=False):
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
                X, y = X.to(device), y.to(device).view(-1)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(X)
                    loss = criterion(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item() / y.shape[0])
                accuracy.append(accuracy_score(torch.argmax(torch.exp(output), dim=1), y))
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

def predict_sm(model=None, dataloader=None, batch_size=None, device=None, path=None):
    model.eval()
    model.to(device)

    Preds = torch.zeros(batch_size, 1).to(device)

    for X, y in dataloader:
        X = X.to(device)
        with torch.no_grad():
            Prob = torch.exp(model(X))
        Pred = torch.argmax(Prob, dim=1)
        Preds = torch.cat((Preds, Pred.view(-1, 1)), dim=0)

    return Preds[batch_size:].cpu().numpy().reshape(-1).astype(int)
