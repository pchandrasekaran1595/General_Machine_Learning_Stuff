import torch
from time import time

def breaker():
    print("\n" + 50*"-" + "\n")


def fit_sm(model=None, optimizer=None, scheduler=None, epochs=None,
           dataloader=None, criterion=None, device=None, verbose=False):

    breaker()
    print("Training ...")
    breaker()

    model.train()
    model.to(device)

    LP = []
    start_time = time()
    for e in range(epochs):
        lossPerPass = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).view(-1)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            lossPerPass += (loss.item() / y.shape[0])
            optimizer.step()
        LP.append(lossPerPass)

        if scheduler:
            scheduler.step()

        if verbose:
            print("Epoch {} | Loss : {}".format(e+1, lossPerPass))

    breaker()
    print("Time Taken [{} Epochs] : {:.2f} seconds".format(epochs, time()-start_time))
    breaker()
    print("Training Complete")
    breaker()

    return LP

def predict_sm(model=None, dataloader=None, batch_size=None, device=None, path=None):
    model.eval()
    model.to(device)

    Preds = torch.zeros(batch_size, 1).to(device)

    for X in dataloader:
        X = X.to(device)
        with torch.no_grad():
            Prob = torch.exp(model(X))
        Pred = torch.argmax(Prob, dim=1)
        Preds = torch.cat((Preds, Pred.view(-1, 1)), dim=0)

    return Preds[batch_size:].cpu().numpy().reshape(-1).astype(int)

