import numpy as np
import torch

def ensemble_eval_fn(model=None, names=None, dataloader=None, num_obs=None, ts_batch_size=None, device=None, path=None):
    y_pred = np.zeros((num_obs, 1))

    for name in names:
        Pred = torch.zeros(ts_batch_size, 1).to(device)

        model.load_state_dict(torch.load(path + name))
        model.eval()

        for X in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = torch.sigmoid(model(X))
            Pred = torch.cat((Pred, output), dim=0)
        Pred = Pred[ts_batch_size:].cpu().numpy()
        y_pred = np.add(y_pred, Pred)

    y_pred = np.divide(y_pred, len(names))

    y_pred[np.argwhere(y_pred <= 0.5)] = 0
    y_pred[np.argwhere(y_pred > 0.5)]  = 1
    return y_pred

def single_eval_fn(model=None, dataloader=None, ts_batch_size=None, device=None):

    Pred = torch.zeros(ts_batch_size, 1).to(device)

    model.eval()

    for X in dataloader:
        X = X.to(device)
        with torch.no_grad():
            output = torch.sigmoid(model(X))
        Pred = torch.cat((Pred, output), dim=0)
    Pred = Pred[ts_batch_size:].cpu().numpy()

    Pred[np.argwhere(Pred <= 0.5)] = 0
    Pred[np.argwhere(Pred > 0.5)]  = 1
    return Pred
