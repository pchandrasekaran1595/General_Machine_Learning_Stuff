import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
from torch.utils.data import DataLoader as DL
from torch import nn

from Model import ANN
from Dataset import Dataset
import eval_sklearn as esk
import fit_predict as fp

def breaker():
    print("\n" + 50*"-" + "\n")

def preprocess(x=None, *args):
    df = x.copy()
    df[args[0]] = df[args[0]].map({"Existing Customer" : 0, "Attrited Customer" : 1})
    df[args[1]] = df[args[1]].map({"M" : 0, "F" : 1})
    df[args[2]] = df[args[2]].map({"High School" : 0, "Graduate" : 1, "Uneducated" : 2, "Unknown" : 3,
                                   "College" : 4, "Post-Graduate" : 5, "Doctorate" : 6})
    df[args[3]] = df[args[3]].map({"Married" : 0, "Single" : 1, "Unknown" : 2, "Divorced" : 3})
    df[args[4]] = df[args[4]].map({"Less than $40K" : 0, "$40K - $60K" : 1, "$60K - $80K" : 2,
                                   "$80K - $120K" : 3, "$120K +" : 4, "Unknown" : 5})
    df[args[5]] = df[args[5]].map({"Blue" : 0, "Silver" : 1, "Gold" : 2, "Platinum" : 3})
    return df


class CFG():
    tr_batch_size = 128
    ts_batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OL = 1

    def __init__(this, IL=None, HL=None, epochs=None, use_dp=False, DP1=None, DP2=None):
        this.IL = IL
        this.epochs = epochs
        this.HL = HL
        this.use_dp = use_dp
        if this.use_dp:
            this.DP1 = DP1
            this.DP2 = DP2


root_dir = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Bank Churners/"

sc_X = StandardScaler()


if __name__ == "__main__":
    data = pd.read_csv(root_dir + "BankChurners.csv")

    data = data.drop(labels=["CLIENTNUM",
                             "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
                             "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)

    data = preprocess(data, "Attrition_Flag", "Gender", "Education_Level", "Marital_Status", "Income_Category",
                     "Card_Category")

    X, X_test, y, y_test = train_test_split(data.iloc[:, 1:].copy().values,
                                            data.iloc[:, 0].copy().values,
                                            test_size=0.2,
                                            shuffle=True,
                                            random_state=0)

    X = sc_X.fit_transform(X)
    X_test = sc_X.transform(X_test)

    accs = []
    pres = []
    recs = []

    breaker()
    acc, pre, rec = esk.evalKNC(X_train=X, X_valid=X_test, y_train=y, y_valid=y_test)
    print("KNC Accuracy  : {:.5f}".format(acc))
    print("KNC Precision : {:.5f}".format(pre))
    print("KNC Recall    : {:.5f}".format(rec))
    accs.append(acc)
    pres.append(pre)
    recs.append(rec)

    breaker()
    acc, pre, rec = esk.evalSVC(X_train=X, X_valid=X_test, y_train=y, y_valid=y_test)
    print("SVC Accuracy  : {:.5f}".format(acc))
    print("SVC Precision : {:.5f}".format(pre))
    print("SVC Recall    : {:.5f}".format(rec))
    accs.append(acc)
    pres.append(pre)
    recs.append(rec)

    breaker()
    acc, pre, rec = esk.evalGNB(X_train=X, X_valid=X_test, y_train=y, y_valid=y_test)
    print("GNB Accuracy  : {:.5f}".format(acc))
    print("GNB Precision : {:.5f}".format(pre))
    print("GNB Recall    : {:.5f}".format(rec))
    accs.append(acc)
    pres.append(pre)
    recs.append(rec)

    breaker()
    acc, pre, rec = esk.evalDTC(X_train=X, X_valid=X_test, y_train=y, y_valid=y_test)
    print("DTC Accuracy  : {:.5f}".format(acc))
    print("DTC Precision : {:.5f}".format(pre))
    print("DTC Recall    : {:.5f}".format(rec))
    accs.append(acc)
    pres.append(pre)
    recs.append(rec)

    breaker()
    acc, pre, rec = esk.evalRFC(X_train=X, X_valid=X_test, y_train=y, y_valid=y_test)
    print("RFC Accuracy  : {:.5f}".format(acc))
    print("RFC Precision : {:.5f}".format(pre))
    print("RFC Recall    : {:.5f}".format(rec))
    accs.append(acc)
    pres.append(pre)
    recs.append(rec)

    breaker()
    acc, pre, rec = esk.evalXGC(X_train=X, X_valid=X_test, y_train=y, y_valid=y_test)
    print("XGC Accuracy  : {:.5f}".format(acc))
    print("XGC Precision : {:.5f}".format(pre))
    print("XGC Recall    : {:.5f}".format(rec))
    accs.append(acc)
    pres.append(pre)
    recs.append(rec)

    cfg = CFG(IL=X.shape[1], HL=[128, 64], epochs=50)

    tr_data_setup = Dataset(X, y.reshape(-1, 1))
    tr_data = DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=True, generator=torch.manual_seed(0))

    ts_data_setup = Dataset(X_test, y_test.reshape(-1, 1))
    ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

    torch.manual_seed(0)
    Model = ANN(IL=cfg.IL, HL=cfg.HL, OL=cfg.OL, use_dp=cfg.use_dp)
    optimizer = Model.getOptimizer(lr=1e-3, wd=0)

    Losses, Accuracies = fp.fit_sm(model=Model, optimizer=optimizer, epochs=cfg.epochs,
                                   trainloader=tr_data, validloader=ts_data,
                                   criterion=nn.BCEWithLogitsLoss(), device=cfg.device,
                                   verbose=True)

    TRL = []
    TVL = []
    TRA = []
    TVA = []

    for i in range(len(Losses)):
        TRL.append(Losses[i]["train"])
        TVL.append(Losses[i]["valid"])
        TRA.append(Accuracies[i]["train"])
        TVA.append(Accuracies[i]["valid"])

    plt.figure()
    plt.plot([i + 1 for i in range(len(TRL))], TRL, "r", label="Training Loss")
    plt.plot([i + 1 for i in range(len(TVL))], TVL, "b--", label="Validation Loss")
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show(block=False)
    plt.pause(2.5)
    plt.close()

    plt.figure()
    plt.plot([i + 1 for i in range(len(TRA))], TRA, "r", label="Training Accuracy")
    plt.plot([i + 1 for i in range(len(TVA))], TVA, "b--", label="Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show(block=False)
    plt.pause(2.5)
    plt.close()

    y_pred = fp.predict(model=Model, dataloader=ts_data, batch_size=cfg.ts_batch_size, device=cfg.device)
    print("ANN Accuracy  : {:.5f}".format(accuracy_score(y_test, y_pred)))
    print("ANN Precision : {:.5f}".format(precision_score(y_test, y_pred, average="weighted")))
    print("ANN Recall    : {:.5f}".format(recall_score(y_test, y_pred, average="weighted")))
    accs.append(accuracy_score(y_test, y_pred))
    pres.append(precision_score(y_test, y_pred, average="weighted"))
    recs.append(recall_score(y_test, y_pred, average="weighted"))

    labels = ["KNC", "SVC", "GNB", "DTC", "RFC", "XGC", "ANN"]

    resolution = 0.1
    plt.figure(figsize=(8, 6))
    plt.bar(x=labels, height=accs, width=0.5)
    plt.ylabel("Accuracy")
    plt.ylim([min(accs) - resolution, 1])
    plt.title("Model Accuracies")
    plt.show(block=False)
    plt.pause(5)
    plt.close()
