from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score

def evalKNC(n_neighbors=5, X_train=None, X_valid=None, y_train=None, y_valid=None):
    knc = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    y_pred = knc.predict(X_valid)

    return accuracy_score(y_valid, y_pred), precision_score(y_valid, y_pred, average="weighted"), recall_score(y_valid, y_pred, average="weighted")

def evalSVC(X_train=None, X_valid=None, y_train=None, y_valid=None):
    svc = SVC(kernel="rbf").fit(X_train, y_train)
    y_pred = svc.predict(X_valid)

    return accuracy_score(y_valid, y_pred), precision_score(y_valid, y_pred, average="weighted"), recall_score(y_valid, y_pred, average="weighted")

def evalGNB(X_train=None, X_valid=None, y_train=None, y_valid=None):
    gnb = GaussianNB().fit(X_train, y_train)
    y_pred = gnb.predict(X_valid)

    return accuracy_score(y_valid, y_pred), precision_score(y_valid, y_pred, average="weighted"), recall_score(y_valid, y_pred, average="weighted")

def evalDTC(X_train=None, X_valid=None, y_train=None, y_valid=None):
    dtc = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    y_pred = dtc.predict(X_valid)

    return accuracy_score(y_valid, y_pred), precision_score(y_valid, y_pred, average="weighted"), recall_score(y_valid, y_pred, average="weighted")

def evalRFC(n_estimators=100, X_train=None, X_valid=None, y_train=None, y_valid=None):
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=0).fit(X_train, y_train)
    y_pred = rfc.predict(X_valid)

    return accuracy_score(y_valid, y_pred), precision_score(y_valid, y_pred, average="weighted"), recall_score(y_valid, y_pred, average="weighted")

def evalXGC(n_estimators=100, learning_rate=0.1, X_train=None, X_valid=None, y_train=None, y_valid=None):
    xgc = XGBClassifier(n_estimators=100, learning_rate=learning_rate, random_state=0).fit(X_train, y_train)
    y_pred = xgc.predict(X_valid)

    return accuracy_score(y_valid, y_pred), precision_score(y_valid, y_pred, average="weighted"), recall_score(y_valid, y_pred, average="weighted")
