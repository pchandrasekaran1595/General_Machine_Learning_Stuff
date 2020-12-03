import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

import gc

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def breaker():
    print("\n" + 50*"-" + "\n")


def head(x=None, no_of_ele=5):
    breaker()
    print(x[:no_of_ele])
    breaker()


def calc_sample_var(x=None, calc_type="full"):
    m = x.shape[0]
    n = x.shape[1]

    if calc_type == "full":
        final_sum = 0
        for i in range(m):
            mu_x = np.mean(x[i, :])
            sum_ = 0
            for j in range(n):
                sum_ += (x[i, j] - mu_x) ** 2
            final_sum += (sum_ / (n - 1))
        return final_sum / m
    elif calc_type == "row":
        final = []
        for i in range(m):
            mu_x = np.mean(x[i, :])
            sum_ = 0
            for j in range(n):
                sum_ += (x[i, j] - mu_x) ** 2
            final.append(sum_ / (n-1))
        return final
    else:
        raise ValueError("calc_type accepts only 'row' or 'full'")


def calc_sample_cov(x1=None, x2=None):
    mu_x = np.mean(x1)
    mu_y = np.mean(x2)

    n = x1.shape[0]  # x2.shape[0]

    cov = 0
    for i in range(n):
        cov = cov + (x1[i] - mu_x) * (x2[i] - mu_y)
    return cov/(n-1)


root_dir = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Letter Recognition/"

if __name__ == "__main__":
    data = pd.read_csv(root_dir + "data.csv", header=None)

    columns = ["Letter", "X_Box", "Y_Box", "Width", "Height", "Onpix",
               "Mean(X)", "Mean(Y)", "Mean(X_Var)", "Mean(Y_Bar)", "XY_Correlation",
               "Mean(X^2**Y)", "Mean(X*Y^2)", "Mean(Edge Count L->R)[X_ege]",
               "(X_ege)Y_Correlation", "Mean(Edge Count B->T)[Y_ege]",
               "X(Y_ege)_Correlation"
               ]

    data.columns = columns

    breaker()
    print(data.head(5))
    breaker()
    print(data.shape)
    breaker()

    """sns.displot(data=data["Letter"], kind="hist", height=4)
    plt.title("Distribution of Labels")
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    Distribution Seems Almost Normal (Balanced Classes)"""

    numpy_data = data.copy().values

    le = LabelEncoder()
    numpy_data[:, 0] = le.fit_transform(numpy_data[:, 0])

    X, X_test, y, y_test = train_test_split(numpy_data[:, 1:],
                                            numpy_data[:, 0],
                                            test_size=0.2,
                                            shuffle=True,
                                            random_state=0,
                                            stratify=numpy_data[:, 0])

    print("Complete Dataset Mean     : {}".format(numpy_data.mean()))
    print("Complete Dataset Variance : {}".format(numpy_data.var()))
    print("Complete Dataset Std      : {}".format(math.sqrt(numpy_data.var())))
    breaker()

    print("Training Set Mean     : {}".format(X.mean()))
    print("Training Set Variance : {}".format(calc_sample_var(X)))
    print("Training Set Std      : {}".format(math.sqrt(calc_sample_var(X))))
    breaker()

    print("Test Set Mean     : {}".format(X_test.mean()))
    print("Test Set Variance : {}".format(calc_sample_var(X_test)))
    print("Test Set Std      : {}".format(math.sqrt(calc_sample_var(X_test))))
    breaker()

    print("Training Set\n")
    for i in range(1, X[:, 1:].shape[1]):
        print("IDV {} Corcoeff : {}".format(i, calc_sample_cov(X[:, 0], X[:, i]) /
             (math.sqrt(calc_sample_var(X[:, 0].reshape(1, -1))) * math.sqrt(calc_sample_var(X[:, i].reshape(1, -1))))))
    breaker()
