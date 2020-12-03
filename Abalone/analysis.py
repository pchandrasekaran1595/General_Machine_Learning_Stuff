import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import LabelEncoder

def breaker():
    print("\n" + 50*"-" + "\n")


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


root_dir = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Abalone/"
le = LabelEncoder()

if __name__ == "__main__":
    data = pd.read_csv(root_dir + "abalone.csv")

    columns = ["Sex", "Length", "Diameter", "Height", "Whole_Weight", "Shucked_Weight", "Viscera_Weight",
               "Shell_Weight", "Rings"]
    data.columns = columns

    breaker()
    print(data.head(5))
    breaker()
    print(data.shape)
    breaker()

    """plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="Rings")
    plt.title("Class Distribution")
    sns.displot(data=data["Rings"], kind="kde")
    plt.title("Class Distribution")
    plt.show()"""

    numpy_data = data.copy().values

    numpy_data[:, 0] = le.fit_transform(numpy_data[:, 0])
    numpy_data[:, -1] = np.subtract(numpy_data[:, -1], 1)

    split_value = 3133
    X = numpy_data[:split_value, :-1]
    X_test = numpy_data[split_value:, :-1]
    y = numpy_data[:split_value, -1]
    y_test = numpy_data[split_value:, -1]

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
    for i in range(X[:, :-1].shape[1]):
        print("IDV {} Corcoeff : {}".format(i+1, calc_sample_cov(X[:, -1], X[:, i]) /
                                            (math.sqrt(calc_sample_var(X[:, -1].reshape(1, -1))) * math.sqrt(
                                                calc_sample_var(X[:, i].reshape(1, -1))))))
    breaker()

    print("Test Set\n")
    for i in range(X_test[:, :-1].shape[1]):
        print("IDV {} Corcoeff : {}".format(i + 1, calc_sample_cov(X_test[:, -1], X_test[:, i]) /
                                            (math.sqrt(calc_sample_var(X_test[:, -1].reshape(1, -1))) * math.sqrt(
                                                calc_sample_var(X_test[:, i].reshape(1, -1))))))
    breaker()


