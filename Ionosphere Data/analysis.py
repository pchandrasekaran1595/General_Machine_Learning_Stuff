import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def breaker():
    print("\n" + 50 * "-" + "\n")

def preprocess(x=None, *args):
    df = x.copy()
    df[args[0]] = df[args[0]].map({"b" : 0, "g" : 1})
    return df


root_dir = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Ionosphere Data/"


data = pd.read_csv(root_dir + "ionosphere.csv")

columns = [str(i+1) for i in range(data.shape[1])]

data.columns = columns
data = preprocess(data, columns[-1])

"""plt.figure(figsize=(6, 4))
sns.countplot(data=data, x=columns[-1])
plt.show(block=False)
plt.pause(2)
plt.close()

breaker()
print("Class 0 Count : {}".format(data[data["35"] == 0].shape[0]))
breaker()
print("Class 1 Count : {}".format(data[data["35"] == 1].shape[0]))
breaker()"""

"""########## Distribution Checks ##########
key = "24"
plt.figure(figsize=(6, 4))
sns.histplot(data=data, x=key, hue="35")
plt.show(block=False)
plt.pause(3)
plt.close()"""

breaker()
print(data.iloc[:, :-1].mean().mean())
breaker()
print(data.iloc[:, :-1].var().var())
breaker()