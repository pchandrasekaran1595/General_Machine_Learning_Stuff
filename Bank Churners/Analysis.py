import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def breaker():
    print("\n" + 50*"-" + "\n")


def getCol(x=None):
    return [col for col in x.columns]


root_dir = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Bank Churners/"

if __name__ == "__main__":
    data = pd.read_csv(root_dir + "BankChurners.csv")

    # for name in getCol(data):
    #    print(name)

    data = data.drop(labels=["CLIENTNUM",
                             "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
                             "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)

    """breaker()
    print(data.head(5))
    breaker()
    print(data.shape)
    breaker()

    for name in getCol(data):
        print(name)"""

    ########## Count Plots ##########
    """plt.figure()
    sns.countplot(data=data, x="Attrition_Flag")
    
    plt.figure()
    sns.countplot(data=data, x="Gender")

    plt.figure()
    sns.countplot(data=data, x="Education_Level")

    plt.figure()
    sns.countplot(data=data, x="Marital_Status")

    plt.figure()
    sns.countplot(data=data, x="Income_Category")

    plt.figure()
    sns.countplot(data=data, x="Card_Category")
    
    plt.show()"""

    """########## Distributions ##########
    sns.histplot(data=data, x="Customer_Age", kde=True)
    plt.show()

    sns.histplot(data=data, x="Dependent_count", kde=True, bins=6)
    plt.show()

    sns.histplot(data=data, x="Months_on_book", kde=True)
    plt.show()

    sns.histplot(data=data, x="Total_Relationship_Count", kde=True, bins=6)
    plt.show()

    sns.histplot(data=data, x="Months_Inactive_12_mon", kde=True, bins=7)
    plt.show()

    sns.histplot(data=data, x="Contacts_Count_12_mon", kde=True, bins=7)
    plt.show()"""

    """########## Distributions ##########
    sns.histplot(data=data, x="Credit_Limit", kde=True)
    plt.show()

    sns.histplot(data=data, x="Total_Revolving_Bal", kde=True)
    plt.show()

    sns.histplot(data=data, x="Avg_Open_To_Buy", kde=True)
    plt.show()

    sns.histplot(data=data, x="Total_Amt_Chng_Q4_Q1", kde=True)
    plt.show()

    sns.histplot(data=data, x="Total_Trans_Amt", kde=True)
    plt.show()

    sns.histplot(data=data, x="Total_Ct_Chng_Q4_Q1", kde=True)
    plt.show()

    sns.histplot(data=data, x="Total_Trans_Ct", kde=True)
    plt.show()

    sns.histplot(data=data, x="Avg_Utilization_Ratio", kde=True)
    plt.show()"""

    """########## Bayesian Inference (Age) ##########
    #sns.histplot(data=data, x="Customer_Age", kde=True)
    #plt.show()

    N = data.shape[0]

    existing_customers = data[data["Attrition_Flag"] == "Existing Customer"]
    attrited_customers = data[data["Attrition_Flag"] == "Attrited Customer"]

    age = 40
    age_above_ = data[data["Customer_Age"] > age]

    breaker()
    print("P(Age > {}) : {:.5f}".format(age, age_above_.shape[0] / N))
    breaker()
    print("P(Existing Customer) : {:.5f}".format(existing_customers.shape[0] / N))
    breaker()
    print("P(Attrited Customer) : {:.5f}".format(attrited_customers.shape[0] / N))

    age_above_and_existing = data[data["Attrition_Flag"] == "Existing Customer"]
    age_above_and_existing = age_above_and_existing[age_above_and_existing["Customer_Age"] > age]

    age_above_and_attrited = data[data["Attrition_Flag"] == "Attrited Customer"]
    age_above_and_attrited = age_above_and_attrited[age_above_and_attrited["Customer_Age"] > age]

    breaker()
    print("P(Age > {} and Existing) : {:.5f}".format(age, age_above_and_existing.shape[0] / N))
    breaker()
    print("P(Age > {} and Attrited) : {:.5f}".format(age, age_above_and_attrited.shape[0] / N))

    breaker()
    print("P(Age > {} / Existing) : {:.5f}".format(age, age_above_and_existing.shape[0] / existing_customers.shape[0]))
    breaker()
    print("P(Age > {} / Attrited) : {:.5f}".format(age, age_above_and_attrited.shape[0] / attrited_customers.shape[0]))

    breaker()
    print("P(Existing / Age > {}) : {:.5f}".format(age, (age_above_and_existing.shape[0] / existing_customers.shape[0] *
                                                          existing_customers.shape[0]) / age_above_.shape[0]))
    breaker()
    print("P(Attrited / Age > {}) : {:.5f}".format(age, (age_above_and_attrited.shape[0] / attrited_customers.shape[0] *
                                                         attrited_customers.shape[0]) / age_above_.shape[0]))"""

    """########## Bayesian Inference (Credit Limit) ##########
    # sns.histplot(data=data, x="Credit_Limit", kde=True)
    # plt.show()

    N = data.shape[0]

    existing_customers = data[data["Attrition_Flag"] == "Existing Customer"]
    attrited_customers = data[data["Attrition_Flag"] == "Attrited Customer"]

    credit_limit = 10000
    credit_limit_above_ = data[data["Credit_Limit"] > credit_limit]

    breaker()
    print("P(Credit Limit > {}) : {:.5f}".format(credit_limit, credit_limit_above_.shape[0] / N))
    breaker()
    print("P(Existing Customer) : {:.5f}".format(existing_customers.shape[0] / N))
    breaker()
    print("P(Attrited Customer) : {:.5f}".format(attrited_customers.shape[0] / N))

    credit_limit_above_and_existing = data[data["Attrition_Flag"] == "Existing Customer"]
    credit_limit_above_and_existing = credit_limit_above_and_existing[credit_limit_above_and_existing["Credit_Limit"] > credit_limit]

    credit_limit_above_and_attrited = data[data["Attrition_Flag"] == "Attrited Customer"]
    credit_limit_above_and_attrited = credit_limit_above_and_attrited[credit_limit_above_and_attrited["Credit_Limit"] > credit_limit]

    breaker()
    print("P(Credit Limit > {} and Existing) : {:.5f}".format(credit_limit, credit_limit_above_and_existing.shape[0] / N))
    breaker()
    print("P(Credit Limit > {} and Attrited) : {:.5f}".format(credit_limit, credit_limit_above_and_attrited.shape[0] / N))

    breaker()
    print("P(Credit Limit > {} / Existing) : {:.5f}".format(credit_limit, credit_limit_above_and_existing.shape[0] / existing_customers.shape[0]))
    breaker()
    print("P(Credit Limit > {} / Attrited) : {:.5f}".format(credit_limit, credit_limit_above_and_attrited.shape[0] / attrited_customers.shape[0]))

    breaker()
    print("P(Existing / Credit Limit > {}) : {:.5f}".format(credit_limit, (credit_limit_above_and_existing.shape[0] / existing_customers.shape[0] *
                                                         existing_customers.shape[0]) / credit_limit_above_.shape[0]))
    breaker()
    print("P(Attrited / Credit Limit > {}) : {:.5f}".format(credit_limit, (credit_limit_above_and_attrited.shape[0] / attrited_customers.shape[0] *
                                                         attrited_customers.shape[0]) / credit_limit_above_.shape[0]))
    breaker()"""
