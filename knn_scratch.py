import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # only to change the working directory

def calculateDistance(dataSet, query_point, type="euclidean"):
    """
    Function calculates specified distance between rows of dataSet(dataFrame)
    and query_point(single row).
    distance returned from this function: Euclidean
    and Manhattan, depends upon the argument value passed.

    distance is returned as dictionary

    """
    if type == "euclidean":
        d = {}
        #diff = np.array((dataSet - query_point)**2)
        # d = dict(enumerate(np.sqrt(np.sum(diff, axis=1)).flatten(), 0))
        
        for i in range(len(dataSet)):
            d[i] = round((abs(sum((dataSet.iloc[i, 0:11])**2 - query_point**2)))**0.5, 2)
        return d
    elif type == "manhattan":
        diff = np.array(np.abs(dataSet-query_point))
        return dict(enumerate(np.sum(diff, axis=1).flatten(), 0))
    else:
        raise Exception("distance argument not correct")
    # return d

# Get Feature set of data
def getFeatureSet(data_frame):
    return data_frame[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides","free sulfur dioxide",
                       "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
    #hard coded value, can be changed accordingly.

# Get Label set of data
def getLabelSet(data_frame):
    return data_frame["Quality"]


def MinMaxScl(data):
    """
    returns normalized data. Data Normalized on Min Max normalization logic
    """
    n_data = ((data-data.min())/(data.max()-data.min()))*(1-0)+0
    return n_data


def ZscoreScl(data):
    """
    returns normalized data. Data Normalized on z-score logic
    """
    n_data = (data - data.mean())/data.std(ddof=0)
    return n_data


def knn(k, scaling="", n=0, dis_type="euclidean", df):
    """
    finds the most relevant class based on KNN logic.
    scaling and dis_type can be set as required in the argument.
    dis_type has default value "euclidean".
    
    n is of no use.
    
    precaution: if K is even, then it may be case that the given point is at 
    same distance from from the classes and it becomes impossible to predict
    the correct class, so, K must be odd.
    """
    df_train = pd.read_csv(df)
    df_test = pd.read_csv(df)
    # splitting and shuffling data into test and train 70:30 ratio
    # itrain = random.sample(range(1,len(df_train)), round(len(df_train)*0.7))
    # d_train = df_train.iloc[itrain,]
    # d_train = df_train[:round(len(df_train)*0.7)]
    # d_test = df_train[round(len(df_train)*0.7):]
    test_pred = []  # store predicted class

    df_train_label = getLabelSet(df_train)
    df_test_label = getLabelSet(df_test)

    if scaling == "ZscoreScl":
        df_train_n = ZscoreScl(getFeatureSet(df_train))
        df_test_n = ZscoreScl(getFeatureSet(df_test))

    elif scaling == "MinMaxScl":
        df_train_n = MinMaxScl(getFeatureSet(df_train))
        df_test_n = MinMaxScl(getFeatureSet(df_test))

    else:
        df_train_n = getFeatureSet(df_train)
        df_test_n = getFeatureSet(df_test)

    for j in range(len(df_test_n)):
        dis = calculateDistance(df_train_n, df_test_n.iloc[j], type=dis_type)
        ind = np.array(sorted(dis, key=dis.get)[:k])  # array of index with lowest distance
        list_knn = list(df_train_label.iloc[ind])  # the list of k-nn output
        test_pred.append(max(list_knn, key = list_knn.count)) 

    acc = (sum([test_pred == df_test_label][0]))/len(df_test)
    return round(acc, 2)


def weighted_knn(k, n=1, scaling="", dis_type="euclidean", df):

    """
    the classes have now weights which decreases with increase in distance.
    The scale of weights can be defined with n.

    precaution: if K is even, then it may be case that the given point is at 
    same distance from from the classes and it becomes impossible to predict
    the correct class, so, K must be odd.

    References: 
    Logic for weight = 1 at line 153 and 159 are taken from:
    https://stats.stackexchange.com/questions/378677/how-does-scikit-learns-knn-model-handle-zero-distances-when-using-inverse-dista
    """
    df_train = pd.read_csv(df)
    df_test  = pd.read_csv(df)

    test_pred = []  # store predicted class
    
    df_train_label = getLabelSet(df_train)
    df_test_label = getLabelSet(df_test)

    if scaling == "ZscoreScl":
        df_train_n = ZscoreScl(getFeatureSet(df_train))
        df_test_n = ZscoreScl(getFeatureSet(df_test))
    
    elif scaling == "MinMaxScl":
        df_train_n = MinMaxScl(getFeatureSet(df_train))
        df_test_n = MinMaxScl(getFeatureSet(df_test))

    else:
        df_train_n = getFeatureSet(df_train)
        df_test_n = getFeatureSet(df_test)

    for j in range(len(df_test_n)):
        dis = calculateDistance(df_train_n, df_test_n.iloc[j], type = dis_type)
        sorted_dis = {}
        sorted_dis = sorted(dis, key=dis.get)  # index sorted with distance as key in ascending order
        ind = np.array(sorted(dis, key=dis.get)[:k])  # selecting top k indexes

        class1 = 0
        class2 = 0

        for i in ind:
            if df_train_label.iloc[i] == -1:
                if dis[i] == 0:
                    class1 += 1  # if distance is 0, add weight as 1.
                    continue
                class1 += 1/(dis[i]**n)

            elif df_train_label.iloc[i] == 1:
                if dis[i] == 0:
                    class2 += 1  # if distance is 0, add weight as 1.
                    continue
                class2 += 1/(dis[i]**n)

        if class1 > class2:
            test_pred.append(-1)
        else:
            test_pred.append(1)

    acc = (sum([test_pred == df_test_label][0]))/len(df_test)
    
    
    return round(acc,2)

def main():
    df = read.csv('dataset1.csv')
    allResults = []
    for k in range(3, 45, 2):
        accuracy = weighted_knn(k, scaling="MinMaxScl", n=2, df)
        allResults.append(accuracy)
    sns.set_style("darkgrid")
    plt.plot(list(range(3, 45, 2)), allResults, marker = "D", color = 'indianred' \
                , markeredgecolor='black', markerfacecolor='white')
    plt.show()

main()

