import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
random.seed(30)

def custom_graph_cluster(data, x, y, cluster):
  plot = plt.scatter(data[x],data[y], c = data[cluster])
  matplotlib.pyplot.show()

def custom_xgb(train_data, train_target, test_data):
    xg = xgb.XGBClassifier()
    xg_class = xg.fit(train_data, train_target)
    y_pred = xg_class.predict(test_data)
    return(y_pred)

def custom_tree(train_data, train_target, test_data):
    clf = DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)
    y_pred = clf.predict(test_data)
    return(y_pred)

#read in wine
wine = pd.read_csv('wine.csv')
wine['Class'] = np.array(wine['Class']) - 1

#saves raw labels
raw_labels = wine['Class']
wine = wine.drop('Class',axis=1)

seed = 30
# generate kmeans labels
kmeans = KMeans(3,random_state=seed).fit(wine)
kmeans_labels = kmeans.labels_

#generate random labels
random_labels = [random.randrange(0, 3, 1) for i in range(178)]

kmeans_size = 11
num_for_avg = 10

average_kmeans_dt = []
average_kmeans_xgb = []

#train_test_split on kmeans labels
for i in range(kmeans_size):
    kmeans = KMeans((i+1), random_state=seed).fit(wine)
    kmeans_labels = kmeans.labels_

    for x in range(num_for_avg):
        X_train, X_test, y_train, y_test = train_test_split(wine, kmeans_labels, test_size=0.25, random_state=x, stratify=kmeans_labels)

        #print('\033[1m' + "random state for test train split of kmeans labels is " + str(x) + '\033[0m')

        #tests decision tree with split
        test_acc = custom_tree(X_train,y_train,X_test)
        #print("test/train split of kmeans label(decision tree)")
        #print(accuracy_score(y_test,test_acc))
        average_kmeans_dt.append(accuracy_score(y_test,test_acc))

        #tests xgb with split
        test_acc = custom_xgb(X_train,y_train,X_test)
        #print("test/train split of kmeans label(xgb)")
        #print(accuracy_score(y_test,test_acc))
        average_kmeans_xgb.append(accuracy_score(y_test,test_acc))

    #makes graph of kmeans labels
    print("kmeans label")
    wine['kmeans labels'] = kmeans_labels
    kmeans_graph = custom_graph_cluster(wine, 'Alcohol','Malic acid','kmeans labels')

    print('kmeans labels at ' + str(i + 1))
    #print(pd.crosstab(raw_labels, kmeans_labels))


    print("The average of kmeans label dt accuracy is " + str(np.average(average_kmeans_dt)))
    print("The average of kmeans label xgb accuracy is " + str(np.average(average_kmeans_xgb)))

    #drops the k means labels from the data
    tree = wine.drop('kmeans labels',axis=1)
    if i > 0:
        print("silhoutette score")
        print(silhouette_score(wine, kmeans_labels))
