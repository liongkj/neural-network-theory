    #!/usr/bin/env python
# coding: utf-8

# Solving the three-class classification problem in the given dataset using SVM classifiers and
# the one-vs-rest strategy. SVM classifiers are provided in LibSVM package and other machine
# learning libraries (sklearn). You can use these libraries to solve this problem.
# Notice: the SVM provided in these third-party modules can handle multi-class classification.
# However, you are required to write the one-vs-rest strategy by yourself in this assignment

# In[23]:


import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import preprocessing 

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import numpy as np
from IPython import display
import time
import seaborn as sns
import pandas as pd


from res.plot_lib import plot_data,plot_model, set_default, plot_submodel
from res.util import one_vs_rest, load_data
set_default(figsize=(4, 4))
seed = 222
random.seed(seed)


# In[3]:


x_train,y_train,n_class =load_data(test=False,shuffle_data=True,seed=seed)
x_test,y_test,_ = load_data(test=True,seed=seed)
label_class = np.unique(y_test,return_counts=True)

# scaler = StandardScaler().fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# minMax = MinMaxScaler().fit(x_train)
# x_train = minMax.transform(x_train)
# x_test = minMax.transform(x_test)

if __name__=="__main__":
    pca = PCA(n_components=0.85).fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    maxAbs = preprocessing.MaxAbsScaler().fit(x_train)
    x_train = maxAbs.transform(x_train)
    x_test = maxAbs.transform(x_test)




    all_data = pd.DataFrame({
        'category':label_class[0],
        'n':label_class[1]
        })
    plot = sns.barplot(data=all_data,x='category',y='n')#,x='category',y = 'n',discrete=True,multiple='dodge')
    fig = plot.get_figure()
    # fig.savefig('./Report/img/'+"class_distribution.png")



    k_class_prob = one_vs_rest(x_train,y_train)
    plot_data = pd.DataFrame({
        'category':[1,2,3,1,2,3],
        'count':k_class_prob[-2]+k_class_prob[-1],
        'class':["+","+","+","-","-","-"]
        })
    plt.figure(figsize=(8, 4))
    plot = sns.barplot(x="category",y="count",hue="class",data=plot_data).set_title("One vs Rest")
    fig = plot.get_figure()
    # fig.savefig('./Report/img/'+"OneVRest_count.png")


    sub_svm = list()
    x,y = k_class_prob[0],k_class_prob[1]
    start = time.time()
    for i in range(len(x)):
        print("svm ",i)
        train_x,dev_x, train_y,dev_y = train_test_split(x[i],y[i], random_state=seed,train_size=0.8,test_size=0.2)
    #     clf = SVC(kernel='linear',gamma=0.01,cache_size = 1000,probability=False)
        clf = LinearSVC(C=100,dual=False)
        clf.fit(train_x,train_y)
        prediction = clf.predict(dev_x)
        df = clf.decision_function(x_test)
        print(classification_report(dev_y, prediction))
        sub_svm.append(df)
    end = time.time()
    print("Time elapse:",end - start)


    d_shape = np.array(sub_svm)
    predicted = label_class[0][np.argmax(d_shape,0)]
    # acc = (y_test == predicted).sum() / len(y_test) *100
    print(classification_report(y_test, predicted))