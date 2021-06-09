

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing 

from sklearn import svm
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA

from IPython import display
import time

import pandas as pd
import math
import numpy as np



from res.util import load_data ,start_train,my_concat,make_prediction,get_partition_size#part_vs_part_random
seed = 12345
np.random.seed(seed)
def ratio(a, b):    
    return float(a/b)

def one_vs_rest(train_x, train_y):
    '''
        Divide the train_x into three parts based on the train_y,
        Three parts can be used to apply one vs rest or other methods.
    '''
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    for i in range(train_x.shape[0]):
        if (train_y[i] == 1):
            x1.append(train_x[i])
            y1.append(train_y[i])
        elif (train_y[i] == 0):
            x2.append(train_x[i])
            y2.append(train_y[i])
        elif (train_y[i] == -1):
            x3.append(train_x[i])
            y3.append(train_y[i])
    return np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(x3), np.array(y3)



class m3_model:
    '''
        Min-Max Module.
    '''

    def __init__(self, train_x1, train_y1, train_x2, train_y2):
        self.model_1 = start_train(train_x1[0:3000,], train_y1[0:3000,], train_x2[0:3000], train_y2[0:3000,])
        self.model_2 = start_train(train_x1[3000:6000,], train_y1[3000:6000,], train_x2[3000:6000], train_y2[3000:6000,])
        self.model_3 = start_train(train_x1[6000:9000,], train_y1[6000:9000,], train_x2[6000:9000], train_y2[6000:9000,])
        self.model_4 = start_train(train_x1[9000:,], train_y1[9000:,], train_x2[9000:], train_y2[9000:,])
        
    def predict(self, test_data):
        '''
            use Min-Max Module to figure out problem.
        '''
        pred1 = self.model_1.predict(test_data)
        pred2 = self.model_2.predict(test_data)
        pred3 = self.model_3.predict(test_data)
        pred4 = self.model_4.predict(test_data)
        predictions = []
        for i in range(pred1.shape[0]):
            min_pred1 = min(pred1[i], pred2[i])
            min_pred2 = min(pred3[i], pred4[i])
            predictions.append(max(min_pred1, min_pred2))
        return np.array(predictions)

# Solving the three-class classification problem using Min-Max-Module SVM and part-vs-part
# task decomposition method. You should divide the three-class problem into three two-class
# problems using one-vs-rest method and then decompose these imbalance two-class problems
# into balance two-class problems following random task decomposition and task
# decomposition with prior knowledge strategies. Please compare the performance of SVMs
# obtained in Problem One and the Min-Max-Module SVMs here.




if __name__=="__main__":

    x_train,y_train,_ = load_data(test=False)
    x_test,y_test,_ = load_data(test=True,seed=seed)

    label_class = np.unique(y_train,return_counts=True)
    print("xshape: ",x_train.shape)

    pca = PCA(n_components=0.85).fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print("PCA Processing")

    maxAbs = preprocessing.MaxAbsScaler().fit(x_train)
    x_train = maxAbs.transform(x_train)
    x_test = maxAbs.transform(x_test)
    print("MaxAbs Processing")
    
    train_x1, train_y1, train_x2, train_y2, train_x3, train_y3 = one_vs_rest(x_train, y_train)

    start = time.time()
    clf_1_2 = m3_model(train_x1, train_y1, train_x2, train_y2) #classifier 1 2
    clf_1_3 = m3_model(train_x1, train_y1, train_x3, train_y3) #classifier 1 3
    clf_2_3 = m3_model(train_x2, train_y2, train_x3, train_y3) #classifier 2 3
    end = time.time()
    print("Time elapse:",end - start)
    prediction = make_prediction([clf_1_2, clf_1_3, clf_2_3], x_test)
    


    print(classification_report(y_test, prediction))