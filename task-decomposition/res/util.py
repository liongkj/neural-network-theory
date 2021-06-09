import copy
import numpy as np
from sklearn.utils import shuffle
import torch
import math
from sklearn.svm import LinearSVC
import random

def load_data(test=False,shuffle_data=True, seed = 12345):
    if(test):
        x = np.load('./Data/test_data.npy')#[:n,:]
        y = np.load('./Data/test_label.npy')#[:n]
    else:
        x = np.load('./Data/train_data.npy')#[:n,:]
        y = np.load('./Data/train_label.npy')#[:n]
    label_class = np.unique(y)
    n_class = len(label_class)
    # print('Class label counts:', n_class)
    y = y.astype(np.integer)
    if (shuffle_data):
        x,y = shuffle(x,y,random_state = seed)
    return x,y,n_class

def one_vs_rest(x,y):
    class_x = list()
    class_y = list()
    pos_ = list()
    neg_ = list()
    label_class = np.unique(y)
    n_class = len(label_class)
    # y=y.cpu()
    for i in range(n_class):
#         x_tmp = np.zeros((x.shape[0],x.shape[1]+1))
        y_tmp = np.zeros(y.shape)
        # print("class ",n_class[i])
        pos_idx = np.where(y==label_class[i])
        neg_idx =np.where(y!=label_class[i])
        y_tmp[pos_idx] = 1
        y_tmp[neg_idx] = -1
        
        pos_.append(len(pos_idx[0]))
        neg_.append(len(neg_idx[0]))
        class_x.append(x)
        class_y.append(y_tmp)
    return class_x,class_y, pos_,neg_
#     for d in

def get_partition_size(x, rho = 10000, gamma = 0.8):
    n_samples = x
    total_samples = n_samples * 2
    part_samples = (total_samples) / rho
    if(total_samples > rho and math.fmod(total_samples,rho) <= gamma):
        return math.floor(part_samples)
    return math.ceil(part_samples)


def make_prediction(model_list, test_y):
    '''
        Make prediction based on the results of sub svm prediction
    '''
    result_ = []
    for model in model_list:
        result_.append(model.predict(test_y).reshape(-1,1))
    res = np.concatenate(result_, axis = 1)
    predictions = []
    for i in range(res.shape[0]):
        nums = res[i]
        if (sum(nums==-1)>=2):
            predictions.append(-1)
        elif (sum(nums==0)>=2):
            predictions.append(0)
        elif (sum(nums==1)>=2):
            predictions.append(1)
        else:
            predictions.append(random.randint(-1,1))
        #print(predictions[-1])
    predictions = np.array(predictions)
    return predictions

def my_concat(train_x1, train_y1, train_x2, train_y2):
    '''
        Combine the two dataset into one.
        Use this to make a rest dataset in one vs rest.
    '''
    res_data = np.concatenate([train_x1, train_x2], axis = 0)
    res_label = np.concatenate([train_y1, train_y2], axis = 0)
    return res_data, res_label

def start_train(train_x1, train_y1, train_x2, train_y2):
    '''
        Use the 2-class data to train a single svm.
    '''
    svc = LinearSVC(C=1)
    train_x, train_y = my_concat(train_x1, train_y1, train_x2, train_y2)
    
    model = svc.fit(train_x, train_y)
   
    return model
