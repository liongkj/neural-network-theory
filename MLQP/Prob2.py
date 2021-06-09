#!/usr/bin/env python
# coding: utf-8

# In[139]:


import matplotlib.pyplot as plt
import torch
from torch import nn,optim,functional as F
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import random
import numpy as np
from keras.utils.np_utils import to_categorical  
from IPython import display
from skorch import NeuralNetClassifier
import time
import os


# In[140]:


from res.plot_lib import plot_data,plot_model, set_default, plot_submodel
from models.mlqp import *
set_default(figsize=(4, 4))
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
logging=False
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


data = np.genfromtxt('train_data.txt',autostrip=True)
X,y = data[:,:2],data[:,2]
y_train = y.astype(np.int)
x_train = X.astype(np.double)
print('Class label counts:', np.bincount(y_train))
print('X.shape:',X.shape)
print('y.shape:',y.shape)

#Shuffle data and train/test split
x_train,y_train = shuffle(x_train,y_train)
# #Normalize data? Do i need this? I think no, data is already at mean zero
x_train = preprocessing.StandardScaler().fit_transform(x_train) * 0.5

x_train = torch.tensor(x_train,dtype=torch.float).to(device)
y_train = torch.tensor(y_train,dtype=torch.long).to(device)


# In[4]:


# plot_data(x_train,y_train,True)
# y_train
num_features = 2 #dimension
num_hidden = 200 #hidden_units
num_output = 2 #output
# lr = 1e-2
lambda_l2 = 1e-5
# loss_fn = 
epoch=1000


# ## Model

# In[5]:




net = NeuralNetClassifier(
    Mlqp,
    module__num_features=num_features,
    module__num_output=num_output,
    module__num_hidden=num_hidden,
    max_epochs = epoch,
    criterion=nn.CrossEntropyLoss,
    optimizer__weight_decay = lambda_l2,
    optimizer = torch.optim.Adam,
    device = device
)



# ## Hyperparameter tuning

# In[220]:


lr_list = [1e-3,1e-1,3,] 
hidden_list = [50]

param_grid = {
    "lr":lr_list,
    "module__num_hidden":hidden_list    
}

grid = GridSearchCV(net,
                    param_grid,
                    n_jobs=-1,
                    refit=True,
                    return_train_score=True,
                    verbose=1,
                    scoring='accuracy')
# print best parameter after tuning
# print(grid.get_params().keys())


# In[97]:



grid.fit(x_train, y_train)


# In[98]:


report(grid.cv_results_,10)


# In[52]:


# Utility function to report best scores (found online)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with score: {0}".format(i))
            print("Mean fit time: {0:.3f} (Mean train score: {1:.3f})".format(
                  results['mean_fit_time'][candidate],
                  results['mean_train_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[53]:


grid.cv_results_


# ## Plot Models

# In[227]:


loss_fn = nn.CrossEntropyLoss()
def train(model,learning_rate):
    print(model)
    print(learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
    start = time.time()
    loss_list = []
    acc_list = []
    time_list = []
    for t in range(500):
        current_loss = 0.0
        prediction = model(x_train)
    #     print('y_pred shape:',y_pred.shape)
        loss = loss_fn(prediction,y_train)

        score, predicted = torch.max(prediction, 1)

        acc = (y_train == predicted).sum().float() / len(y_train)
        if(t+1)%100 == 0:
    #         plot_model(x_train, y_train, model)
            end = time.time()
            print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f [ELAPSED]:%s" % (t, loss.item(), acc,end - start))
        current_loss+=loss.item() * x_train.size(0)

#         display.clear_output(wait=True)
        optimizer.zero_grad()

        # Backward pass to compute the gradient
        # of loss w.r.t our learnable params. 
        loss.backward()

        # Update params
        optimizer.step()
        loss_list.append(current_loss)
        acc_list.append(acc)
    print("")
    
    return loss_list, acc_list

    


# In[228]:


models = list()
loss_=list()
acc_=list()
for hid in hidden_list:
    for learn in lr_list:
        model = Mlqp(num_features = num_features,
                     num_hidden = hid,num_output=num_output,
                    logging=logging)
        model.to(device)
        loss_list, acc_list = train(model,learn)
        models.append(model)
        loss_.append(loss_list)
        acc_.append(acc_list)


# In[226]:



fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(loss_[0])

ax2.plot(loss_[1])
ax3.plot(loss_[2])
ax1.set_title(lr_list[0])
ax2.set_title(lr_list[1])
ax3.set_title(lr_list[2])
plt.savefig('./img/lr_loss.png')


# In[219]:


fig = plt.figure(figsize=(5,6))
ax1 = fig.add_subplot(221)

# plt.figure(figsize=(10,10))
# plt.subplot(2,2,1,)
plot_model(x_train,y_train,models[1])
ax2 = fig.add_subplot(222)
# plt.subplot(2,2,2)
plot_model(x_train,y_train,models[4])
ax1.set_title('12 hidden')
ax2.set_title('50 hidden')
plt.savefig('./img/num_hidden_decision.png')


# In[108]:


normalized_data = np.zeros(data.shape)
normalized_data[:,:2] = x_train
normalized_data[:,-1] = y_train

def train_xy(shuffled):
    X,y =shuffled[:,:2],shuffled[:,2]
    x_train = torch.tensor(X,dtype=torch.float).to(device)
    y_train = torch.tensor(y,dtype=torch.long).to(device)
    return x_train,y_train

def partition (list_in, n):
#     np.random.shuffle(list_in)
    list_out =  [train_xy(list_in[i::n]) for i in range(n)]
    return list_out

sub_prob = partition(normalized_data,1)   

# print(normalized_data.shape)
lst_data = sub_prob*6
# print(lst_data[0].shape)
# plot_model(normalized_data[:,:2],models[0])
print("hidden:",hidden_list[0])
plot_submodel(lst_data,models,figsize=(10,2),caption=None,imgName="learning_rate_comparison_test")


# In[ ]:





# In[ ]:




