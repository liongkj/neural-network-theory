#!/usr/bin/env python
# coding: utf-8

# In[62]:


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
import pandas as pd


# In[2]:


from res.plot_lib import plot_data,plot_model, set_default,plot_subproblem,plot_submodel
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
# y_train = y.astype(np.int)
# x_train = X.astype(np.double)
# print('Class label counts:', np.bincount(y_train))
# print('X.shape:',X.shape)
# print('y.shape:',y.shape)

normalized_data = np.zeros(data.shape)
# #Normalize data? Do i need this? I think no, data is already at mean zero
normalized_x = preprocessing.StandardScaler().fit_transform(data[:,:2]) * 0.5
normalized_data[:,:2] = normalized_x
normalized_data[:,-1] = y
# normalized_data= np.concatenate((normalized_data,normalized_data[:2,:]))
# plot_data(train_xy(normalized_data)[0],train_xy(normalized_data)[1])
# normalized_data


# ### Divide the two spirals problem into four sub-problems randomly and with prior knowledge, respectively

# #### Random
# 

# In[4]:


def train_xy(shuffled):
#     print(shuffled.shape)
    
    X,y =shuffled[:,:2],shuffled[:,2]
    
    x_train = torch.tensor(X,dtype=torch.float).to(device)
    y_train = torch.tensor(y,dtype=torch.long).to(device)
    return x_train,y_train
    
def partition (list_in, n):
    np.random.shuffle(list_in)
    list_out =  [train_xy(list_in[i::n]) for i in range(n)]
    x_train_unpart = torch.tensor(list_in[:,:2],dtype=torch.float).to(device)
    y_train_unpart = torch.tensor(list_in[:,2],dtype=torch.long).to(device)
    return list_out,x_train_unpart, y_train_unpart

sub_prob,x_train, y_train= partition(normalized_data,4)

plot_subproblem(sub_prob)
# print(normalized_data[:,:2])


# In[102]:



for x, y, label in normalized_data:
    if x<0 and y<0:
        data_1.loc[len(data_1)+1] = {'x':x, 'y':y, 'label':label}
        
    elif x<0 and y>0:
        data_2.loc[len(data_2)+1] = {'x':x, 'y':y, 'label':label}
        
    elif x>0 and y>0:
        data_3.loc[len(data_3)+1] = {'x':x, 'y':y, 'label':label}
        
    elif x>0 and y<0:
        data_4.loc[len(data_4)+1] = {'x':x, 'y':y, 'label':label}
# d1 = train_xy(data_1)
data_1 = train_xy(torch.from_numpy(data_1.values))
data_2 = train_xy(torch.from_numpy(data_2.values))
data_3 = train_xy(torch.from_numpy(data_3.values))
data_4 = train_xy(torch.from_numpy(data_4.values))

data_know = [data_1]+[data_2]+[data_3]+[data_4]
print(len(data_know))


# ## Model

# In[93]:


lr = 1e-1
lambda_l2 = 1e-5
loss_fn = nn.CrossEntropyLoss()
epoch=500

def train(model,x_t,y_t):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=lambda_l2) # built-in L2
    loss_list = []
    acc_list = []
    i = 0
#     print(x_t.shape)
    for t in range(epoch):
        current_loss = 0.0
        prediction = model(x_t)
        print('y_t.shape shape:',y_t.shape)
#         assert prediction.shape == y_t.shape, prediction.shape #y_t.shape
        loss = loss_fn(prediction,y_t)
        
        score, predicted = torch.max(prediction, 1)

        acc = (y_t == predicted).sum().float() / len(y_t)
        if(t+1)%10 == 0:
            print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, loss.item(), acc))
#         current_loss+=loss.item() * x_train.size(0)

        display.clear_output(wait=True)
        optimizer.zero_grad()

        # Backward pass to compute the gradient
        # of loss w.r.t our learnable params. 
        loss.backward()

        # Update params
        optimizer.step()
        loss_list.append(current_loss)
        acc_list.append(acc)
        i+=1
    return loss_list, acc_list



# In[94]:


models = list()
for x_train,y_train in sub_prob:
    print(x_train.shape,y_train.shape)
    
    model = Mlqp(num_features = x_train.shape[1],
             num_hidden = 50,num_output=2,
            logging=logging)
    model.to(device)
    models.append(model)
    loss_list, acc_list = train(model,x_train,y_train)



# In[103]:


models_know = list()
for x_train,y_train in data_know:
    print(x_train.shape,y_train.shape)
    
    model = Mlqp(num_features = x_train.shape[1],
             num_hidden = 50,num_output=2,
            logging=logging)
    model.to(device)
    models_know.append(model)
    loss_list, acc_list = train(model,x_train,y_train)


# In[104]:


plot_submodel(data_know,models_know,figsize=(10,2),caption=None,imgName="m3_mlqp_know",plot_points = True)


# In[7]:


plot_submodel(sub_prob,models,figsize=(10,2),caption=None,imgName="m3_mlqp_random",plot_points = True)


# # Model

# In[106]:


def plot_model_m3(X, y, models, plot_points=False):
    model.cpu()
    mesh = np.arange(-1.3, 1.3, 0.05)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        in_1 = torch.max(models[0](data),1)[1]
        in_2 = torch.max(models[1](data),1)[1]
        in_3 = torch.max(models[2](data),1)[1]
        in_4 = torch.max(models[3](data),1)[1]
        min1=torch.min(in_1,in_2)
        min2=torch.min(in_2,in_3)
        Z = torch.max(min1,min2)
#         print(Z.shape)
#     Z = np.argmax(Z,axis=1).reshape(xx.shape)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=1)
    if(plot_points):
        plot_data(X, y)
    plt.savefig("./img/m3_know.png")


# In[11]:



# print(ans)
# print(x_train.shape)
plot_model_m3(train_xy(normalized_data)[0],train_xy(normalized_data)[1],models,True)


# In[107]:


plot_model_m3(train_xy(normalized_data)[0],train_xy(normalized_data)[1],models_know,True)


# In[ ]:




