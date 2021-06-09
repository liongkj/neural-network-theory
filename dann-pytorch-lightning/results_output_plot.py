#!/usr/bin/env python
# coding: utf-8

# In[17]:


import matplotlib as plt
import seaborn as sns
import pandas as pd


# In[41]:


df = pd.read_csv("result.txt",sep=' ')
df


# In[44]:


df.groupby(['subject']).var()


# In[30]:


sns_plot=sns.catplot(data=df,x="network",y="accuracy",kind="bar")
sns_plot.savefig("../img/network_performance.png")


# In[31]:


sns_plot = sns.catplot(data=df,x="subject",y="accuracy",hue="network",kind="bar")
sns_plot.savefig("../img/subject_accuracy.png")


# In[ ]:




