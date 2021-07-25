#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('NIFTY 50.csv',parse_dates=['Date'])


# In[3]:


data.head()


# In[4]:


data.duplicated().sum()


# In[5]:


data.info()


# In[6]:


data.set_index('Date',inplace=True)


# In[7]:


data.describe()


# In[8]:


num_col = [cname for cname in data if data[cname].dtype in [int,float]]


# In[9]:


from scipy.stats import mode
for i in num_col:
    mode_col = mode(data[i])
    print(i,mode_col)


# In[10]:


data['Open'].plot()


# In[11]:


data['Close'].plot()


# In[12]:


data['Volume'].plot()


# In[13]:


data['Turnover'].plot()


# In[14]:


data['P/E'].plot()


# In[15]:


data['P/B'].plot()


# In[16]:


data['Div Yield'].plot()


# In[17]:


data['High'].plot()


# In[18]:


data['Low'].plot()


# In[19]:


sns.pairplot(data)


# In[20]:


cor = data.corr()
sns.heatmap(cor,annot=True)


# In[21]:


from statsmodels.tsa.stattools import adfuller


# In[22]:


num_col = [cname for cname in data if data[cname].dtype in [int,float]]


# In[23]:


# Ho : Data is not stationary
# H1 : Data is stationary
for i in num_col:  
    test_result = adfuller(data[i])
    print(i,test_result)
    
# Format - test statistics, p-value, usedlag(number of lags used), nobs(the number of observations used for 
# the adf regression and calculation of the critical values), critical values at 1%,5%,10% levels, 
# icbest(the maximized information criterion if autolag is not None)
# Ho > H1 : accept the null hypothesis
# H1 > Ho : reject the null hypothesis
# p-value(alpha) > p-value(cal) : reject the null hypothesis
# p-value(cal)  > p-value(aplha) : accept the null hypothesis


# In[24]:


# At 5% level of significance 
# P/B and Div Yield is stationary


# In[25]:


data['open_d'] = data['Open'] - data['Open'].shift(10)
data['close_d'] = data['Close'] - data['Close'].shift(10)
data['high_d'] = data['High'] - data['High'].shift(10)
data['low_d'] = data['Low'] - data['Low'].shift(10)
data['vol_d'] = data['Volume'] - data['Volume'].shift(10)
data['turn_d'] = data['Turnover'] - data['Turnover'].shift(10)
data['pe_d'] = data['P/E'] - data['P/E'].shift(10)


# In[26]:


data.head(15)


# In[27]:


# adfuller((data['open_d']).dropna())
# adfuller((data['close_d']).dropna())
# adfuller((data['high_d']).dropna())
# adfuller((data['low_d']).dropna())
# adfuller((data['vol_d']).dropna())
# adfuller((data['turn_d']).dropna())
# adfuller((data['pe_d']).dropna())


# In[28]:


num_col = [cname for cname in data if data[cname].dtype in [int,float]]


# In[29]:


for i in num_col:
    test_value = adfuller((data[i]).dropna())
    print(i,test_value)


# In[30]:


data['pe_d'].dropna().plot()


# In[31]:


from  pandas.plotting import autocorrelation_plot
autocorrelation_plot(data['Turnover'])
plt.show()


# In[32]:


from statsmodels.graphics.tsaplots import plot_acf , plot_pacf


# In[33]:


plt.figure(figsize=(12,10))
# ax1 = fig.add_subplot(211)
plot_acf(data['close_d'].iloc[11:],lags=40)


# In[34]:


# ax1 = fig.add.subplot(212)
plt.figure(figsize=(12,10))
plot_pacf(data['close_d'].iloc[11:],lags=40)
plt.show()


# In[35]:


# AR = p 
# MA = q 
# d 


# In[36]:


from statsmodels.tsa.arima_model import ARIMA


# In[37]:


model = ARIMA(data['Close'].dropna(),order=(0,1,1)) # order = p,d,q
model_fit = model.fit()


# In[38]:


model_fit.aic


# In[39]:


# 4,1,1 = 61865.20726962036
# 1,1,1  = 61878.151800414176
# 0,1,1 = 61882.3165299742


# In[40]:


get_ipython().system('jupyter nbconvert --to html ARIMA_model.ipynb')


# In[ ]:





# In[ ]:




