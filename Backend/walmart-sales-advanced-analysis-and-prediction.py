#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from tabulate import tabulate
import lightgbm as lgb
from catboost import CatBoostRegressor


# In[14]:


# Input data files are available in the "../input/" directory.
# First let us load the datasets into different Dataframes
def load_data(datapath):
    data = pd.read_csv(datapath)
   # Dimensions
    print('Shape:', data.shape)
    # Set of features we have are: date, store, and item
    display(data.sample(10))
    return data
traindf=load_data(r'F:\Final Year project\New_Dataset\train.csv')
testdf=load_data(r'F:\Final Year project\New_Dataset\test.csv')
featuresdf=load_data(r'F:\Final Year project\New_Dataset\features.csv')
storesdf=load_data(r'F:\Final Year project\New_Dataset\stores.csv')


# # DATA PREPARATION & ANALYSIS

#  *Merging the features and training data to get cumulative insights from overall*

# In[15]:


traindf1=traindf.merge(featuresdf,how='left',indicator=True).merge(storesdf,how='left')


# In[18]:


traindf1


# *Markdown values are typically a promotional factors and it contains 58% null values,So here Im avoiding it to perform neat analysis.*

# In[19]:


traindf2=traindf1.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)


# In[20]:


traindf2.isna().sum()


# *Let's check any outliers on sales values*

# In[21]:


traindf2.loc[traindf2['Weekly_Sales']<=0] #outliers


# In[22]:



traindf3=traindf2.loc[traindf2['Weekly_Sales']>0]
traindf4=traindf3.drop(['_merge'],axis=1)


# In[24]:


traindf4.sort_values(by='Date')


# In[25]:


traindf4['Type'].unique() #Store varities


# In[26]:


# Import libraries
import matplotlib.pyplot as plt
import numpy as np


# Creating dataset
stores = ['Type A','Type B','Type C']

data = traindf4['Type'].value_counts()

# Creating plot
fig, ax = plt.subplots()
plt.pie(data, labels = stores,autopct='%.0f%%')
ax.set_title('Which Type of stores has more sales')
# show plot
plt.show()


# In[27]:


traindf4['year'] = pd.DatetimeIndex(traindf4['Date']).year #Separating year data.


# In[28]:


# import modules
import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sns

# import file with data
data = traindf4

# prints data that will be plotted
# columns shown here are selected by corr() since
# they are ideal for the plot
print(data.corr())
sns.set_theme(style="whitegrid")
# plotting correlation heatmap
dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
sns.set(rc = {'figure.figsize':(25,8)})

# displaying heatmap
mp.show()


# In[29]:


print(traindf4.dtypes)


# **Year vs Fuel_price**

# In[30]:


import seaborn as sns
sns.set_theme(style="whitegrid")
tips =traindf4
ax = sns.barplot(x="year", y="Fuel_Price", data=tips)
sns.set(rc = {'figure.figsize':(10,4)})


# **Weekly sales vs Store**

# In[31]:


import seaborn as sns
sns.set_theme(style="whitegrid")
tips = traindf4
ax = sns.barplot(x='Store', y="Weekly_Sales", data=tips)


# **Store vs Unemployment**

# In[32]:


# importing packages
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset
data = traindf4

# draw lineplot
sns.lineplot(x="Store", y="Unemployment", data=data)
plt.show()


# In[33]:


traindf4


# In[34]:


traindf4['Dept'].unique()


# In[35]:


# importing required packages
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset
data =traindf4

# draw pointplot
sns.pointplot(x ='Dept',
			y = "Weekly_Sales",
			data = data)
# show the plot
sns.set(rc = {'figure.figsize':(25,8)})
plt.show()


# In[36]:


traindf4['month'] = pd.DatetimeIndex(traindf4['Date']).month #extract month data


# In[37]:


traindf4['week'] = pd.DatetimeIndex(traindf4['Date']).week #extract week data


# In[38]:


traindf5=traindf4.drop(['Date'],axis=1)


# In[39]:


month_wise_sales = pd.pivot_table(traindf5, values = "Weekly_Sales", columns = "year", index = "month")
month_wise_sales.plot()


# **Label encoding for Holiday column and Type**

# In[40]:


# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
traindf5['IsHoliday']= label_encoder.fit_transform(traindf5['IsHoliday'])
traindf5['Type']= label_encoder.fit_transform(traindf5['Type'])

traindf5


# **Correlation Map 2**

# In[41]:


data = traindf5

# prints data that will be plotted
# columns shown here are selected by corr() since
# they are ideal for the plot
print(data.corr())
sns.set_theme(style="whitegrid")
# plotting correlation heatmap
dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
sns.set(rc = {'figure.figsize':(25,8)})

# displaying heatmap
mp.show()


# **Feature Importance Test using various techniques**

# In[48]:


from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
#import shape


# In[49]:


Features=traindf5.drop(['Weekly_Sales'],axis=1)
Target=traindf5['Weekly_Sales']


# In[ ]:





# In[ ]:





# In[50]:


Features


# In[ ]:





# In[79]:


F=Features.drop(["CPI",'Unemployment'],axis=1)


# In[80]:


F


# In[81]:


# from sklearn.model_selection import train_test_split  
# x_train, x_test, y_train, y_test= train_test_split(F, Target, test_size= 0.25, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(F,Target, random_state=101, test_size=0.2)


# In[83]:


F.describe()


# In[84]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()


# In[88]:


X_train_std= sc.fit_transform(X_train)


# In[ ]:





# In[86]:


X_test_std= sc.fit_transform(X_test)


# In[90]:


Y_test


# In[95]:


import joblib
joblib.dump(sc,r'F:\Final _year\Backend\models\sc.sav')


# Linear Regression

# In[96]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()


# In[97]:


lr.fit(X_train_std,Y_train)


# In[98]:


X_test.head()


# In[99]:


Y_pred_lr=lr.predict(X_test_std)
Y_pred_lr


# In[100]:


Y_test


# In[101]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[102]:


print(r2_score(Y_test,Y_pred_lr))
print(mean_absolute_error(Y_test,Y_pred_lr))
print(np.sqrt(mean_squared_error(Y_test,Y_pred_lr)))


# .

# In[ ]:





# XGBOOST

# In[106]:


from xgboost import XGBRegressor


# In[107]:


regressor = XGBRegressor(n_estimators=1000)


# In[108]:


regressor.fit(X_train_std, Y_train)


# In[109]:


Y_pred_xg= regressor.predict(X_train_std)


# In[116]:


r2_train = metrics.r2_score(Y_train, Y_pred_xg)
print("**********XGBOOST**********")
print("Accuracy=",r2_train*100,"%")
print("MAE=",mean_absolute_error(Y_train,Y_pred_xg))
print("MSE=",np.sqrt(mean_squared_error(Y_train,Y_pred_xg)))


# In[122]:


model=joblib.dump(regressor,r'F:\Final _year\Backend\models\XGBoost.sav')


# In[121]:


Y_pred_xg


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Y_pred_rf= rf.predict(X_test_std)


# In[ ]:


print(r2_score(Y_test,Y_pred_rf))
print(mean_absolute_error(Y_test,Y_pred_rf))
print(np.sqrt(mean_squared_error(Y_test,Y_pred_rf)))


# In[ ]:





# In[74]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt


# In[75]:


DTRmodel = DecisionTreeRegressor(max_depth=3,random_state=0)
DTRmodel.fit(x_train,y_train)
y_pred = DTRmodel.predict(x_test)


# In[76]:


print("R2 score  :",r2_score(y_test, y_pred))
print("MSE score  :",mean_squared_error(y_test, y_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred)))


# In[60]:


rf1 = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features = 'sqrt',min_samples_split = 10)
rf1.fit(x_train,y_train)
y_pred1 = rf1.predict(x_test)


# In[61]:


print("R2 score  :",r2_score(y_test, y_pred))
print("MSE score  :",mean_squared_error(y_test, y_pred1))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred1)))


# In[62]:


from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(x_train,y_train)


# In[119]:


y_pred2 = model.predict(x_test)


# In[64]:


print("R2 score  :",r2_score(y_test, y_pred2))
print("MSE score  :",mean_squared_error(y_test, y_pred2))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred2)))


# In[65]:


y_pred2


# In[66]:


#Regularization
from sklearn.linear_model import Ridge
rr_model = Ridge(alpha=0.5)
rr_model.fit(x_train,y_train)


# In[67]:


y_pred3 = model.predict(x_test)


# In[68]:


y_pred3


# In[69]:


print("R2 score  :",r2_score(y_test, y_pred3))
print("MSE score  :",mean_squared_error(y_test, y_pred3))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred3)))


# In[70]:


y_test


# In[ ]:





# In[ ]:




