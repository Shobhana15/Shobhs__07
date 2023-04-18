#!/usr/bin/env python
# coding: utf-8

# # Project   :

# ### Business Problem : Identifying the key drivers for estimating loan elgibility for the customer

# ### Statistical Problem :  Estimating loan amount is comes under Linear Regression problem.

# In[2]:


#Importing Packages
import numpy as np
import pandas as pd
import pandas_profiling

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Importing data
data = pd.read_csv("Credit Card Data.csv")


# In[3]:


#Data Understanding 
import pandas as pd

#These will help you to see all records as well as columns in the data set
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
data


# In[4]:


#Checking Missing values
sum(data.isnull().sum())


# In[5]:


#Checking column  and their data types
data.info()


# In[6]:


#Checking Data Dimension
data.shape


# ### Data Analysis :

# In[7]:


#Printing all columns in data set
data.columns


# In[8]:


#Renaming column with proper names
data.rename(columns={"Account Balance":"Account_Balance","Duration of Credit (month)":"Duration_Months","Payment Status of Previous Credit":"Payment_Status","Credit Amount":"Credit_Amount","Value Savings/Stocks":"Value_Savings","Length of current employment":"Length_of_current_employment","Instalment per cent":"Instalment_per_cent","Sex & Marital Status":"Marital_Status","Duration in Current address":"Duration_in_Current_address","Most valuable available asset":"Most_valuable_available_asset","Age (years)":"Age","Concurrent Credits":"Concurrent_Credits","Type of apartment":"Type_of_apartment","No of Credits at this Bank":"No_of_Credits_at_this_Bank","No of dependents":"No_of_dependents","Foreign Worker":"Foreign_Worker"},inplace=True)


# In[9]:


# Univarite Analysis:


# In[10]:


for i in data.columns:
    data[i].hist()
    plt.xlabel(str(i))
    plt.show()


# In[11]:


# Insights from above plot

# --> From Payment Status 0,1 and 3 category customers are low. Chance of selecting category 2 and 4 customers are high.
# --> Credit Amount following Exponential distribution.
# --> Age Varible is similar to exponential distribtuion. 
# --> Most of the customers have 1 or 2 accounts in this bank
# --> More than 60% customers are from category 3 Occupation.
# --> More than 90% of customers belonging to 1 in Foreign_Worker Variable. This in Insignificant variable.
data.Guarantors.value_counts()


# In[12]:


# Bivariate Analysis:


# In[13]:


plt.scatter(data.Credit_Amount,data.Duration_Months, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('Credit Amount')
plt.ylabel('Duration Months')
plt.show()

plt.scatter(data.Credit_Amount,data.Age, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('Credit Amount')
plt.ylabel('Age')
plt.show()


# In[14]:


# Credit_Amount and Age variables have positive Correlation.
# Credit Amount and Duration in Months variables having positive Correlation.

# Based on this Scatter Plot Months_Duration variable is Most Significant compared with Age variable.


# In[15]:


# Checking Bi-Variate Relationship with taget variable
for i in data.columns.difference(['Credit_Amount','Age','Duration_Months','Bad_Flag']):
    plt.scatter(data.Credit_Amount,data[i], alpha=0.5)
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('Credit Amount')
    plt.ylabel(str(i))
    plt.show()  


# In[16]:


#Insights from above plots

#--> if the customers from Account_Balance category is 3 then their Credit Amount is Low
#--> Most of the customers belonging to 1 category in Foreign_Worker variable.
#--> Customers from four category in Purpose variable have low Credit Amount Value.


# ### Data Preparation :

# In[20]:


#Seperating numerical columns
#Continuous Variables : Duration_Months, Credit_Amount, Age
data_num = data.loc[:,("Duration_Months","Credit_Amount","Age")]


# In[21]:


#Creating summary for numerical columns
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.std()/x.mean(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'CV','MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])


# In[22]:


#Implementing user defined summary function
data_num.apply(var_summary).T


# In[24]:


#Outlier treatment
def outlier_capping(x):
    x = x.clip(lower = x.quantile(0.01), upper = x.quantile(0.99))
    return x

#Implementing outlier function 
data_num=data_num.apply(outlier_capping)


# In[25]:


#Categorical Variables:
### Ordinal Variables: Account_Balance, Concurrent_Credits, Duration_in_Current_address, Instalment_per_cent, Length_of_current_employment, No_of_Credits_at_this_Bank, Value_Savings  
### Nominal Variables: Bad_Flag, Foreign_Worker, Guarantors, Marital_Status, Most_valuable_available_asset, No_of_dependents, Occupation, Payment_Status, Purpose, Telephone, Type_of_apartment 
data_cat = data.loc[:,(data.columns.difference(["Duration_Months","Credit_Amount","Age"]))]


# In[26]:


#Seperating nominal columns in Categorical variables
nom_cal = data_cat.columns.difference(["Account_Balance","Concurrent_Credits","Duration_in_Current_address","Instalment_per_cent","Length_of_current_employment","No_of_Credits_at_this_Bank","Value_Savings","Bad_Flag"])


# In[27]:


#Creating dummies for nominal categorical variables
def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df

for c_feature in nom_cal:
    data_cat[c_feature] = data_cat[c_feature].astype('category')
    data_cat = create_dummies(data_cat , c_feature )


# In[28]:


data_cat.head()


# In[29]:


#Combining both categorical and Continuous data
data_final = pd.concat([data_num,data_cat],axis=1)


# # Linear Regression

# #### Normality Checking

# In[26]:


#Checking the Distribution of Credit_Amount
# Target variable following Exponential distribution.
data_final.Credit_Amount.hist()


# In[27]:


#Applying transformation for target variable to make it into Near normal
data_final['Credit_Amount'] = np.log(data_final.Credit_Amount)


# In[28]:


#Plotting transformed dependent variable
sns.distplot(data_final.Credit_Amount)


# #### Variable Reduction

# In[29]:


#Selecting features
features = data_final[data_final.columns.difference( ['Credit_Amount'] )]
target = data_final['Credit_Amount']


# In[30]:


#RFE


# In[31]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

#import itertools

#Checking RFE 

lm = LinearRegression()
#create the RFE model and select 30 attributes
rfe = RFE(lm, n_features_to_select=30)
rfe = rfe.fit(features, target)


# In[32]:


#Alternative of capturing the important variables
RFE_features=features.columns[rfe.get_support()]
features1 = features[RFE_features]


# #### F - Regression

# In[33]:


# Feature Selection based on importance
from sklearn.feature_selection import f_regression
F_values, p_values  = f_regression(features1, target )


# In[34]:


import itertools
f_reg_results = [(i, v, z) for i, v, z in itertools.zip_longest(features1.columns, F_values,  ['%.3f' % p for p in p_values])]
f_reg_results=pd.DataFrame(f_reg_results, columns=['Variable','F_Value', 'P_Value'])


# In[35]:


f_reg_results=pd.DataFrame(f_reg_results, columns=['Variable','F_Value', 'P_Value'])
f_reg_results = f_reg_results.sort_values(by=['P_Value']).head(20)


# In[36]:


f_reg_results


# #### Train and Test data

# In[37]:


#Splitting data into training and testing data sets
train_reg, test_reg = train_test_split(data_final, test_size = 0.3)


# ### Modeling 

# In[38]:


model_reg = sm.ols('Credit_Amount~Telephone_2+Purpose_10+Purpose_1+Occupation_4+Marital_Status_4+Type_of_apartment_3+Duration_Months+Marital_Status_3+Instalment_per_cent+Guarantors_2+Most_valuable_available_asset_3',data = train_reg).fit()


# In[39]:


print(model_reg.summary2())


# In[40]:


#Transforming log values into actual values in both training and testing data sets
train_reg['pred'] =np.exp(model_reg.predict(train_reg))
test_reg['pred'] =np.exp(model_reg.predict(test_reg))
train_reg['Credit_Amount'] = np.exp(train_reg.Credit_Amount)
test_reg['Credit_Amount'] = np.exp(test_reg.Credit_Amount)


# In[41]:


#Checking Metrics for both training and testing data sets

print("MAPE for Training and testing data sets:")
MAPE_train = np.mean(np.abs(train_reg.Credit_Amount - train_reg.pred)/train_reg.Credit_Amount)
print(MAPE_train)
MAPE_train = np.mean(np.abs(test_reg.Credit_Amount - test_reg.pred)/test_reg.Credit_Amount)
print(MAPE_train)
print("\nMSE for Training and testing data sets:")
#MSE for training and testing data sets                           
print(metrics.mean_squared_error(train_reg.Credit_Amount,train_reg.pred)) 
print(metrics.mean_squared_error(test_reg.Credit_Amount,test_reg.pred))
print("\nRMSE for Training and testing data sets:")
#RMSE for training and testing data sets                           
print(np.sqrt(metrics.mean_squared_error(train_reg.Credit_Amount,train_reg.pred)))
print(np.sqrt(metrics.mean_squared_error(test_reg.Credit_Amount,test_reg.pred)))


# In[42]:


#Checking Errors distribution
model_reg.resid.hist(bins=15)
model_reg.resid.to_csv("Residual.csv")
# Residuals following Normal distribution. This is one of the Assumption in Linear Regression.


# In[43]:


### Decile Analysis
#Decile analysis for validation of models - Business validation

train_reg['Deciles']=pd.qcut(train_reg['pred'],10, labels=False)
test_reg['Deciles']=pd.qcut(test_reg['pred'],10, labels=False)


# In[44]:


# Decile Analysis for train data
Predicted_avg = train_reg[['Deciles','pred']].groupby(train_reg.Deciles).mean().sort_index(ascending=False)['pred']
Actual_avg = train_reg[['Deciles','Credit_Amount']].groupby(train_reg.Deciles).mean().sort_index(ascending=False)['Credit_Amount']

Decile_analysis_train = pd.concat([Predicted_avg, Actual_avg], axis=1)

Decile_analysis_train


# In[45]:


# Decile Analysis for train data
Predicted_avg = test_reg[['Deciles','pred']].groupby(test_reg.Deciles).mean().sort_index(ascending=False)['pred']
Actual_avg = test_reg[['Deciles','Credit_Amount']].groupby(test_reg.Deciles).mean().sort_index(ascending=False)['Credit_Amount']

Decile_analysis_test = pd.concat([Predicted_avg, Actual_avg], axis=1)

Decile_analysis_test


# In[46]:


#Exporting Decile analysis in csv formate
#Decile_analysis_train.to_csv('Decile_analysis_train.csv')
#Decile_analysis_test.to_csv('Decile_analysis_test.csv')


# ## Machine Learning 

# In[47]:


#Independent columns
Ind_col = data_final.columns.difference(['Credit_Amount','pred'])


# In[48]:


#Considering Significant Variables for Machine Learning algorithms
train_x, test_x, train_y, test_y = train_test_split(data_final[Ind_col],data_final.Credit_Amount, test_size=0.3, random_state=6)


# In[49]:


# Importing Required Packages
from sklearn.linear_model import Ridge,Lasso


# In[50]:


# Ridge Regression


# In[51]:


Ridge_Reg = Ridge(alpha=0.0001,normalize=True)
Ridge_Reg.fit(train_x,train_y)


# In[52]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y -Ridge_Reg.predict(train_x))/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- Ridge_Reg.predict(test_x))/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,Ridge_Reg.predict(train_x)))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,Ridge_Reg.predict(test_x)))
print(RMSE_test)


# In[53]:


# Lasso Regression :


# In[54]:


Lasso_Reg = Lasso(alpha=0.0001,normalize=True)
Lasso_Reg.fit(train_x,train_y)


# In[55]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y -Lasso_Reg.predict(train_x))/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- Lasso_Reg.predict(test_x))/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,Lasso_Reg.predict(train_x)))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,Lasso_Reg.predict(test_x)))
print(RMSE_test)


# ### Decision Trees :

# In[56]:


#Importing packages
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export
from sklearn.model_selection import GridSearchCV


# In[57]:


#Model building
param_grid = {'max_depth': np.arange(2, 5),
             'max_features': np.arange(4,7)}
data_tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv = 3)
data_tree.fit(train_x,train_y)


# In[58]:


#Predicting values for both training and testing data sets
tree_train_pred= data_tree.predict(train_x)
tree_test_pred=data_tree.predict(test_x)


# In[59]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y -tree_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- tree_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,tree_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,tree_test_pred))
print(RMSE_test)


# ## 2.Ensemble Learining

# ### 2.1 Bagging

# In[60]:


#Importing required packages
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[61]:


#Model building
pargrid_ada = {'n_estimators': [50,60,70,80,100]}

gscv_bagging = GridSearchCV(estimator=BaggingRegressor(), 
                        param_grid=pargrid_ada, 
                        cv=3,n_jobs=-1)

data_bagg =gscv_bagging.fit(train_x, train_y)


# In[62]:


#Predicting values for both training and testing data sets
bagg_train_pred=data_bagg.predict(train_x)
bagg_test_pred=data_bagg.predict(test_x)


# In[63]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y - bagg_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- bagg_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,bagg_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,bagg_test_pred))
print(RMSE_test)


# ### 2.2 Random Forest

# In[64]:


pargrid_ada = {'n_estimators': [50,60,70,80,100],
               'max_depth':[2,3,4]}

data_rf = GridSearchCV(estimator=RandomForestRegressor(), 
                        param_grid=pargrid_ada, 
                        cv=3,n_jobs=-1)
data_rf.fit(train_x,train_y)


# In[65]:


#Predicting values for both training and testing data sets
rf_train_pred=data_rf.predict(train_x)
rf_test_pred=data_rf.predict(test_x)


# In[66]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y - rf_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- rf_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,rf_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,rf_test_pred))
print(RMSE_test)


# ### 2.3 Boosting

# #### Ada Boost

# In[68]:


pargrid_ada = {'n_estimators': [60,70,80,100],
               'learning_rate': [10 ** x for x in range(-4, 2)]}

gscv_ada = GridSearchCV(estimator=AdaBoostRegressor(), 
                        param_grid=pargrid_ada, 
                        cv=3,
                        verbose=True, n_jobs=-1)

data_boost=gscv_ada.fit(train_x, train_y)


# In[69]:


#Predicting values for both training and testing data sets
boost_train_pred=data_boost.predict(train_x)
boost_test_pred=data_boost.predict(test_x)


# In[70]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y - boost_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- boost_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,boost_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,boost_test_pred))
print(RMSE_test)


# #### Gradient boost

# In[72]:


pargrid_ada = {'n_estimators': [60,70,80,100],
               'max_depth':[2,3,4,5],
               'learning_rate': [10 ** x for x in range(-4, 2)]}

gscv_ada = GridSearchCV(estimator=GradientBoostingRegressor(), 
                        param_grid=pargrid_ada, 
                        cv=3,
                        verbose=True, n_jobs=-1)

data_gbm=gscv_ada.fit(train_x, train_y)


# In[73]:


#Predicting values for both training and testing data sets
gbm_train_pred=data_gbm.predict(train_x)
gbm_test_pred=data_gbm.predict(test_x)


# In[74]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y - gbm_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- gbm_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,gbm_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,gbm_test_pred))
print(RMSE_test)


# #### XGBoost

# In[77]:


#Importing Required packages
import xgboost


# In[84]:


pargrid_xg = {'n_estimators': [60,70,80,100,120],
               'max_depth':[2,3,4,5]}

gscv_xg = GridSearchCV(estimator=xgboost.XGBRegressor(), 
                        param_grid=pargrid_xg, 
                        cv=3,n_jobs=-1)

data_xg=gscv_xg.fit(train_x, train_y)


# In[85]:


#Predicting values for both training and testing data sets
xg_train_pred=data_xg.predict(train_x)
xg_test_pred=data_xg.predict(test_x)


# In[86]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y - xg_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- xg_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,xg_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,xg_test_pred))
print(RMSE_test)


# ### K - Nearest Neighbours

# In[88]:


#Importing required modules
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# In[89]:


#Standardising the data because KNN is distance based algorithm 
scaler = StandardScaler()
train_X = scaler.fit_transform(train_x)
test_X = scaler.fit_transform(test_x)


# In[90]:


#Model Building using different tuning parameters
tuned_parameters = [{'n_neighbors': [3,5, 7, 9,11],
                    'leaf_size':[20,30,40,50,60]}]

knn_reg = GridSearchCV(KNeighborsRegressor(),
                   tuned_parameters,
                   cv=3)

data_knn = knn_reg.fit(train_X,train_y)


# In[91]:


#Predicting values for both training and testing data sets
knn_train_pred=data_knn.predict(train_X)
knn_test_pred=data_knn.predict(test_X)


# In[92]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y - knn_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- knn_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,knn_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,knn_test_pred))
print(RMSE_test)


# ## SVM(Support Vector Machine) 

# In[93]:


#Importing required modules
from sklearn.svm import SVR
from sklearn.svm import LinearSVR 


# In[94]:


#Model building using Linear svm 
tuned_parameters = [{'C': [1,0.1,0.001,10,100],
                    'gamma':[0.0001, 0.001, 0.01, 0.1],
                    'kernel':['linear','rbf','poly']}]

svr_reg = GridSearchCV(SVR(),
                   tuned_parameters,
                   cv=3)

data_svr = svr_reg.fit(train_X,train_y)


# In[95]:


#Predicting values for both training and testing data sets
L_svr_train_pred=data_svr.predict(train_x)
L_svr_test_pred=data_svr.predict(test_x)


# In[96]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y - L_svr_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- L_svr_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,L_svr_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,L_svr_test_pred))
print(RMSE_test)


# ## ANN(Artificial Neural Network)

# In[97]:


#Importing required Module
from sklearn.neural_network import MLPRegressor


# In[98]:


pargrid_ann = {'activation': ['relu','tanh','logistic','identity'],
               'alpha':[0.0001,0.001,0.01,1,10,100,1000]}

ann_reg = GridSearchCV(MLPRegressor(),pargrid_ann,cv=3)

data_ann = ann_reg.fit(train_X,train_y)


# In[99]:


#Predicting values for both training and testing data sets
ann_train_pred=data_ann.predict(train_X)
ann_test_pred=data_ann.predict(test_X)


# In[100]:


#Metrics for both training and testing data 

#Mean Absolute Percentage Error 
print("MAPE values for training and testing data :")
MAPE_train = np.mean(np.abs(train_y - ann_train_pred)/train_y)
print(MAPE_train)
MAPE_test = np.mean(np.abs(test_y- ann_test_pred)/test_y)
print(MAPE_test)

#Root Mean Squared error
print("\nRMSE values for training and testing data :")
RMSE_test = np.sqrt(mean_squared_error(train_y,ann_train_pred))
print(RMSE_test)
RMSE_test =np.sqrt(mean_squared_error(test_y,ann_test_pred))
print(RMSE_test)


# # --------------------------------------------------------------------------------------------------------------
# 

# In[ ]:





# In[ ]:




