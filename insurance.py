import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('insurance.csv',sep=',')
data

df = pd.DataFrame(data)
df

df.dtypes

df.describe()

df.info()

miss_val = (df.isna().sum()/len(df))*100
miss_val

cols = ['sex','smoker','region']
le = LabelEncoder()
for cols in df.columns:
    (df[cols]) = le.fit_transform(df[cols])

data
df

## Sex ##
## female --> 0 ##
## Male --> 1###

## SMoker ##
## Yes --> 1 ##
## No--> 0 ##

## Region ##

# Southwest --> 3
# Southeast --> 2
# northwest --> 1
# northeast --> 0

sns.pairplot(df)
plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)
plt.show()

df.columns

x = df.drop(['charges'],axis=1)
y = df.charges

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

linreg = LinearRegression()
linreg.fit(x_train,y_train)
y_pred = linreg.predict(x_test)
r2_score_linreg = r2_score(y_test,y_pred)
print(r2_score_linreg)
rmse_linreg = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse_linreg)

plt.scatter(y_test,y_pred)
plt.show()

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
r2_score_rf = r2_score(y_test,rf_pred)
print(r2_score_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test,rf_pred))
print(rmse_rf)

plt.scatter(y_test,rf_pred)
plt.show()

from sklearn.model_selection import cross_val_score,GridSearchCV
cv_score = cross_val_score(rf,x,y,cv=5)
cv_score

params = {'n_estimators':[100,200,300,400,500,600,700,800,900,1000],
          'max_depth':[1,2,3,4,5],
          'max_features':['log2','sqrt'],
          'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
          'bootstrap':[True,False],
          'oob_score':[True,False]
          }

from statsmodels.stats.outliers_influence import variance_inflation_factor 
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif.round()

import pickle
pickle.dump(linreg,open('model.pkl','wb'))


len(df.columns)
