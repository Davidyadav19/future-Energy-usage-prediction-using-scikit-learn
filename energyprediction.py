#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/David Raj/Downloads/energy.csv")
df = df.set_index('timestamp')
df.index = pd.to_datetime(df.index)

df.plot(style='.',
        figsize=(15, 5),
        color=sns.color_palette()[0],
        title='PJME Energy Use in MW')
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2014']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
#plt.show()
def create_features(df):
    """
    Create time series features based on the time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)

#plt.show()

df.head()
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='load')
ax.set_title('MW by Hour')
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='load', palette='Blues')
ax.set_title('MW by Month')
plt.show()
FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'load'
from sklearn.model_selection import train_test_split

# Assuming `df` is your full dataset
train, test = train_test_split(df, test_size=0.2, random_state=42)

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()
test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['load']].plot(figsize=(15, 5))
test['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Predictions')
plt.show()
score = np.sqrt(mean_squared_error(test['load'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
print(test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(5))



#plt.show()


# In[ ]:




