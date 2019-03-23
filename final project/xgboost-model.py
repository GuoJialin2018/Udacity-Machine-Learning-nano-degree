# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 08:41:46 2018

@author: HP
"""
import pandas as pd
import numpy as np
from sklearn import cross_validation
import xgboost as xgb
import random
import matplotlib.pyplot as plt
# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe
#把竞争年份和月份转化为float
#def convertCompetitionOpen(df):
#    try:
#        date = '{}-{}'.format(int(df['CompetitionOpenSinceYear']),int(df['CompetitionOpenSinceMonth']))
#        return pd.to_datetime(date)
#    except:
#        return np.nan
#
##将竞争开始年份和月份转化为float:
#def convertPromo2(df):
#    try:
#        date='{}{}1'.format(int(df['Promo2SinceYear']),int(df['Promo2SinceWeek']))
#        return pd.to_datetime(date,format='%Y%W%w')
#    except:
#        return np.nan
# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'
                     ])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)
    data['PromoInterval1'] = data['PromoInterval1'].astype(float)
    data['PromoInterval2'] = data['PromoInterval2'].astype(float)
    data['PromoInterval3'] = data['PromoInterval3'].astype(float)
    data['PromoInterval4'] = data['PromoInterval4'].astype(float)
    features.append('PromoInterval1')
    features.append('PromoInterval2')
    features.append('PromoInterval3')
    features.append('PromoInterval4')
    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)
    
    features.append('StoreType')
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)
   
    features.append('Assortment')
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)
    
    features.append('StateHoliday')
    data.loc[data["StateHoliday"]=='a','StateHoliday']=1
    data.loc[data["StateHoliday"]=='b','StateHoliday']=2
    data.loc[data["StateHoliday"]=='c','StateHoliday']=3
    data['StateHoliday']=data['StateHoliday'].astype(int)

print("Load the training, test and store data using pandas")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
store = pd.read_csv("store.csv")

#清洗CompetitionDistance,针对没有竞争对手的距离设置为100000，开业时间设置为2018.1
store['CompetitionDistance']=store['CompetitionDistance'].fillna(1000000)
store['CompetitionDistance']=store['CompetitionDistance'].astype(int)
store.loc[store["CompetitionDistance"]==100000,'CompetitionOpenSinceYear']=2018
store.loc[store["CompetitionDistance"]==100000,'CompetitionOpenSinceMonth']=1

#训练值最小日期是2013-01-01,把没有开业时间的竞争对手店面开店时间随机设置到训练时间前
random.seed(20)
store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(random.randint(2002,2012))
store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(random.randint(1,12))
store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].astype(int)
store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].astype(int)

#清洗Promo2Since数据
store['Promo2SinceWeek']=store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear']=store['Promo2SinceYear'].fillna(0)
store['Promo2SinceWeek']=store['Promo2SinceWeek'].astype(int)
store['Promo2SinceYear']=store['Promo2SinceYear'].astype(int)

#处理PromoInterval
num_month = {'Jan,Apr,Jul,Oct':'1,4,7,10','Feb,May,Aug,Nov':'2,5,8,11','Mar,Jun,Sept,Dec':'3,6,9,12'}
store['PromoInterval'] =store['PromoInterval'].map(num_month)
store['PromoInterval']=store['PromoInterval'].fillna('0,0,0,0')
PromoInterval1=[]
PromoInterval2=[]
PromoInterval3=[]
PromoInterval4=[]
m=store['PromoInterval']
for x in m:
    y=x.split(',')
    PromoInterval1.append(y[0])
    PromoInterval2.append(y[1])
    PromoInterval3.append(y[2])
    PromoInterval4.append(y[3])
store['PromoInterval1']=pd.Series(PromoInterval1)
store['PromoInterval2']=pd.Series(PromoInterval2)
store['PromoInterval3']=pd.Series(PromoInterval3)
store['PromoInterval4']=pd.Series(PromoInterval4)
store.drop('PromoInterval',axis=1, inplace=True)
store = store.drop_duplicates()


print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[(train["Open"] != 0) & (train['Sales'] != 0)]

print("Join with store")
train = pd.merge(train, store, on='Store',how='inner')
test = pd.merge(test, store, on='Store',how='inner')

#根据时间划分数据集
train = train.sort_values(by='Date')
test=test.sort_values(by='Date')

#根据趋势去掉部分商店的异常数据
train['DateInt']=pd.to_datetime(train['Date']).astype(np.int64)
store_dates_to_remove = {   105:1.368e18, 163:1.368e18,
                            172:1.366e18, 364:1.37e18,
                            378:1.39e18, 523:1.39e18,
                            589:1.37e18, 663:1.39e18,
                            676:1.366e18, 681:1.37e18,
                            700:1.373e18, 708:1.368e18,
                            709:1.423e18, 730:1.39e18,
                            764:1.368e18, 837:1.396e18,
                            845:1.368e18, 861:1.368e18,
                            882:1.368e18, 969:1.366e18,
                            986:1.368e18, 192:1.421e18,
                            263:1.421e18, 500:1.421e18,
                            797:1.421e18, 815:1.421e18,
                            825:1.421e18}
for key,value in store_dates_to_remove.items():
    train.loc[(train['Store'] == key) & (train['DateInt'] < value), 'Delete'] = True
train = train.loc[train['Delete'] != True]
del train['DateInt'],train['Delete']

#删除outlier值
def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
for i in train['Store'].unique():
    train.loc[train['Store'] == i, 'Outlier'] = mad_based_outlier(train.loc[train['Store'] == i]['Sales'], 3)
train = train.loc[train['Outlier'] == False]    
  
features = []
build_features(features, train)
build_features([], test)
train.to_csv("newtrain.csv", index=False)
test.to_csv("newtest.csv",index=False)

#-------------------------以上为数据预处理部分-------------------------------

#train = pd.read_csv("newtrain.csv")
#test = pd.read_csv("newtest.csv")

params = {"objective": "reg:linear",
          "eta": 0.2,
          "max_depth":10,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1,
          "learning_rate":0.04,
          "n_jobs":6
          }
num_trees =10000
print("Train a XGBoost model")

#预测后0.02的数据
val_size=int(len(train)*0.99)
#X_train=train[1:val_size]
#X_test=train[val_size:]
X_train, X_test = cross_validation.train_test_split(train, test_size=0.01)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=40, feval=rmspe_xg, verbose_eval=40)

print("Validating")
train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
print('error', error)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test[features]))
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
submission.to_csv("submission.csv", index=False)

fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(gbm, max_num_features=50, height=0.8, ax=ax)
plt.show()

