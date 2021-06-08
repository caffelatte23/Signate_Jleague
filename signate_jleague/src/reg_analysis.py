#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import sort
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ## XGBoost

# In[2]:


import xgboost as xgb
from sklearn.model_selection import cross_validate,KFold


# In[3]:

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


def xgb_analysis(x,y,param = None,print= False):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state= 2400)
    
    if param != None:
        model = xgb.XGBRegressor(**param)
    else:
        model = xgb.XGBRegressor(
            learning_rate = 0.03,
            n_estimators=500,
            max_depth=9,
            min_child_weight=5,
            gamma=0.9,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_alpha=160,
            objective= 'reg:squarederror',
            nthread=4,
            scale_pos_weight=1,
            eval_metrics= metrics.make_scorer(fmean_squared_error, greater_is_better=False),
            seed=17)


    
    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    b_line = 97
    border = np.percentile(pred,b_line)
    for i in range(len(pred)):
        if pred[i] >= border:
            pred[i] = border
    if print:
        print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))
    return model

    
def xgb_analysis2(x,y,param = None):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=17)
    
    if param != None:
        model = xgb.XGBRegressor(**param)
    else:
        model = xgb.XGBRegressor(
            learning_rate = 0.03,
            n_estimators=500,
            max_depth=9,
            min_child_weight=5,
            gamma=0.9,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_alpha=160,
            objective= 'reg:squarederror',
            nthread=4,
            scale_pos_weight=1,
            eval_metrics= metrics.make_scorer(fmean_squared_error, greater_is_better=False),
            seed=17)


    
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test,pred))



# ## LightGBM

# In[4]:


import lightgbm as lgb

def lgbm_analysis(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=17)

    model = lgb.LGBMRegressor( random_state = 17)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))
    return model

# ## CatBoost

# In[5]:


import catboost as cb


# In[6]:


def cat_analysis(x,y,params=None):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=17)

    train_pool = cb.Pool(X_train,label=y_train)
    test_pool = cb.Pool(X_test,label=y_test)

    if params != None:
        model = cb.CatBoostRegressor(**params)
    else:
        model = cb.CatBoostRegressor(loss_function='RMSE')

    model.fit(train_pool,eval_set=[test_pool])

    pred = model.predict(X_test)
    print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))
    return model


# ## RandomForest

# In[7]:


from sklearn.ensemble import RandomForestRegressor


# In[8]:


def rft_analysis(x,y,estimators=100):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=17)

    model = RandomForestRegressor(n_estimators=estimators)
    model.fit(X_train,y_train)
    
    pred = model.predict(X_test)
    print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))
    return model


# ## Lasso回帰

# In[9]:


from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


# In[10]:


def Lasso_analysis(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=17)

    scaler = StandardScaler()
    clf = LassoCV(alphas=10 ** np.arange(-6, 1, 0.1), cv=5)

    scaler.fit(X_train)
    clf.fit(scaler.transform(X_train),y_train)

    pred = clf.predict(scaler.transform(X_test))
    print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))
    return clf


# ## Ridge回帰

# In[11]:


from sklearn.linear_model import Ridge


# In[12]:


def ridge_analysis(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=17)
    model  = Ridge(alpha = 10)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))
    return model


# ## 普通の重回帰分析

# In[13]:


from sklearn.linear_model import LinearRegression 


# In[14]:


def n_analysis(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=17)

    model = LinearRegression()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))
    return model


# ## 多次元回帰

# In[15]:


from sklearn.preprocessing import PolynomialFeatures


# In[16]:


def polyn_analysis(x,y,deg = 2):
  
    cubic = PolynomialFeatures(degree=deg)
    # 生成した基底関数で変数変換を実行
    x_cubic = cubic.fit_transform(x)
    # ホールドアウト法で分割
    X_train, X_test, y_train, y_test = train_test_split(x_cubic, y, test_size = 0.2, random_state = 17)
    # 線形回帰による学習
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))
    return model


# ## SVR 解析

# In[1]:


from sklearn.svm import SVR


# In[18]:


def svr_analysis(x,y,ts = 0.2,rs = 10,result = False):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = ts, random_state=rs)

    model = SVR(kernel='rbf', gamma='auto')
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    print('R^2 : %.3f, RMSE : %.3f' % (r2_score(y_test, pred),np.sqrt(mean_squared_error(y_test,pred))))


    return model


# ## その他の機能



# In[20]:


def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[21]:


def plot_scatter(df):
    
    size = int(len(df.columns)/2 - 1)

    fig, ax = plt.subplots(size - 1, size,figsize=(14,8))
    for i, col in enumerate(df.columns):
        sns.scatterplot(x=col, y="y", ax=ax[i//size][i%size], data = df)
    plt.tight_layout()
    plt.show()


# In[22]:


def Export_csv(model,test,name,idx= False,hd = False):
    try:
        test_id = test['id']
    except:
        print('Not id in test data')
        return None
    
    test = test.drop(['id'],axis=1)
    pred = model.predict(test)
    
    sub = pd.DataFrame({
    "id":test_id,
    "prediction" : pred,
     })
    sub.to_csv(f"../data/submission/{name}.csv",index = idx, header = hd)  

def plot_result(model, X_train, y_train, X_test, y_test, score):
    # 予測値の計算
    p = model.predict(np.sort(X_test))

    # グラフ化
    # plt.scatter(X_test, y_test,label="test data")

    plt.clf()
    plt.scatter(X_test, y_test, label="test data", edgecolor='k',facecolor='w')
    plt.scatter(X_train, y_train, label="Other training data", facecolor="r", marker='x')
    plt.scatter(X_train[model.support_], y_train[model.support_], label="Support vectors", color='c')

    plt.title("predicted results")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    x = np.reshape(np.arange(-3,3,0.01), (-1, 1))
    plt.plot(x, model.predict(x), label="model ($R^2=%1.3f$)" % (score), color='b')

    plt.legend()

def Compare_data(model,test):
    reald = pd.read_csv('../data/res.csv')
    ry = reald['y']
    test = test.drop(['id'],axis=1)
    pred  = model.predict(test)

    # print(f'Actual RMSE : {np.sqrt(mean_squared_error(ry,pred))}')

    return np.sqrt(mean_squared_error(ry,pred))
# %%

def calc_blend(pre1,pre2,y_test,val1=0.0,val2=1.1,step = 0.05,plot = True):
    rmse = []
    val = np.arange(val1,val2,step)
    for i in range(len(val)):
        pred = pre1*val[i] + pre2*(1-val[i])
        rmse.append(np.sqrt(mean_squared_error(y_test,pred)))
        if plot:
            print(f"Blend rate : {round(val[i],3)}, RMSE : {np.sqrt(mean_squared_error(y_test,pred))}")
           
    print(f"best score : {min(rmse)}")

# %%
