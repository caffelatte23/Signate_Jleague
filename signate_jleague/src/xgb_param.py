#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn import metrics



# In[46]:


def param_optimize(train,test):
    y = pd.DataFrame(train['y'])
    x = train.drop(['y'],axis=1)
    target = 'y'

    predictors = [x for x in train.columns if x not in ["y"]]

    def parametors_update(params_test,h_params,train):
        RMSE  = metrics.make_scorer(fmean_squared_error, greater_is_better=False)
        gsearch1 = GridSearchCV(estimator = XGBRegressor(**h_params), param_grid = params_test, scoring= RMSE ,n_jobs=4,iid=False, cv=5)
        gsearch1.fit(train[predictors],train[target])
        h_params.update( gsearch1.best_params_ )

    h_params = {    
        'learning_rate' : 0.1,
        'n_estimators' : 1000,
        'max_depth' : 5,
        'min_child_weight' : 1,
        'gamma' : 0,
        'subsample' : 0.8,
        'colsample_bytree' : 0.8,
        'objective' :  'reg:squarederror',
        'nthread' : 4,
        'scale_pos_weight' : 1,
        'eval_metrics':metrics.make_scorer(fmean_squared_error, greater_is_better=False),
        'seed' : 17
    }

    parametors_update( {'max_depth':[3,5,7,9],'min_child_weight':[1,3,5] },h_params,train)
    parametors_update( {'gamma':[i/10.0 for i in range(0,5)] },h_params,train)
    parametors_update( {'subsample':[i/100.0 for i in range(65,80,5)],'colsample_bytree':[i/100.0 for i in range(65,80,5)]},h_params,train)
    parametors_update( {'max_depth':[3,5,7,9],'min_child_weight':[1,3,5] },h_params,train)
    parametors_update( {'reg_alpha':[140,150,160]} ,h_params,train)
    parametors_update( {'learning_rate':[i/100.0 for i in range(1,11)]} ,h_params,train)



    print(h_params)


# In[47]:


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        
    X_train, X_test, y_train, y_test = train_test_split(dtrain[predictors].values, dtrain[target].values, test_size=0.2, random_state=17)
        
    #Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='rmse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_test)
        
    #Print model report:
    print("\nModel Report")
    print("RMSE Score (Train): %f" % np.sqrt(mean_squared_error(y_test, dtrain_predictions)))


# In[48]:


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_



import optuna

def op_hyper(x,y):

    def objective(trial):
        
        # トレーニングデータとテストデータを分割
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=17)

        # パラメータの指定
        params = {
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'n_estimators' : trial.suggest_int('n_estimators',0 ,1000),
            'max_depth' : trial.suggest_int('max_depth',1,10),
            'min_child_weight' : trial.suggest_int('min_child_weight',1,10),
            'gamma' : trial.suggest_float('gamma',0.0,1),
            'alpha' : trial.suggest_float('alpha',0.0,1),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.05),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.05),
            'objective': 'reg:squarederror',
            'nthread': 4,
            'scale_pos_weight':1,
            # 'eval_metrics':'rmse',
            'seed': 17
        }

        # 学習
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_metric='rmse')
        # 予測
        preds = model.predict(X_test)
        pred_labels = np.rint(preds)
        # 精度の計算
        rmse = metrics.mean_squared_error(y_test, pred_labels)**(1/2)
        return  rmse
        
    study = optuna.create_study()
    study.optimize(objective, n_trials=200)
    return study.best_params


def cat_hyper(x,y,n_trials=200):

    def objective(trial):
        
        # トレーニングデータとテストデータを分割
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=17)

        # パラメータの指定
        params = {
            'iterations' : trial.suggest_int('iterations', 100, 1000,100),                         
            'depth' : trial.suggest_int('depth', 4, 10),                                       
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),               
            'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
            'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'od_wait' :trial.suggest_int('od_wait', 10, 50)
        }

        
        # 学習
        model = cb.CatBoostRegressor(**params)
        train_pool = cb.Pool(X_train,label=y_train)
        test_pool = cb.Pool(X_test,label=y_test)
      
        model.fit(train_pool,eval_set=[test_pool])
        # 予測
        preds = model.predict(X_test)
        pred_labels = np.rint(preds)
        # 精度の計算
        rmse = metrics.mean_squared_error(y_test, pred_labels)**(1/2)
        return  rmse
        
    study = optuna.create_study()
    study.optimize(objective, n_trials)
    return study.best_params


