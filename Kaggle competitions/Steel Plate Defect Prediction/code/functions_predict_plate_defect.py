import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier


df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')


def transform_plus(df, columnas=df_train.columns[1:-7]):
    
    X = df[columnas].copy()
    
    X['X_Difference'] = X['X_Maximum'] - X['X_Minimum']
    X['Y_Difference'] = X['Y_Maximum'] - X['Y_Minimum']
    luminosity_cols = ['Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Sum_of_Luminosity']
    X['Mean_Luminosity'] = X[luminosity_cols].mean(axis=1)
    X['Perimeter_Area_Ratio'] = (X['X_Perimeter'] + X['Y_Perimeter']) / X['Pixels_Areas']
    
    return X


def xgboost_grid_models(df_train, df_test, output):
    
    X_train = transform_plus(df_train)
    y_train = df_train[output]
    X_test = transform_plus(df_test)
    
    xgb = XGBClassifier()
    
    params= {'base_score':[0.5] ,
             'booster': ['gbtree'],
             'colsample_bylevel': [1],
             'colsample_bytree': [0.7],
             'gamma': [0, 0.01],
             'learning_rate': [0.1],
             'max_depth': [5,7,10],
             'min_child_weight': [1],
             'n_estimators': [100,150,200],
             'n_jobs': [-1],
             'random_state': [0],
             'reg_alpha': [0.1],
             'reg_lambda': [0.01,0.1],
             'scale_pos_weight': [1],
             'subsample': [0.9],
            }
    
    scoring = ['roc_auc', 'accuracy']
    
    grid_solver_xgb = GridSearchCV (estimator = xgb,
                                       cv = 5,
                                       param_grid = params, 
                                       scoring = scoring,
                                       refit = 'roc_auc',
                                       verbose = 1)
    
    grid_solver_xgb.fit(X_train,y_train)
    
    y_predict = grid_solver_xgb.predict_proba(X_test)
    
    return y_predict[:,1]



