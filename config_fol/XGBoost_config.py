from .base_config import base_config
from copy import deepcopy

XGBoost_config = deepcopy(base_config)

XGBoost_config.model.search_space = {
    
    'gpu_id' : ['fixed', 2],
    'booster': ['suggest_categorical', ['booster', ['gbtree', 'gblinear', 'dart']]],
    'max_depth': ['suggest_int', ['max_depth', 3, 10]],
    'learning_rate': ['suggest_float', ['learning_rate', 0.02, 0.11]],
    'n_estimators': ['suggest_int', ['n_estimators', 50, 300]],
    'min_child_weight': ['suggest_int', ['min_child_weight', 1, 10]],
    'subsample': ['suggest_float', ['subsample', 0.5, 1.0]],
    'colsample_bytree': ['suggest_float', ['colsample_bytree', 0.3, 1.0]],
    'gamma': ['suggest_float', ['gamma', 0.05, 0.6]],
    'lambda': ['suggest_float', ['lambda', 1e-8, 1.0, 'log']],
    'alpha': ['suggest_float', ['alpha', 1e-8, 2.0, 'log']],
    'objective': ['constant', 'binary:logistic'],
    'eval_metric': ['constant', 'logloss']
}