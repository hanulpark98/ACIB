from .base_config import base_config
from copy import deepcopy

LightGBM_config = deepcopy(base_config)

LightGBM_config.model.search_space = {
    
    'objective' : ['fixed', 'binary'],  # Fixed binary classification objective
    'metric' : ['fixed', 'binary_logloss'],  # Fixed evaluation metric
    'boosting_type' : ['suggest_categorical', ['boosting_type', ['gbdt']]], #, 'dart', 'goss'
    'num_leaves' : ['suggest_int', ['num_leaves', 31, 255]],
    'max_depth' : ['suggest_int', ['max_depth', -1, 15]],
    'learning_rate' : ['suggest_float', ['learning_rate', 0.01, 0.3]],
    'n_estimators' : ['suggest_int', ['n_estimators', 50, 300]],
    'min_child_samples' : ['suggest_int', ['min_child_samples', 1, 100]],
    'subsample' : ['suggest_float', ['subsample', 0.5, 1.0]],
    'colsample_bytree' : ['suggest_float', ['colsample_bytree', 0.3, 1.0]],
    'reg_alpha' : ['suggest_float', ['reg_alpha', 1e-3, 10.0, 'log']],
    'reg_lambda' : ['suggest_float', ['reg_lambda', 1e-3, 10.0, 'log']]
}