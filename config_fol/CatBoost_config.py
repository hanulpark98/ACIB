from .base_config import base_config
from copy import deepcopy

CatBoost_config = deepcopy(base_config)

CatBoost_config.model.search_space = {

    'task_type' : ['fixed', 'GPU'],  # Enable GPU training
    'devices' : ['fixed', '3'],      # Use GPU device 3
    'iterations' : ['suggest_int', ['iterations', 100, 1000]],
    'depth' : ['suggest_int', ['depth', 3, 10]],
    'learning_rate' : ['suggest_float', ['learning_rate', 0.001, 0.2]],
    'l2_leaf_reg' : ['suggest_float', ['l2_leaf_reg', 1e-4, 10.0, 'log']],
    'bagging_temperature' : ['suggest_float', ['bagging_temperature', 0.0, 1.0]],
    'random_strength' : ['suggest_float', ['random_strength', 0.0, 10.0]],
    'border_count' : ['suggest_int', ['border_count', 32, 255]],
    'scale_pos_weight' : ['suggest_float', ['scale_pos_weight', 0.5, 2.0]]
}