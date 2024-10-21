from .base_config import base_config
from copy import deepcopy

GRANDE_config = deepcopy(base_config)

GRANDE_config.model.search_space = {
    
    'depth' : ['suggest_int', ['depth', 3, 7]],
    'n_estimators' : ['suggest_int', ['n_estimators', 512, 2048]],

    'learning_rate_weights' : ['suggest_float', ['learning_rate_weights', 0.001, 0.01]],
    'learning_rate_index' : ['suggest_float', ['learning_rate_index', 0.001, 0.02]],
    'learning_rate_values' : ['suggest_float', ['learning_rate_values', 0.001, 0.02]],
    'learning_rate_leaf' : ['suggest_float', ['learning_rate_leaf', 0.001, 0.02]],

    'cosine_decay_steps' : ['suggest_categorical', ['cosine_decay_steps', [0, 100]]],

    'dropout' : ['suggest_categorical', ['dropout', [0, 0.25, 0.5]]],

    'selected_variables' : ['suggest_categorical', ['selected_variables', [1.0, 0.75, 0.5]]],
    'data_subset_fraction' : ['suggest_categorical', ['data_subset_fraction', [1.0, 0.8]]]
}