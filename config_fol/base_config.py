from types import SimpleNamespace
from typing import Dict, List, Any

base_config = SimpleNamespace()

base_config.model = SimpleNamespace()
base_config.model.hyperparams = None
base_config.model.search_range = None


base_config.model.optuna = SimpleNamespace()
base_config.model.optuna.n_trials = 50
base_config.model.optuna.direction = None