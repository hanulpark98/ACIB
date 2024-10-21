from GRANDE import GRANDE
from .base_model import Base

class GRANDE_container(Base):
    def __init__(self, cat_col, cat_idx, task="classification"):
        """
        :param task: Either 'classification' or 'regression' depending on your task
        :param kwargs: Additional LightGBM parameters like max_depth, learning_rate, etc.
        """
        self.task = task
        self.model_name = "grande"
        self.model = None

        self.categorical_cols= cat_col
        self.categorical_indices = cat_idx

        self.params =  {
            'depth': 5, # tree depth
            'n_estimators': 2048, # number of estimators / trees

            'learning_rate_weights': 0.005, # learning rate for leaf weights
            'learning_rate_index': 0.01, # learning rate for split indices
            'learning_rate_values': 0.01, # learning rate for split values
            'learning_rate_leaf': 0.01, # learning rate for leafs (logits)

            'optimizer': 'adam', # optimizer
            'cosine_decay_steps': 0, # decay steps for lr schedule (CosineDecayRestarts)

            'loss': 'crossentropy', # loss function (default 'crossentropy' for binary & multi-class classification and 'mse' for regression)
            'focal_loss': False, # use focal loss {True, False}
            'temperature': 0.0, # temperature for stochastic re-weighted GD (0.0, 1.0)

            'from_logits': True, # use logits for weighting {True, False}
            'use_class_weights': True, # use class weights for training {True, False}

            'dropout': 0.0, # dropout rate (here, dropout randomly disables individual estimators of the ensemble during training)

            'selected_variables': 0.8, # feature subset percentage (0.0, 1.0)
            'data_subset_fraction': 1.0, # data subset percentage (0.0, 1.0)
        }

        self.args  = {
            'epochs': 1_000, # number of epochs for training
            'early_stopping_epochs': 25, # patience for early stopping (best weights are restored)
            'batch_size': 64,  # batch size for training

            'cat_idx': self.categorical_indices, # put list of categorical indices
            'objective': 'binary', # objective / task {'binary', 'classification', 'regression'}
            
            'random_seed': 42,
            'verbose': 1,       
        }

        # Initialize LightGBM model based on task type
        if self.task == "classification":
            self.model = GRANDE(self.params, self.args)
        else:
            raise ValueError("task should be classification")
        
    def assign_new_parameters(self, params):
        self.model = GRANDE(params,self.args)

    def assign_new_args(self, args):
        self.model = GRANDE(self.params,args)
    
    def fit(self, X_train, Y_train, X_val, Y_val, **kwargs):
        """
        Fit the model with custom pre-processing or logging if needed.
        """
        print(f"Fitting {self.task} model")
        self.model.fit(X_train, Y_train, X_val, Y_val)

    def predict(self, X):
        """
        Use the model to make predictions.
        """
        print(f"Predicting using {self.task} model")
        return self.model.predict(X)

    def predict_for_shap(self, X):
        return self.model.predict(X).unsqueeze(0)

    def predict_proba(self, X):
        """
        Predict class probabilities. Only valid for classification tasks.
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")
        print("Predicting probabilities using classification model")
        return self.model.predict_proba(X)
    


    
        
class MyGRANDE(GRANDE):
    def __init__(self, cat_col, cat_idx, task="classification"):
        """
        :param task: Either 'classification' or 'regression' depending on your task
        :param kwargs: Additional LightGBM parameters like max_depth, learning_rate, etc.
        """
        super().__init__(
            {
            'depth': 5, # tree depth
            'n_estimators': 2048, # number of estimators / trees

            'learning_rate_weights': 0.005, # learning rate for leaf weights
            'learning_rate_index': 0.01, # learning rate for split indices
            'learning_rate_values': 0.01, # learning rate for split values
            'learning_rate_leaf': 0.01, # learning rate for leafs (logits)

            'optimizer': 'adam', # optimizer
            'cosine_decay_steps': 0, # decay steps for lr schedule (CosineDecayRestarts)

            'loss': 'crossentropy', # loss function (default 'crossentropy' for binary & multi-class classification and 'mse' for regression)
            'focal_loss': False, # use focal loss {True, False}
            'temperature': 0.0, # temperature for stochastic re-weighted GD (0.0, 1.0)

            'from_logits': True, # use logits for weighting {True, False}
            'use_class_weights': True, # use class weights for training {True, False}

            'dropout': 0.0, # dropout rate (here, dropout randomly disables individual estimators of the ensemble during training)

            'selected_variables': 0.8, # feature subset percentage (0.0, 1.0)
            'data_subset_fraction': 1.0, # data subset percentage (0.0, 1.0)
        },
        {
            'epochs': 1_000, # number of epochs for training
            'early_stopping_epochs': 25, # patience for early stopping (best weights are restored)
            'batch_size': 64,  # batch size for training

            'cat_idx': self.categorical_indices, # put list of categorical indices
            'objective': 'binary', # objective / task {'binary', 'classification', 'regression'}
            
            'random_seed': 42,
            'verbose': 1,       
        }
        )

        self.params = {
            'depth': 5, # tree depth
            'n_estimators': 2048, # number of estimators / trees

            'learning_rate_weights': 0.005, # learning rate for leaf weights
            'learning_rate_index': 0.01, # learning rate for split indices
            'learning_rate_values': 0.01, # learning rate for split values
            'learning_rate_leaf': 0.01, # learning rate for leafs (logits)

            'optimizer': 'adam', # optimizer
            'cosine_decay_steps': 0, # decay steps for lr schedule (CosineDecayRestarts)

            'loss': 'crossentropy', # loss function (default 'crossentropy' for binary & multi-class classification and 'mse' for regression)
            'focal_loss': False, # use focal loss {True, False}
            'temperature': 0.0, # temperature for stochastic re-weighted GD (0.0, 1.0)

            'from_logits': True, # use logits for weighting {True, False}
            'use_class_weights': True, # use class weights for training {True, False}

            'dropout': 0.0, # dropout rate (here, dropout randomly disables individual estimators of the ensemble during training)

            'selected_variables': 0.8, # feature subset percentage (0.0, 1.0)
            'data_subset_fraction': 1.0, # data subset percentage (0.0, 1.0)
        }

        self.args = {
            'epochs': 1_000, # number of epochs for training
            'early_stopping_epochs': 25, # patience for early stopping (best weights are restored)
            'batch_size': 64,  # batch size for training

            'cat_idx': self.categorical_indices, # put list of categorical indices
            'objective': 'binary', # objective / task {'binary', 'classification', 'regression'}
            
            'random_seed': 42,
            'verbose': 1,       
        }

        self.task = task
        self.model_name = "grande"

        self.categorical_cols= cat_col
        self.categorical_indices = cat_idx
        
    def assign_new_parameters(self, params = None, args = None):
        if params is None:
            params = self.params
        if args is None:
            args = self.args

        super().__init__(params,args)
    
    def fit(self, X_train, Y_train, X_val, Y_val, **kwargs):
        """
        Fit the model with custom pre-processing or logging if needed.
        """
        print(f"Fitting {self.task} model")
        super().fit(X_train, Y_train, X_val, Y_val)
        # self.model.fit(X_train, Y_train, X_val, Y_val)

    def predict(self, X):
        """
        Use the model to make predictions.
        """
        print(f"Predicting using {self.task} model")
        preds = super().predict(X)
        if len(preds.shape == 1):
            preds = preds.unsqueeze(0)
        return preds
        # return self.model.predict(X)