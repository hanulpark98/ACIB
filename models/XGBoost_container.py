from xgboost import XGBClassifier, XGBRegressor
from .base_model import Base

class XGBoost_container(Base):
    def __init__(self, task="classification", **kwargs):
        """
        :param task: Either 'classification' or 'regression' depending on your task
        :param kwargs: Additional XGBoost parameters like max_depth, learning_rate, etc.
        """
        self.task = task
        self.model = None
        self.model_name = "xgboost"

        # Initialize XGBoost model based on task type
        if self.task == "classification":
            self.model = XGBClassifier(**kwargs)
        elif self.task == "regression":
            self.model = XGBRegressor(**kwargs)
        else:
            raise ValueError("task should be either 'classification' or 'regression'")

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, **kwargs):
        """
        Fit the model with custom pre-processing or logging if needed.
        """
        print(f"Fitting {self.task} model")
        self.model.fit(X, y, eval_set=eval_set, **kwargs)

    def predict(self, X):
        """
        Use the model to make predictions.
        """
        print(f"Predicting using {self.task} model")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities. Only valid for classification tasks.
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")
        print("Predicting probabilities using classification model")
        return self.model.predict_proba(X)
    
    # def save_model(self, saving_path: str = None) -> None:
    #     assert saving_path is not None, "saving_path cannot be None"
        
    #     if saving_path.split('.')[-1] != 'json':
    #         saving_path += '.json'
            
    #     self.model.save_model(saving_path)
    #     return saving_path

    # def load_model(self, model_path: str = None) -> None:
        
    #     self.model = self.xgb_class()
    #     if model_path is None:
    #         self.model.load_model(self.config.model.model_path)
    #     else:
    #         self.model.load_model(model_path)