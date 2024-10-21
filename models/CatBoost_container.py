from catboost import CatBoostClassifier, CatBoostRegressor
from .base_model import Base

class CatBoost_container(Base):
    def __init__(self, task="classification", **kwargs):
        """
        :param task: Either 'classification' or 'regression' depending on your task
        :param kwargs: Additional CatBoost parameters like depth, learning_rate, etc.
        """
        self.task = task
        self.model = None
        self.model_name = "catboost"

        # Initialize CatBoost model based on task type
        if self.task == "classification":
            self.model = CatBoostClassifier(**kwargs)
        elif self.task == "regression":
            self.model = CatBoostRegressor(**kwargs)
        else:
            raise ValueError("task should be either 'classification' or 'regression'")

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, **kwargs):
        """
        Fit the model with custom pre-processing or logging if needed.
        """
        print(f"Fitting {self.task} model")
        self.model.fit(X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, **kwargs)

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