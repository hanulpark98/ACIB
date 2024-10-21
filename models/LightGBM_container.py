from lightgbm import LGBMClassifier, LGBMRegressor
from .base_model import Base

class LightGBM_container(Base):
    def __init__(self, task="classification", **kwargs):
        """
        :param task: Either 'classification' or 'regression' depending on your task
        :param kwargs: Additional LightGBM parameters like max_depth, learning_rate, etc.
        """
        self.task = task
        self.model = None
        self.model_name = "lightgbm"

        # Initialize LightGBM model based on task type
        if self.task == "classification":
            self.model = LGBMClassifier(**kwargs)
        elif self.task == "regression":
            self.model = LGBMRegressor(**kwargs)
        else:
            raise ValueError("task should be either 'classification' or 'regression'")

    def fit(self, X, y, **kwargs):
        """
        Fit the model with custom pre-processing or logging if needed.
        """
        print(f"Fitting {self.task} model")
        self.model.fit(X, y, **kwargs)

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
