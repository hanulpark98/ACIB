from sklearn.preprocessing import LabelEncoder
from fancyimpute import IterativeImputer
import pandas as pd
import numpy as np
import optuna
import shap
import tqdm
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

class DataHandler():
    def __init__(self, data):
        self.column_names = data.columns
        self.feature_data = data.iloc[:, :-1]
        self.label_data = data.iloc[:, -1]
   
    # Encode categorical variables if needed
    def apply_preprocessing(self):
        for column in self.feature_data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.feature_data[column] = le.fit_transform(self.feature_data[column])
    
    def apply_imputation(self, max_iteration = 10):
        imputer = IterativeImputer(max_iter= max_iteration, random_state=0)
        data_imputed = imputer.fit_transform(self.feature_data)
        self.feature_data = pd.DataFrame(data_imputed, columns=self.column_names)


class OptunaHandler():
    def __init__(self, Model_Name, Model, Data_handler: DataHandler ,  Optuna_search_space, OptunaTrials = 300):
        self.model_class = Model
        self.model_name = Model_Name

        self.best_hyperparameters = None
        self.best_hyperparameters_metrics = None

        self.data_x = Data_handler.feature_data
        self.data_y = Data_handler.label_data
        
        self.cat_col = self.data_x.select_dtypes(include=['object', 'category']).columns.tolist()
        self.cat_idx = [self.data_x.columns.get_loc(col) for col in self.cat_col]

        # initialy assigned random dataset split states
        self.random_states =  [416, 182] #[416, 182, 901, 97, 364, 827, 484, 55, 132, 571]
        self.search_space = Optuna_search_space
        self.trials = OptunaTrials
    
    def assign_randomStates(self, rslist):
        self.random_states = rslist

    def objective(self, trial, randomState):
            
        # Initialize the parameter grid according to the search_space format
        param_grid = {}

        # Extracting parameters based on the defined search space
        for key, value in self.search_space.items():
            if value[0] == 'suggest_categorical':
                param_grid[key] = trial.suggest_categorical(value[1][0], value[1][1])
            elif value[0] == 'suggest_int':
                param_grid[key] = trial.suggest_int(value[1][0], value[1][1], value[1][2])
            elif value[0] == 'suggest_float':
                # Check if it's using log scale or not
                if len(value[1]) == 4:  # For log scale
                    param_grid[key] = trial.suggest_float(value[1][0], value[1][1], value[1][2], log=True)  # Fixed line
                elif len(value[1]) == 3:  # If you have only 3 elements (low, high, log)
                    param_grid[key] = trial.suggest_float(value[1][0], value[1][1], value[1][2], log=False)
                else:
                    raise ValueError(f"Expected 3 or 4 parameters for 'suggest_float', got {len(value[1])}.")
            elif value[0] == 'constant':
                param_grid[key] = value[1]

        # Grande Optimziation
        if self.model_name == "GRANDE":

            X_temp, X_test, y_temp, y_test = train_test_split(self.data_x, self.data_y, test_size=0.15, random_state=randomState)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=randomState)

            # Make a GRANDE instance
            Model_Container = self.model_class(self.cat_col, self.cat_idx)

            # Assign grande with hyperparameters
            Model_Container.assign_new_parameters(param_grid)
            
            # Train model with best hyperparameters
            Model = Model_Container.model
            Model .fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            

            pred_prob = Model.predict(X_test)
            binary_preds = np.round(pred_prob[:, 1])

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test,binary_preds).ravel()
            # Calculate sensitivity
            sensitivity = tp / (tp + fn)

            return sensitivity

        # FT-transformer Optimization
        elif self.model_name == "ft-transformer":
            pass


        # XGBoost / CatBoost / LightGBM Optimization
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.data_x, self.data_y, test_size=0.2, random_state=randomState)

            # Make a model instance & assign the test hyperparameters for the trial
            Model_Container = self.model_class(task="classification", **param_grid)
            Model_Container .fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred_proba = Model_Container.predict_proba(X_test)[:, 1]

            # Convert probabilities to binary predictions (threshold of 0.5)
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            # Calculate sensitivity
            sensitivity = tp / (tp + fn)

            return sensitivity

    def run_HPO(self):

        # Main evaluation loop
        best_hyperparameters = None
        best_avg_metrics = None
        all_hyperparameters_metrics = []
        
        # Iterate among the random state list and find the best hyperparameters for each random state
        for index, value in enumerate(self.random_states):
            print(f"Processing split state: {value}, {index+1}/10")
            
            # Optimize hyperparameters for the split state i
            study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
            study.optimize(lambda trial: self.objective(trial, randomState=value), n_trials= self.trials)
            
            best_params = study.best_params
            print(f"Best parameters for split {index+1}: {best_params}")
            
            # All metrices for the 30 random splitted datasets
            split_metrics = []

            if self.model_name == "GRANDE":

                for j in tqdm(range(2)):
                    
                    X_temp_j, X_test_j, y_temp_j, y_test_j = train_test_split(self.data_x, self.data_y, test_size=0.15, random_state=j)
                    X_train_j, X_val_j, y_train_j, y_val_j = train_test_split(X_temp_j, y_temp_j, test_size=0.2, random_state=j)
                    
                    # Make a GRANDE instance
                    Model_Container = self.model_class(self.cat_col, self.cat_idx)

                    # Assign grande with best hyperparameters
                    Model_Container.assign_new_parameters(best_params)
                    
                    # Train model with best hyperparameters
                    Model = Model_Container.model
                    Model.fit(X_train_j, y_train_j, X_val_j, y_val_j)
                    
                    pred_prob = Model.predict(X_test_j)
                    binary_preds = np.round(pred_prob[:, 1])

                    # AUC_ROC and F1 score metrics
                    auc_roc = roc_auc_score(y_test_j, pred_prob[:, 1])

                    f1 = f1_score(y_test_j, binary_preds)

                    # Calculate confusion matrix to derive sensitivity and specificity
                    tn, fp, fn, tp = confusion_matrix(y_test_j, binary_preds).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                    # Store metrics for this iteration
                    split_metrics.append([auc_roc, sensitivity, specificity, f1])
            
            elif self.model_name == "ft-transformer":
                pass

            else:
                for j in tqdm(range(5)):

                    X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(self.data_x, self.data_y, test_size=0.2, random_state=j)    
                    
                    # Make a model instance & assign the test hyperparameters for the trial
                    Model_Container = self.model_class(task="classification", **best_params)
                    Model_Container.fit(X_train_j, y_train_j)
                    
                    y_pred_j_prob = Model_Container.predict_proba(X_test_j)[:, 1]
                    y_pred_j_class = Model_Container.predict(X_test_j)
                    
                    # AUC_ROC and F1 score metrics
                    auc_roc = roc_auc_score(y_test_j, y_pred_j_prob)
                    f1 = f1_score(y_test_j, y_pred_j_class)

                    # Calculate confusion matrix to derive sensitivity and specificity
                    tn, fp, fn, tp = confusion_matrix(y_test_j, y_pred_j_class).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                    # Store metrics for this iteration
                    split_metrics.append([auc_roc, sensitivity, specificity, f1])
            

            # Calculate average metrics for this hyperparameter set
            avg_metrics = np.mean(split_metrics, axis=0)
            all_hyperparameters_metrics.append({
                'hyperparameters': best_params,
                'metrics': avg_metrics
            })

            # Check if this is the best set of hyperparameters based on AUC-ROC
            if best_avg_metrics is None or avg_metrics[0] > best_avg_metrics[0]:  # Using AUC-ROC (avg_metrics[0]) as the main criterion
                best_hyperparameters = best_params
                best_avg_metrics = avg_metrics

        # Save the best hyperparameters and their metrics
        self.best_hyperparameters = best_hyperparameters
        self.best_hyperparameters_metrics = best_avg_metrics

        print(f"Best Hyperparameters: {self.best_hyperparameters}")
        print(f"Best Metrics: {self.best_hyperparameters_metrics}")

    def get_optimized_hp(self):
        return self.best_hyperparameters
    
    def get_optimized_hp_metrics(self):
        return self.best_hyperparameters_metrics

    def test(self):
        self.model_instance = self.model_class()
        print(self.model_class)
        print(self.model_instance)

class ResultHandler():
    def __init__(self, m_name, Model, Hp, Hp_metrics, Data_Handler: DataHandler):
        self.model_name = m_name
        self.model = Model

        self.optimized_Hp = Hp
        self.optimized_Hp_metrics = Hp_metrics

        self.data_X = Data_Handler.feature_data
        self.data_Y = Data_Handler.label_data

        self.cat_col = self.data_X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.cat_idx = [self.data_X.columns.get_loc(col) for col in self.cat_col]

        self.shap_model = None
        self.shap_explainer = None
        self.shap_values = None
        self.shap_testData = None

    def explain(self, feature_name= "age", instance_index = 0):
        
        shap.initjs()
        # GRANDE has no explicit SHAP explainer yet. The authors are working on Tree Shap explainer but it is on progress. 
        # Here we used Kernel explainer to get the SHAP values
        if self.model_name == "GRANDE":

            X_temp, X_test, y_temp, y_test = train_test_split(self.data_X, self.data_Y, test_size=0.15, random_state=1)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=1)
            
            # Make a GRANDE instance
            Model_Container = self.model(self.cat_col, self.cat_idx)

            # Assign grande with best hyperparameters
            Model_Container.assign_new_parameters(self.optimized_Hp)
            
            # Train model with best hyperparameters
            Model = Model_Container.model
            Model.fit(X_train, y_train, X_val, y_val)

            background_sample = shap.sample(X_train, 50)
            explainer = shap.KernelExplainer( Model.predict_for_shap, background_sample, link="logit", keep_index=True)
            shap_values = explainer.shap_values(X_test)

            shap.summary_plot(shap_values, X_test)

        # FT-transformer can also use the Kernel explainer for shap but can also use the Deep Explainer / Gradient Shap made for deep learning models using SHAP values
        elif self.model_name == "FT_Transformer":
            pass
        
        elif self.model_name == "LightGBM":
            X_train, X_test, y_train, y_test = train_test_split(self.data_X, self.data_Y, test_size=0.2, random_state= 1)

            # Make a model instance & assign the test hyperparameters for the trial
            Model= self.model(task="classification", **self.optimized_Hp)
            Model= Model.model
            Model.fit(X_train, y_train)

            background_sample = shap.sample(X_train, 50)
            explainer = shap.KernelExplainer(Model.predict, background_sample, link="logit", keep_index=True)
            shap_values = explainer.shap_values(X_test)

            self.shap_model = Model
            self.shap_explainer = explainer
            self.shap_values = shap_values
            self.shap_testData = X_test

        # XGBoost / CatBoost / LightGBM on the otherhand uses tree shap
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.data_X, self.data_Y, test_size=0.2, random_state= 1)


            # Make a model instance & assign the test hyperparameters for the trial
            Model= self.model(task="classification", **self.optimized_Hp)
            Model= Model.model
            Model.fit(X_train, y_train)

            explainer = shap.TreeExplainer(Model)           
            shap_values = explainer(X_test)

            self.shap_model = Model
            self.shap_explainer = explainer
            self.shap_values = shap_values
            self.shap_testData = X_test

    def summary_plot(self):
        shap.initjs()
        shap.summary_plot(self.shap_values, self.shap_testData)

    # def bar_plot(self):
    #     shap.initjs()
    #     shap.plots.bar(self.shap_values)

    # def decision_plot(self):
    #     shap.initjs()
    #     shap.plots.decision(self.shap_values)

    def heatmap(self):
        shap.initjs()
        shap.plots.heatmap(self.shap_values)

    def force_plot(self, instance_index = 0):
        shap.initjs()
        shap.plots.force(self.shap_values[instance_index])

    def waterfall(self, instance_index = 0):
        shap.initjs()
        shap.plots.waterfall(self.shap_values[instance_index])

    def performance_scores(self):
        return self.optimized_Hp_metrics

class ConfigSelection():
    def __init__(self):
        self.model_configs = {}

    def load_model_config(self, model_name):
        try:
            # Construct the module name dynamically
            module_name = f'config_fol.{model_name}_config'

            # Import the module dynamically
            model_config_module = importlib.import_module(module_name)

            # Retrieve the config attribute
            model_config = getattr(model_config_module, f'{model_name}_config')
            self.model_configs[model_name] = model_config
            return model_config
        
        except ModuleNotFoundError:
            raise ValueError(f"No configuration found for model: {model_name}")
        except AttributeError:
            raise ValueError(f"Configuration for {model_name} does not contain '{model_name}_config'")

class ModelSelection():
    def __init__(self, model_name):

        # Dynamically import the models package
        model_files = importlib.import_module('models')

        # Retrieve the model configuration using the model_name
        model = getattr(model_files, f'{model_name}_container', None)

        if model is None:
            raise ValueError(f"No model found: {model_name}")

        # Store the model configuration
        self.model = model

    def get_model(self):
        return self.model

# class ACIB():
#     def __init__(self):

#         # self.config = config
#         # self.X = X
#         # self.y = y
#         # self.model_type = config.get('model_type', 'xgboost')  # Default to XGBoost
#         # self.hyperparam_ranges = config.get('hyperparam_ranges', self.default_hyperparam_ranges())
#         # self.kfold = StratifiedKFold(n_splits=config.get('kfold', 5))
#         # self.model = None
#         # self.best_params = None
#         # self.start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
#         # self.random_seed = config.get('random_seed', 42)

#         pass


