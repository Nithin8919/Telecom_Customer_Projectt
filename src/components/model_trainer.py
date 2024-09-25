import os
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import save_object, evaluate_model, load_object

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV."""
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_score_

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent features from the data')
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]            
            )

            # Define the models and hyperparameter grids for tuning
            models = {
                'DecisionTreeClassifier': {
                    'model': DecisionTreeClassifier(),
                    'param_grid': {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None, 10, 20, 30, 40],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'RandomForestClassifier': {
                    'model': RandomForestClassifier(),
                    'param_grid': {
                        'n_estimators': [100, 200, 500],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                },
                'SVC': {
                    'model': SVC(),
                    'param_grid': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf', 'poly'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'XGBClassifier': {
                    'model': XGBClassifier(),
                    'param_grid': {
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'n_estimators': [100, 200, 300]
                    }
                },
            }

            # Evaluate models after hyperparameter tuning
            model_report = {}
            for model_name, model_info in models.items():
                logging.info(f"Tuning hyperparameters for {model_name}")
                tuned_model, best_score = self.hyperparameter_tuning(
                    model_info['model'],
                    model_info['param_grid'],
                    X_train, y_train
                )
                model_report[model_name] = best_score
                models[model_name] = tuned_model  # Replace with the tuned model

            print(model_report)
            print('\n========================================================================')
            logging.info(f"Model Report: {model_report}")
            
            # Determine the best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            print(f'The best model found: {best_model_name}, Score: {best_model_score}')
            print('\n==========================================================================')
            logging.info(f"Best model found: {best_model_name}, Best model score: {best_model_score}")
            
            # Save the best model using save_object
            save_object(
                path_file=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.error("An exception occurred in model training")
            raise customexception(e, sys)
