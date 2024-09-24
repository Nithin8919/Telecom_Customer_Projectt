import os 
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import save_object, evaluate_model,load_object  # Ensure these functions are defined

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent features from the data')
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]            
            )
            
            # Define the models to train
            models = {
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'SVC': SVC(),
            }

            # Evaluate the models using the evaluate_model function
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n========================================================================')
            logging.info(f"Model Report: {model_report}")
            
            # Determine the best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            print(f'The best model found: {best_model_name}, R2_score: {best_model_score}')
            print('\n==========================================================================')
            logging.info(f"Best model found: {best_model_name}, Best model score: {best_model_score}")
            
            # Save the best model using save_object
            save_object(
                path_file=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.info("An exception occurred in model training")
            raise customexception(e, sys)
