import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.logger.logging import logging
from src.exception.exception import customexception


def save_object(path_file, obj):
    try:
        dir_path = os.path.dirname(path_file)
        
        os.makedirs(dir_path, exist_ok=True)  # Ensure the directory exists
        
        with open(path_file, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.info("An exception occurred during saving the model object")
        raise customexception(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            # Evaluate the model
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
    
        return report

    except Exception as e:
        logging.info("An exception occurred during the evaluation of the model")
        raise customexception(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:  # Corrected 'pile_path' to 'file_path'
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("An exception occurred during loading the model object")
        raise customexception(e, sys)
