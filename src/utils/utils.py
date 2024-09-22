import os 
import sys
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customexception
import pickle
from pathlib import Path
import numpy as np

from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error

def save_object(path_file, obj):
    try:
        dir_path = os.path.dirname(path_file)
        
        os.mkdir(dir_path, exist_ok = True)
        
        with open(path_file, 'wb') as file_obj:
            pickle.dump(obj , file_obj)
            
    except Exception as e:
        logging.info("exception occured during thee saving the model")
        raise customexception(e,sys)

def evaluate_obj(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            
            model = list(models.values())[i]
        
            model.fit(X_train,y_train)
        
            y_test_pred = model.predict(X_test)
        
            #evaluate
            test_model_score = r2_score(y_test,y_test_pred)
        
            report[list(models.keys())[i]] = test_model_score
    
        return report

    except Exception as e:
        logging.info("exception occured during the evaluation of the model")
        raise customexception(e,sys)
    
def load_obj(file_path):
    try:
        with open(pile_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured during loading the model")
        raise customexception(e,sys)