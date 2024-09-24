import mlflow
import os 
import sys
from src.logger.logging import logging
from src.exception.exception import customexception
import mlflow.sklearn
import pickle
from urllib.parse import urlparse
from sklearn.metrics import precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Assuming load_object is defined in your utils
from src.utils.utils import load_object,save_object

class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation started")
        
    def eval_metrics(self, actual, pred):
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info('Evaluation metrics captured')
        return rmse, mae, r2
    
    def initiate_model_evaluation(self, train_array, test_array):
        try:
            # Splitting the test data into features and target
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Load the model from the pickle file
            model_path = os.path.join("artifacts", 'model.pkl')
            model = load_object(model_path)
            
            logging.info("Model has been loaded successfully")
            
            # Determine the MLflow tracking URL scheme
            tracking_url_type_store = urlparse(mlflow.get_tracking_url()).scheme
            
            # Starting an MLflow run
            with mlflow.start_run():
                
                # Making predictions on the test set
                prediction = model.predict(X_test)
                
                # Calculate metrics
                rmse, mae, r2 = self.eval_metrics(y_test, prediction)
                
                # Log the metrics in MLflow
                mlflow.log_metric('rmse', rmse)
                mlflow.log_metric('mae', mae)
                mlflow.log_metric('r2', r2)
                
                # Log the model to MLflow with or without registration
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                    
                logging.info("Model evaluation completed and logged in MLflow.")
                    
        except Exception as e:
            logging.error("An exception occurred during model evaluation.")
            raise customexception(e, sys)
