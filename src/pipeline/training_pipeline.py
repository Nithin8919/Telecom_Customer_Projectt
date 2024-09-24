import os 
import sys
from src.logger.logging import logging
from src.exception.exception import customexception
from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer

data_path = "/Users/nitin/Documents/Telecom_Customer_Projectt/artifacts/raw.csv"

class TrainingPipeline:
    def start_data_ingestion(self):
        """Initiates data ingestion process."""
        try:
            data_ingestion = Data_Ingestion(data_path)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
            return train_data_path, test_data_path
        except Exception as e:
            logging.error("Error during data ingestion.")
            raise customexception(e, sys)
        
    def start_data_transformation(self, train_data_path, test_data_path):
        """Initiates data transformation process."""
        try:
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            logging.info("Data transformation completed.")
            return train_arr, test_arr
        except Exception as e:
            logging.error("Error during data transformation.")
            raise customexception(e, sys)
        
    def start_model_training(self, train_arr, test_arr):
        """Initiates model training process."""
        try:
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_arr, test_arr)  # Ensure this method exists
            logging.info("Model training completed.")
        except Exception as e:
            logging.error("Error during model training.")
            raise customexception(e, sys)
    
    def start_training(self):
        """Starts the entire training pipeline."""
        try:
            train_data_path, test_data_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            self.start_model_training(train_arr, test_arr)
        except Exception as e:
            logging.error("Error during the training pipeline.")
            raise customexception(e, sys)

if __name__ == "__main__":
    try:
        # Instantiate the training pipeline
        training_pipeline = TrainingPipeline()
        
        # Start the training process
        training_pipeline.start_training()
        
        logging.info("Training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {str(e)}")
        raise customexception(e, sys)