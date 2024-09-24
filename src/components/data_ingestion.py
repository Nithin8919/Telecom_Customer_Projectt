import pandas as pd
import numpy as np
import sys
import os
from src.logger.logging import logging
from src.exception.exception import customexception
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", 'train.csv')
    test_data_path: str = os.path.join("artifacts", "test.csv")

class Data_Ingestion:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion is started")
        
        try:
            # Read the data
            data = pd.read_csv(self.data_path)
            data.columns = data.columns.str.strip()
            logging.info('Dataframe is read')
        
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved the raw dataset in artifacts folder")
            
            # Perform train-test split
            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(data, test_size=0.25)
            train_data.columns = train_data.columns.str.strip()
            test_data.columns = test_data.columns.str.strip()

            logging.info("Train-test split completed")
            
            # Save train and test datasets
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion part completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("Data ingestion failed")
            raise customexception(e, sys)

# Make sure this block runs only when the script is executed directly
if __name__ == '__main__':
    data_path = "/Users/nitin/Documents/Telecom_Customer_Projectt/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    obj = Data_Ingestion(data_path)
    obj.initiate_data_ingestion()
