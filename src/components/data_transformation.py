import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation(self):
        try:
            logging.info('Data transformation is initiated')
            
            categorical_cols = ['gender','Partner','Dependents','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection',
                                'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn','MultipleLines',
                                'InternetService','Contract','PaymentMethod']
            numerical_cols = ['MonthlyCharges','TotalCharges','tenure']
            
            logging.info('pipeline initiated')
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('label', LabelEncoder(handle_unknown='ignore'))])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline.numerical_cols),
                ('cat_pipeline', cat_pipeline.categorical_cols) 
            ])
            
            return preprocessor
        
        except Exception as e:
            logging.info('There is an exception in data transformation')
            raise customexception(e,sys)
        
        def initiate_data_transformation(self, train_path, test_path):
            try:
                test_path = pd.read_csv('test_path')
                train_path = pd.read_csv('train_path')
                
                logging.info('read train and test data completed')
                logging.info(f'Train dataframe head:\n{train_data.head().to_string()}')
                logging.info(f'Test dataframe head:\n{train_data.head().to_string()}')
                
                preprocessor_obj = self.get_data_transformation()
                
                target_column_name = 'Churn'
                drop_columns = [target_column_name,'customerID']
                
                input_feature_train_df = 
        
            
            