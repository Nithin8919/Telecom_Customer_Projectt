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
                                'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
            categorical_cols2 = np.array(['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']).reshape(-1,1)
            
            
            numerical_cols = ['MonthlyCharges','TotalCharges','tenure']
            
            logging.info('pipeline initiated')
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('label', LabelEncoder(handle_unknown='ignore'))])
            
            cat_pipeline2 = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy="median")),
                ('onehot', OneHotEncoder(handle_unknown="ignore"))
            ])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline.numerical_cols),
                ('cat_pipeline', cat_pipeline.categorical_cols),
                ('cat_pipeline2',cat_pipeline2.categorical_cols2) 
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
                
                input_feature_train_df = train_df.drop(columns= drop_columns,axis =1)
                target_feature_train_df = train_df[target_column_name]
                
                input_feature_test_df = test_df.drop(columns = drop_columns, axis =1)
                target_feature_test_df = test_df[target_column_name]
                
                input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                
                input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df)
                
                logging.info("applying preprocessing object to the columns")
                
                train_arr = np.c_[input_feature_test_arr,np.array[input_feature_train_df]]        
                
                save_object(
                    filepath = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessor_obj
                )
                
                logging.info('preprocessed pickle file saved')
                
                return (
                    train_arr,
                    test_arr
                )           
                
            except Exception as e:
                logging.info("an exception has occured in the data transformation")
                raise customexception(e,sys) 
            