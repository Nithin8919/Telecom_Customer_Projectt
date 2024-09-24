import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass

# Utility functions for saving and loading objects (assuming these are defined in your utils)
from src.utils.utils import save_object, load_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data transformation is initiated')

            # Define categorical and numerical columns
            categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                'PaperlessBilling']
            categorical_cols2 = ['MultipleLines', 'InternetService', 'Contract', 
                                 'PaymentMethod']
            numerical_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']

            logging.info('Pipeline initiated')

            # Numerical pipeline with mean strategy for imputation
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Changed to mean strategy
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline for binary columns (One-Hot Encoding)
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown="ignore"))
            ])

            # Categorical pipeline for multi-class columns (One-Hot Encoding)
            cat_pipeline2 = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown="ignore"))
            ])

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols),
                ('cat_pipeline2', cat_pipeline2, categorical_cols2)
            ])

            return preprocessor

        except Exception as e:
            logging.error('There is an exception in data transformation')
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Strip whitespace from column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            logging.info('Read train and test data completed')
            
            # Clean numerical columns: replace empty strings with NaN and convert to numeric
            numerical_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
            for col in numerical_cols:
                train_df[col] = train_df[col].replace(r'^\s*$', np.nan, regex=True)  # Ensure all empty strings and whitespaces are replaced with NaN
                test_df[col] = test_df[col].replace(r'^\s*$', np.nan, regex=True)    # Apply the same cleaning for test data
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

            # Check for NaN values and fill with mean if necessary
            train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].mean())
            test_df[numerical_cols] = test_df[numerical_cols].fillna(test_df[numerical_cols].mean())

            # Encode the target variable (Churn) as 0 and 1 instead of using One-Hot Encoding.
            label_encoder = LabelEncoder()
            train_df['Churn'] = label_encoder.fit_transform(train_df['Churn'])  # Converts 'Yes'/'No' to 1/0
            test_df['Churn'] = label_encoder.transform(test_df['Churn'])  # Use transform to avoid fitting again

            preprocessor_obj = self.get_data_transformation()

            target_column_name = 'Churn'
            drop_columns = [target_column_name, 'customerID']

            # Prepare input and target feature DataFrames
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transform the features using the preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying SMOTE to the training data")

            # Applying SMOTE to the training dataset to handle class imbalance
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(input_feature_train_arr, target_feature_train_df)

            logging.info("SMOTE resampling completed")

            # Merging input features and target features for train and test arrays
            train_arr = np.c_[X_resampled, y_resampled]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object to disk using your defined save_object function.
            save_object(
                path_file=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj  # Pass the object as the second argument.
            )

            logging.info('Preprocessed pickle file saved')

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
           logging.error("An exception has occurred in the data transformation")
           raise customexception(e, sys)   

# Example of how to use this class in your main script or pipeline

if __name__ == "__main__":
    try:
        data_transformation = DataTransformation()
        train_data_path = "/Users/nitin/Documents/Telecom_Customer_Projectt/artifacts/train.csv"  # Update with your actual path.
        test_data_path = "/Users/nitin/Documents/Telecom_Customer_Projectt/artifacts/test.csv"    # Update with your actual path.
        
        train_array, test_array = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        print("Training and testing arrays have been created successfully.")
        
    except Exception as e:
        logging.error(f"Error in data transformation pipeline: {str(e)}")