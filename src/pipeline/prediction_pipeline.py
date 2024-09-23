import os
import sys
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import load_object  # Assuming load_object is a utility function to load your model and preprocessor


class PredictPipeline:
    
    def __init__(self):
        print("Initializing the PredictPipeline object")
        try:
            # Load the preprocessor and model
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            self.model_path = os.path.join("artifacts", "model.pkl")
            
            self.preprocessor = load_object(self.preprocessor_path)
            self.model = load_object(self.model_path)
            
            logging.info("Preprocessor and model loaded successfully")
        
        except Exception as e:
            raise customexception(e, sys)
        
    def predict(self, features):
        try:
            # Transform the features using the preprocessor
            scaled_features = self.preprocessor.transform(features)
            
            # Predict using the loaded model
            predictions = self.model.predict(scaled_features)
            
            return predictions
        
        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 MonthlyCharges: float,
                 TotalCharges: float,
                 tenure: float,
                 gender: str,
                 Partner: str,
                 Dependents: str,
                 PhoneService: str,
                 OnlineSecurity: str,
                 OnlineBackup: str,
                 DeviceProtection: str,
                 TechSupport: str,
                 StreamingTV: str,
                 StreamingMovies: str,
                 PaperlessBilling: str,
                 Churn: str,
                 MultipleLines: str,
                 InternetService: str,
                 Contract: str,
                 PaymentMethod: str):
        
        # Numerical columns
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges
        self.tenure = tenure
        
        # Categorical columns
        self.gender = gender
        self.Partner = Partner
        self.Dependents = Dependents
        self.PhoneService = PhoneService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.PaperlessBilling = PaperlessBilling
        self.Churn = Churn
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.Contract = Contract
        self.PaymentMethod = PaymentMethod
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'MonthlyCharges': [self.MonthlyCharges],
                'TotalCharges': [self.TotalCharges],
                'tenure': [self.tenure],
                'gender': [self.gender],
                'Partner': [self.Partner],
                'Dependents': [self.Dependents],
                'PhoneService': [self.PhoneService],
                'OnlineSecurity': [self.OnlineSecurity],
                'OnlineBackup': [self.OnlineBackup],
                'DeviceProtection': [self.DeviceProtection],
                'TechSupport': [self.TechSupport],
                'StreamingTV': [self.StreamingTV],
                'StreamingMovies': [self.StreamingMovies],
                'PaperlessBilling': [self.PaperlessBilling],
                'Churn': [self.Churn],
                'MultipleLines': [self.MultipleLines],
                'InternetService': [self.InternetService],
                'Contract': [self.Contract],
                'PaymentMethod': [self.PaymentMethod]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame gathered')
            return df
          
        except Exception as e:
            logging.info('An exception occurred while creating the DataFrame from the custom data')
            raise customexception(e, sys)
