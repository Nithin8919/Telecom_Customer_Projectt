import os 
import sys
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils import sava_object,load_object

from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer

obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)


model_trainer_obj=ModelTrainer()
model_trainer_obj.initate_model_training(train_arr,test_arr)

model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr,test_arr)
    