#data ingestion refers to the process of importing, collecting, and preparing data for analysis.

import os
import sys
# Add the project root to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
#print(project_root)
from exception import CustomException
from logger import logging
from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path_data_path: str=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_injection(self):
        logging.info("Initiating data injection")
        try:
            df=pd.read_csv('D:\\ML_Project\\notebook\\data\\stud.csv')
            logging.info("Read Data Successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.train_data_path,index=False, header=True)

            logging.info("Train Test Split  Intiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            logging.info("Train Test Split Completed / Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initiate_data_injection()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)


        