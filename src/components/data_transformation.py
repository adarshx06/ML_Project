
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Add the project root to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
#print(project_root)
from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessing_obj_path=  os.path.join('artifact','preprocessing.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    def get_data_transformer_obj(self):
        '''
        This function will return the data transformation object'''
        try:
            numeric_features = ['writing_score','reading_score']
            categorical_features = ["gender",
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"
                                    ]
            num_pipline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler(with_mean=False)),
            ]
            )
            cat_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    ("std_scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numeric and Categorical Pipeline Created")
            logging.info(f"Numeric Features: {numeric_features}")
            logging.info(f"Categorical Features: {categorical_features}")

            preproccessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipline, numeric_features),
                    ("cat_pipeline", cat_pipline, categorical_features),
                ]
            )
            return preproccessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Data Read Successfully")
            logging.info("Data Transformation Initiated")
            preprocessor_obj = self.get_data_transformer_obj()

            target_column = "math_score"
            numerical_columns = ["writing_score","reading_score"]   

            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_features_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            target_features_test_df = test_df[target_column]

            logging.info("Applying Transformation on Train and Test Data")

            input_features_train_array = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_array = preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_array, target_features_train_df]
            test_arr=np.c_[input_features_test_array, target_features_test_df]

            logging.info("Saved preprocessor object.")

            save_object(
                file_path = self.transformation_config.preprocessing_obj_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessing_obj_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
            