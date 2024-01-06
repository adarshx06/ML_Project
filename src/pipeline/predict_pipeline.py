import os 
import sys
import pandas as pd
# Add the project root to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
from exception import CustomException
from logger import logging
from utils import load_object

class predictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            model_path='artifact/model.pkl' # for model prediction 
            preprocessor_path='artifact/preprocessing.pkl' # for preprocessing categorical data and feature scaling
            print("Before Loading")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            print("After Loading")
            print(features)
            print(features.loc[0])
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e, sys)

### this will be responible for mapping all the input given to the HTML to backend.
class CustomData:

    def __init__(self, gender: str,
            race_ethnicity: str,
            parental_level_of_education ,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int):
    
            self.gender = gender
            self.race_ethnicity = race_ethnicity
            self.parental_level_of_education = parental_level_of_education
            self.lunch = lunch
            self.test_preparation_course = test_preparation_course
            self.reading_score = reading_score
            self.writing_score = writing_score

    def getData_as_dataFrame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            c_D= pd.DataFrame(custom_data_input_dict)
            print(c_D)
            return c_D

        except Exception as e:
            raise CustomException(e, sys)
