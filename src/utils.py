#use for common functions entire project will use

import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

# Add the project root to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
#print(project_root)
from exception import CustomException
from logger import logging

def save_object(file_path, obj):
    '''
    This function will save the object in the given path
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
       
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for i in range(len(models)):

            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2_score = r2_score(y_train,y_train_pred)
            test_r2_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_r2_score

            return report
        
    except Exception as e:
        raise CustomException(e, sys)