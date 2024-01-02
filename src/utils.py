#use for common functions entire project will use

import os
import sys
import numpy as np
import pandas as pd
import dill

# Add the project root to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
#print(project_root)
from exception import CustomException

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