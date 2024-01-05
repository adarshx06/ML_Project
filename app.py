import os
import sys
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Add the project root to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)
from src.pipeline.predict_pipeline import CustomData, predictPipeline


application = Flask(__name__)
app = application

#ROUTE FOR Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():

    if request.method =='GET':
        return  render_template('home.html') 
    
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df = data.getData_as_dataFrame()
        #print(pred_df)

        predict_pipeline = predictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)