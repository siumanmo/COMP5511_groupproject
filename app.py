# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    logging.info(f"Received input data: {input_data}")

    # Set defaults for omitted fields if they are important for the model
    input_data.setdefault('sun_hours_per_week', 0)  # Default to 0 if not provided
    input_data.setdefault('pregnant', 0)  # Default to 0 if not provided
    input_data.setdefault('smoker', 0)  # Default to 0 if not provided
    
    # Convert input data to DataFrame or suitable format for prediction
    df_input = pd.DataFrame(input_data, index=[0])
    
    # Ensure numeric columns are in correct dtype
    numeric_columns = ['age', 'sun_hours_per_week', 'vitamin_d_level', 'pregnant', 'smoker']
    try:
        df_input[numeric_columns] = df_input[numeric_columns].apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        logging.error(f"Error converting input data to numeric: {str(e)}")
        return "Error processing input data."

    try:
        prediction = model.predict(df_input)
        logging.info(f"Prediction made: {prediction[0]}")
        return render_template('index.html', prediction=prediction[0])
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return "Error making prediction."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)