# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
import traceback
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    app.logger.info("Model loaded successfully!")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    model = None

# Input configuration
INPUT_CONFIG = {
    'age': {'type': 'range', 'min': 1, 'max': 100, 'step': 1, 'default': 30},
    'gender': {'type': 'select', 'options': ['male', 'female', 'other'], 'default': 'female'},
    'diet': {'type': 'select', 'options': ['omnivore', 'vegetarian', 'vegan', 'pescatarian'], 'default': 'omnivore'},
    'sun_exposure': {'type': 'select', 'options': ['low', 'moderate', 'high'], 'default': 'moderate'},
    'health_condition': {'type': 'select', 'options': ['none', 'osteoporosis', 'malabsorption', 'chronic_kidney'], 'default': 'none'},
    'activity_level': {'type': 'select', 'options': ['sedentary', 'lightly_active', 'moderately_active', 'very_active'], 'default': 'moderately_active'},
    'pregnant': {'type': 'binary', 'options': ['No', 'Yes'], 'default': 'No'},
    'smoker': {'type': 'binary', 'options': ['No', 'Yes'], 'default': 'No'}
}

def preprocess_input(form_data):
    """Convert form data to model input format"""
    try:
        # Convert binary inputs
        pregnant = 1 if form_data.get('pregnant') == 'Yes' else 0
        smoker = 1 if form_data.get('smoker') == 'Yes' else 0
        
        # Create one-hot encoded features
        features = [
            float(form_data.get('age', 30)),  # age
            10,  # Default sun_hours_per_week (not in our form)
            20,  # Default vitamin_d_level (not in our form)
            pregnant,
            smoker,
            # Gender one-hot
            1 if form_data.get('gender') == 'female' else 0,
            1 if form_data.get('gender') == 'male' else 0,
            1 if form_data.get('gender') == 'other' else 0,
            # Diet one-hot
            1 if form_data.get('diet') == 'omnivore' else 0,
            1 if form_data.get('diet') == 'pescatarian' else 0,
            1 if form_data.get('diet') == 'vegan' else 0,
            1 if form_data.get('diet') == 'vegetarian' else 0,
            # Sun exposure one-hot
            1 if form_data.get('sun_exposure') == 'high' else 0,
            1 if form_data.get('sun_exposure') == 'low' else 0,
            1 if form_data.get('sun_exposure') == 'moderate' else 0,
            # Activity level one-hot
            1 if form_data.get('activity_level') == 'lightly_active' else 0,
            1 if form_data.get('activity_level') == 'moderately_active' else 0,
            1 if form_data.get('activity_level') == 'sedentary' else 0,
            1 if form_data.get('activity_level') == 'very_active' else 0,
            # Health condition one-hot
            1 if form_data.get('health_condition') == 'chronic_kidney' else 0,
            1 if form_data.get('health_condition') == 'malabsorption' else 0,
            1 if form_data.get('health_condition') == 'none' else 0,
            1 if form_data.get('health_condition') == 'osteoporosis' else 0
        ]
        
        return features
    except Exception as e:
        app.logger.error(f"Preprocessing error: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html', config=INPUT_CONFIG)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', 
                               prediction_text='Model not loaded',
                               config=INPUT_CONFIG)
        
        form_data = request.form.to_dict()
        app.logger.info(f"Form data: {form_data}")
        
        input_features = preprocess_input(form_data)
        input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES)
        
        prediction = model.predict(input_df)[0]
        return render_template('index.html', 
                            prediction_text=f'Recommended Vitamin: {prediction}',
                            form_data=form_data,
                            config=INPUT_CONFIG)
    
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return render_template('index.html', 
                            prediction_text='Error processing your request',
                            config=INPUT_CONFIG)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)