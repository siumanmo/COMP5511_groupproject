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

# Load the model with error handling
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    app.logger.info("Model loaded successfully!")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    app.logger.error(traceback.format_exc())
    model = None

# Define default values for each feature
DEFAULT_VALUES = {
    # Numeric features
    'age': 30,
    'sun_hours_per_week': 10,
    'vitamin_d_level': 20,
    'pregnant': 0,
    'smoker': 0,
    
    # Categorical features
    'gender': 'female',
    'diet': 'omnivore',
    'sun_exposure': 'moderate',
    'activity_level': 'moderately_active',
    'health_condition': 'none'
}

def get_feature_value(feature_name, form_data):
    """Get feature value from form or use default"""
    value = form_data.get(feature_name)
    if value is None or value == '':
        return DEFAULT_VALUES[feature_name]
    return value

def preprocess_input(form_data):
    """Convert form data to model input format with defaults for missing fields"""
    try:
        # Process each field with fallback to defaults
        processed = {
            'age': float(get_feature_value('age', form_data)),
            'sun_hours_per_week': float(get_feature_value('sun_hours_per_week', form_data)),
            'vitamin_d_level': float(get_feature_value('vitamin_d_level', form_data)),
            'pregnant': int(get_feature_value('pregnant', form_data)),
            'smoker': int(get_feature_value('smoker', form_data)),
            'gender': get_feature_value('gender', form_data),
            'diet': get_feature_value('diet', form_data),
            'sun_exposure': get_feature_value('sun_exposure', form_data),
            'activity_level': get_feature_value('activity_level', form_data),
            'health_condition': get_feature_value('health_condition', form_data)
        }
        
        # Create one-hot encoded features
        features = [
            processed['age'],
            processed['sun_hours_per_week'],
            processed['vitamin_d_level'],
            processed['pregnant'],
            processed['smoker'],
            # Gender one-hot encoding
            1 if processed['gender'] == 'female' else 0,
            1 if processed['gender'] == 'male' else 0,
            1 if processed['gender'] == 'other' else 0,
            # Diet one-hot encoding
            1 if processed['diet'] == 'omnivore' else 0,
            1 if processed['diet'] == 'pescatarian' else 0,
            1 if processed['diet'] == 'vegan' else 0,
            1 if processed['diet'] == 'vegetarian' else 0,
            # Sun exposure one-hot encoding
            1 if processed['sun_exposure'] == 'high' else 0,
            1 if processed['sun_exposure'] == 'low' else 0,
            1 if processed['sun_exposure'] == 'moderate' else 0,
            # Activity level one-hot encoding
            1 if processed['activity_level'] == 'lightly_active' else 0,
            1 if processed['activity_level'] == 'moderately_active' else 0,
            1 if processed['activity_level'] == 'sedentary' else 0,
            1 if processed['activity_level'] == 'very_active' else 0,
            # Health condition one-hot encoding
            1 if processed['health_condition'] == 'chronic_kidney' else 0,
            1 if processed['health_condition'] == 'malabsorption' else 0,
            1 if processed['health_condition'] == 'none' else 0,
            1 if processed['health_condition'] == 'osteoporosis' else 0
        ]
        
        return features
    except Exception as e:
        app.logger.error(f"Preprocessing error: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('index.html', default_values=DEFAULT_VALUES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', 
                               prediction_text='Model not loaded. Please try again later.',
                               form_data=request.form.to_dict(),
                               default_values=DEFAULT_VALUES)
        
        form_data = request.form.to_dict()
        app.logger.info(f"Received form data: {form_data}")
        
        # Log which fields are missing and using defaults
        for field in DEFAULT_VALUES:
            if field not in form_data or form_data[field] == '':
                app.logger.info(f"Using default value for {field}: {DEFAULT_VALUES[field]}")
        
        input_features = preprocess_input(form_data)
        input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES)
        
        prediction = model.predict(input_df)[0]
        return render_template('index.html', 
                            prediction_text=f'Recommended Vitamin: {prediction}',
                            form_data=form_data,
                            default_values=DEFAULT_VALUES)
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                            prediction_text='Error making prediction. Please check your inputs.',
                            form_data=request.form.to_dict(),
                            default_values=DEFAULT_VALUES)

# ... (keep the rest of your existing code, like predict_api)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)