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

# Define expected feature names (update these to match your model's exact requirements)
FEATURE_NAMES = [
    'age', 'sun_hours_per_week', 'vitamin_d_level', 'pregnant', 'smoker',
    'gender_female', 'gender_male', 'gender_other',
    'diet_omnivore', 'diet_pescatarian', 'diet_vegan', 'diet_vegetarian',
    'sun_exposure_high', 'sun_exposure_low', 'sun_exposure_moderate',
    'activity_level_lightly_active', 'activity_level_moderately_active',
    'activity_level_sedentary', 'activity_level_very_active',
    'health_condition_chronic_kidney', 'health_condition_malabsorption',
    'health_condition_none', 'health_condition_osteoporosis'
]

def preprocess_input(form_data):
    """Convert form data to model input format"""
    try:
        # Convert numeric fields
        processed = {
            'age': float(form_data.get('age', 0)),
            'sun_hours_per_week': float(form_data.get('sun_hours_per_week', 0)),
            'vitamin_d_level': float(form_data.get('vitamin_d_level', 0)),
            'pregnant': int(form_data.get('pregnant', 0)),
            'smoker': int(form_data.get('smoker', 0)),
            'gender': form_data.get('gender', ''),
            'diet': form_data.get('diet', ''),
            'sun_exposure': form_data.get('sun_exposure', ''),
            'activity_level': form_data.get('activity_level', ''),
            'health_condition': form_data.get('health_condition', '')
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            app.logger.error("Model not loaded")
            return render_template('index.html', 
                               prediction_text='Model not loaded. Please try again later.',
                               form_data=request.form.to_dict())
        
        # Get and preprocess form data
        form_data = request.form.to_dict()
        app.logger.info(f"Received form data: {form_data}")
        
        # Preprocess input
        input_features = preprocess_input(form_data)
        
        # Convert to DataFrame with correct feature names
        input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES)
        app.logger.info(f"Model input: {input_df.iloc[0].to_dict()}")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        app.logger.info(f"Prediction: {prediction}")
        
        return render_template('index.html', 
                            prediction_text=f'Recommended Vitamin: {prediction}',
                            form_data=form_data)
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return render_template('index.html', 
                            prediction_text='Error making prediction. Please check your inputs and try again.',
                            form_data=request.form.to_dict())

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json(force=True)
        app.logger.info(f"API request data: {data}")
        
        # Preprocess input
        input_features = preprocess_input(data)
        
        # Convert to DataFrame with correct feature names
        input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'recommended_vitamin': prediction,
            'status': 'success'
        })
    
    except Exception as e:
        app.logger.error(f"API error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)