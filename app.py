from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature names in the order the model expects them
feature_names = ['age', 'sun_hours_per_week', 'vitamin_d_level', 'pregnant', 'smoker',
                 'gender', 'diet', 'sun_exposure', 'activity_level', 'health_condition']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Convert to DataFrame with correct column order
        input_data = pd.DataFrame([data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', 
                             prediction_text=f'Recommended Vitamin: {prediction}',
                             form_data=data)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get JSON data
        data = request.get_json(force=True)
        
        # Convert to DataFrame with correct column order
        input_data = pd.DataFrame([data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({'recommended_vitamin': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)