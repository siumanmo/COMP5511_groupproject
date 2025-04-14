from flask import Flask, request, render_template
import pickle
import pandas as pd
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    raise RuntimeError(f"Model file not found at {model_path}")

@app.route('/')
def home():
    app.logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("Prediction request received")
        
        # Extract form data
        gender = request.form['gender']
        diet = request.form['diet']
        sun_exposure = request.form['sun_exposure']
        activity_level = request.form['activity_level']
        health_condition = request.form['health_condition']
        age = float(request.form['age'])
        sun_hours_per_week = float(request.form['sun_hours_per_week'])
        vitamin_d_level = float(request.form['vitamin_d_level'])
        pregnant = int(request.form['pregnant'])
        smoker = int(request.form['smoker'])

        # Create a DataFrame for input
        input_data = pd.DataFrame({
            'gender': [gender],
            'diet': [diet],
            'sun_exposure': [sun_exposure],
            'activity_level': [activity_level],
            'health_condition': [health_condition],
            'age': [age],
            'sun_hours_per_week': [sun_hours_per_week],
            'vitamin_d_level': [vitamin_d_level],
            'pregnant': [pregnant],
            'smoker': [smoker]
        })

        # Make prediction
        prediction = model.predict(input_data)
        app.logger.info(f"Prediction: {prediction[0]}")
        
        return render_template('index.html', prediction=prediction[0])
    
    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)