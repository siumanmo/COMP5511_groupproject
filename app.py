from flask import Flask, request, render_template
import pickle
import pandas as pd
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs

# Load your model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        app.logger.info("Model loaded successfully.")
    except Exception as e:
        app.logger.error(f"Failed to load the model: {str(e)}")
        raise RuntimeError(f"Failed to load the model: {str(e)}")
else:
    app.logger.error(f"Model file not found at {model_path}")
    raise RuntimeError(f"Model file not found at {model_path}")

@app.route('/')
def home():
    app.logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Prediction request received with form data: %s", request.form)
    
    try:
        # Extract form data with validation
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
        
        app.logger.debug(f"Input DataFrame created: {input_data}")

        # Make prediction
        prediction = model.predict(input_data)
        app.logger.info(f"Prediction: {prediction[0]}")
        
        return render_template('index.html', prediction=prediction[0])
    
    except Exception as e:
        app.logger.error(f"Error occurred during prediction: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)