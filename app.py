# app.py
from flask import Flask, request, render_template
import pickle
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

    # Extract health conditions
    health_conditions = request.form.getlist('health_conditions')
    if not health_conditions:
        health_conditions = ['none']  # Default to 'none' if no condition is selected

    # Combine selected health conditions into a string for the input DataFrame
    input_data['health_condition'] = ', '.join(health_conditions)

    # Set defaults for omitted fields if they are important for the model
    input_data.setdefault('sun_hours_per_week', 0)
    input_data.setdefault('pregnant', 0)
    input_data.setdefault('smoker', 0)  
    
    # Convert input data to DataFrame or suitable format for prediction
    df_input = pd.DataFrame(input_data, index=[0])
    
    # Ensure numeric columns are in correct dtype
    numeric_columns = ['age', 'sun_hours_per_week', 'vitamin_d_level', 'pregnant', 'smoker']
    
    # Check if all required numeric columns are present
    for col in numeric_columns:
        if col not in df_input.columns:
            logging.error(f"Missing expected column: {col}")

    try:
        df_input[numeric_columns] = df_input[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        # Display the DataFrame before prediction for debugging purposes
        logging.info(f"Input DataFrame for prediction:\n{df_input}")
        
        prediction = model.predict(df_input)

        # Log successful prediction
        logging.info(f"Prediction made: {prediction[0]}")
        return render_template('index.html', prediction=prediction[0])
        
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return "Error making prediction."  # Respond with a more user-friendly message.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)