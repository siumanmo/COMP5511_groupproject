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
    raise

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
        health_conditions = ['none']

    # Combine selected health conditions into a string
    input_data['health_condition'] = ', '.join(health_conditions)

    # Set defaults for omitted fields if they are important for the model
    input_data.setdefault('sun_hours_per_week', 0)
    input_data.setdefault('pregnant', 0)
    input_data.setdefault('smoker', 0)  

    # Convert input data to DataFrame
    df_input = pd.DataFrame(input_data, index=[0])
    
    # Ensure numeric columns are in correct dtype
    numeric_columns = ['age', 'sun_hours_per_week', 'vitamin_d_level', 'pregnant', 'smoker']
    
    try:
        # Convert appropriate columns to numeric
        df_input[numeric_columns] = df_input[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Log the DataFrame before prediction
        logging.info(f"Input DataFrame for prediction:\n{df_input}")

        # Making the prediction
        prediction = model.predict(df_input)
        logging.info(f"Prediction made: {prediction[0]}")
        return render_template('index.html', prediction=prediction[0])
    
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        logging.error(f"Data Used for Prediction: {df_input}")
        # Return a relevant error message to the HTML template
        return render_template('index.html', error="Error making prediction. Please check your input.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)