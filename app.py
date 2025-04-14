from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load your model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request
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
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)