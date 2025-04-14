from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from form
        user_data = {
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'diet': request.form['diet'],
            'sun_exposure': request.form['sun_exposure'],
            'activity_level': request.form['activity_level'],
            'health_condition': request.form['health_condition'],
            'sun_hours_per_week': int(request.form['sun_hours_per_week']),
            'vitamin_d_level': float(request.form['vitamin_d_level']),
            'pregnant': int(request.form.get('pregnant', 0)),
            'smoker': int(request.form.get('smoker', 0))
        }
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([user_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df).max()
        
        return render_template('result.html', 
                             recommendation=prediction,
                             confidence=f"{probability*100:.1f}%",
                             user_input=user_data)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)