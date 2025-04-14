# app.py
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the trained model
def load_model():
    # Define the model structure (same as training)
    categorical_features = ['gender', 'diet', 'sun_exposure', 'activity_level', 'health_condition']
    numeric_features = ['age', 'sun_hours_per_week', 'vitamin_d_level', 'pregnant', 'smoker']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    # Load the saved model weights
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    return model

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'diet': request.form['diet'],
            'sun_exposure': request.form['sun_exposure'],
            'sun_hours_per_week': float(request.form['sun_hours_per_week']),
            'vitamin_d_level': float(request.form['vitamin_d_level']),
            'activity_level': request.form['activity_level'],
            'pregnant': int(request.form.get('pregnant', 0)),
            'smoker': int(request.form.get('smoker', 0)),
            'health_condition': request.form['health_condition']
        }
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return render_template('index.html', 
                             prediction_text=f'Recommended Vitamin: {prediction}',
                             form_data=data)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)