from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and encoder
nb_model = joblib.load('models/vitamin_recommender_nb.pkl')
target_encoder = joblib.load('models/target_encoder.pkl')

# Define feature columns in the same order as training
FEATURE_COLUMNS = [
    'age', 'gender', 'dietary_habits', 'physical_activity', 
    'sleep_pattern', 'stress_level', 'health_goal', 'existing_condition'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'dietary_habits': request.form['dietary_habits'],
            'physical_activity': request.form['physical_activity'],
            'sleep_pattern': request.form['sleep_pattern'],
            'stress_level': request.form['stress_level'],
            'health_goal': request.form['health_goal'],
            'existing_condition': request.form['existing_condition']
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables (using same encoding as training)
        # Note: In a production app, you should save and load the encoders for each feature
        gender_map = {'male': 1, 'female': 0}
        dietary_map = {'vegetarian': 2, 'vegan': 1, 'omnivore': 0}
        activity_map = {'sedentary': 0, 'moderate': 1, 'active': 2}
        sleep_map = {'poor': 0, 'average': 1, 'good': 2}
        stress_map = {'high': 0, 'medium': 1, 'low': 2}
        goal_map = {'weight_loss': 0, 'muscle_gain': 1, 'general_health': 2}
        condition_map = {'none': 0, 'diabetes': 1, 'hypertension': 2, 'anemia': 3}
        
        input_df['gender'] = input_df['gender'].map(gender_map)
        input_df['dietary_habits'] = input_df['dietary_habits'].map(dietary_map)
        input_df['physical_activity'] = input_df['physical_activity'].map(activity_map)
        input_df['sleep_pattern'] = input_df['sleep_pattern'].map(sleep_map)
        input_df['stress_level'] = input_df['stress_level'].map(stress_map)
        input_df['health_goal'] = input_df['health_goal'].map(goal_map)
        input_df['existing_condition'] = input_df['existing_condition'].map(condition_map)
        
        # Make prediction
        prediction = nb_model.predict(input_df)
        recommended_vitamin = target_encoder.inverse_transform(prediction)[0]
        
        return render_template('index.html', 
                             prediction_text=f'Recommended Vitamin: {recommended_vitamin}')
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)