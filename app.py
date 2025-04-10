from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Check if the model and encoder files exist
model_path = 'model/vitamin_recommender_nb.pkl'
encoder_path = 'model/target_encoder.pkl'

print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Encoder file exists: {os.path.exists(encoder_path)}")

# Load the model and the target encoder
try:
    loaded_model_and_encoder = joblib.load(model_path)
    
    # Determine if the model was saved as a dictionary or not
    if isinstance(loaded_model_and_encoder, dict):
        model = loaded_model_and_encoder['model']
    else:
        model = loaded_model_and_encoder  # direct model load

    # Load the target encoder, only if you aren't packing it with the model
    target_encoder = joblib.load(encoder_path)
    print(f"Model and encoder loaded successfully: {type(model)}, {type(target_encoder)}")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None
    target_encoder = None

# Define mappings for categorical variables
CATEGORY_MAPPINGS = {
    'gender': {'male': 0, 'female': 1},
    'diet': {'poor': 0, 'average': 1, 'good': 2, 'excellent': 3},
    'sun_exposure': {'low': 0, 'medium': 1, 'high': 2},
    'pregnant': {'no': 0, 'yes': 1},
    'smoker': {'no': 0, 'yes': 1},
    'activity_level': {'sedentary': 0, 'light': 1, 'moderate': 2, 'active': 3},
    'health_condition': {'poor': 0, 'fair': 1, 'good': 2, 'excellent': 3}
}

DEFAULTS = {
    'diet': CATEGORY_MAPPINGS['diet']['average'],
    'sun_exposure': CATEGORY_MAPPINGS['sun_exposure']['medium'],
    'pregnant': CATEGORY_MAPPINGS['pregnant']['no'],
    'smoker': CATEGORY_MAPPINGS['smoker']['no'],
    'health_condition': CATEGORY_MAPPINGS['health_condition']['good'],
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not model or not target_encoder:
            return render_template('error.html', error="Model or encoder not loaded")

        try:
            # Collecting user input and mapping categorical variables
            input_data = {
                'age': float(request.form['age']),
                'gender': CATEGORY_MAPPINGS['gender'][request.form['gender']],
                'diet': CATEGORY_MAPPINGS['diet'].get(request.form.get('diet', 'average'), DEFAULTS['diet']),
                'sun_exposure': CATEGORY_MAPPINGS['sun_exposure'].get(request.form.get('sun_exposure', 'medium'), DEFAULTS['sun_exposure']),
                'pregnant': CATEGORY_MAPPINGS['pregnant'].get(request.form.get('pregnant', 'no'), DEFAULTS['pregnant']),
                'smoker': CATEGORY_MAPPINGS['smoker'].get(request.form.get('smoker', 'no'), DEFAULTS['smoker']),
                'activity_level': CATEGORY_MAPPINGS['activity_level'][request.form['activity_level']],
                'health_condition': CATEGORY_MAPPINGS['health_condition'].get(request.form.get('health_condition', 'good'), DEFAULTS['health_condition']),
            }

            # Create DataFrame with the columns in the same order as training
            input_df = pd.DataFrame([input_data], columns=[
                'age',
                'gender',
                'diet',
                'sun_exposure',
                'pregnant',
                'smoker',
                'activity_level',
                'health_condition'
            ])
            
            prediction = model.predict(input_df)
            predicted_vitamin = target_encoder.inverse_transform(prediction)  # Decode the prediction to vitamin name
            output = f"Recommended Vitamin: {predicted_vitamin[0]}"  # Convert back to string
            
            return render_template('result.html', prediction=output)
        except Exception as e:
            return render_template('error.html', error=f"Prediction failed: {e}")
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)