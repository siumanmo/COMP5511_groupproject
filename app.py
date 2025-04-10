from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load('models/random_forest_model.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
target_encoder = joblib.load('models/target_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'age': int(request.form['age']),
                'gender': request.form['gender'],
                'diet': request.form['diet'],
                'sun_exposure': request.form['sun_exposure'],
                'pregnant': request.form['pregnant'],
                'smoker': request.form['smoker'],
                'activity_level': request.form['activity_level'],
                'health_condition': request.form['health_condition']
            }

            # Preprocess input
            input_df = pd.DataFrame([form_data])
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col])

            # Predict
            prediction = model.predict(input_df)[0]
            vitamin = target_encoder.inverse_transform([prediction])[0]
            probabilities = model.predict_proba(input_df)[0]

            # Format results
            vitamin_probs = {
                target_encoder.classes_[i]: f"{prob*100:.1f}%"
                for i, prob in enumerate(probabilities)
            }

            return render_template('index.html', 
                                result=vitamin,
                                probabilities=vitamin_probs,
                                form_data=form_data)

        except Exception as e:
            return render_template('index.html', error=str(e))

    # Default options for form
    form_options = {
        'gender': ['Male', 'Female', 'Other'],
        'diet': ['Vegetarian', 'Vegan', 'Omnivore', 'Pescatarian'],
        'sun_exposure': ['Low', 'Medium', 'High'],
        'pregnant': ['Yes', 'No'],
        'smoker': ['Yes', 'No'],
        'activity_level': ['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'],
        'health_condition': ['None', 'Diabetes', 'Hypertension', 'Anemia', 'Osteoporosis']
    }
    return render_template('index.html', options=form_options)

if __name__ == '__main__':
    app.run(debug=True)