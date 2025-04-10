from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model with error handling
try:
    model_path = os.path.join('model', 'vitamin_recommender_nb.pkl')
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Complete category mappings for all your features
CATEGORY_MAPPINGS = {
    'gender': {'male': 0, 'female': 1},
    'diet': {'poor': 0, 'average': 1, 'good': 2, 'excellent': 3},
    'sun_exposure': {'low': 0, 'medium': 1, 'high': 2},
    'pregnant': {'no': 0, 'yes': 1},
    'smoker': {'no': 0, 'yes': 1},
    'activity_level': {
        'sedentary': 0,
        'lightly_active': 1,
        'moderately_active': 2,
        'very_active': 3
    },
    'health_condition': {
        'poor': 0,
        'fair': 1,
        'good': 2,
        'excellent': 3
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if model is None:
            return render_template('error.html', error="Model not loaded")
        
        try:
            # Process all form fields
            input_data = {
                'age': float(request.form['age']),
                'gender': CATEGORY_MAPPINGS['gender'][request.form['gender']],
                'diet': CATEGORY_MAPPINGS['diet'][request.form['diet']],
                'sun_exposure': CATEGORY_MAPPINGS['sun_exposure'][request.form['sun_exposure']],
                'pregnant': CATEGORY_MAPPINGS['pregnant'][request.form['pregnant']],
                'smoker': CATEGORY_MAPPINGS['smoker'][request.form['smoker']],
                'activity_level': CATEGORY_MAPPINGS['activity_level'][request.form['activity_level']],
                'health_condition': CATEGORY_MAPPINGS['health_condition'][request.form['health_condition']]
            }
            
            # Convert to DataFrame with consistent column order
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)
            
            return render_template('result.html', 
                                supplement=prediction[0],
                                user_input=request.form)
        
        except KeyError as e:
            return render_template('error.html', error=f"Missing field: {str(e)}")
        except ValueError as e:
            return render_template('error.html', error=f"Invalid value: {str(e)}")
        except Exception as e:
            return render_template('error.html', error=f"Prediction error: {str(e)}")
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)