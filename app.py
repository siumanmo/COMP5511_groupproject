from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)

# ======================
# 1. CONFIGURATION
# ======================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

# Debugging - Print directory structure
print("\n=== CURRENT DIRECTORY STRUCTURE ===")
for root, dirs, files in os.walk(BASE_DIR):
    print(f"{root.replace(str(BASE_DIR), '')}/")
    for file in files:
        print(f"  - {file}")

# ======================
# 2. MODEL LOADING
# ======================
def load_models():
    """Safely load models with extensive error handling"""
    try:
        # Verify model directory exists
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}")
        
        # Verify model files exist
        required_files = {
            "model": MODEL_DIR / "vitamin_recommender_nb.pkl",
            "encoder": MODEL_DIR / "target_encoder.pkl"
        }
        
        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} file at {path}")
        
        # Load files
        model_data = joblib.load(required_files["model"])
        encoder = joblib.load(required_files["encoder"])
        
        # Handle different model formats
        model = model_data['model'] if isinstance(model_data, dict) else model_data
        
        print("\n=== MODEL LOAD SUCCESS ===")
        print(f"Model type: {type(model)}")
        print(f"Encoder type: {type(encoder)}")
        
        return model, encoder
        
    except Exception as e:
        print(f"\n=== LOAD FAILED ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        return None, None

# Initialize models
model, target_encoder = load_models()

# ======================
# 3. APPLICATION LOGIC
# ======================
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
    'diet': 'average',
    'sun_exposure': 'medium',
    'pregnant': 'no',
    'smoker': 'no',
    'health_condition': 'good'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not model or not target_encoder:
            return render_template('error.html',
                error="System initialization failed. Contact support.")
        
        try:
            # Process form data
            form_data = {
                'age': float(request.form['age']),
                'gender': request.form['gender'],
                'diet': request.form.get('diet', DEFAULTS['diet']),
                'sun_exposure': request.form.get('sun_exposure', DEFAULTS['sun_exposure']),
                'pregnant': request.form.get('pregnant', DEFAULTS['pregnant']),
                'smoker': request.form.get('smoker', DEFAULTS['smoker']),
                'activity_level': request.form['activity_level'],
                'health_condition': request.form.get('health_condition', DEFAULTS['health_condition'])
            }
            
            # Convert to model format
            model_input = {
                'age': form_data['age'],
                'gender': CATEGORY_MAPPINGS['gender'][form_data['gender']],
                'diet': CATEGORY_MAPPINGS['diet'][form_data['diet']],
                'sun_exposure': CATEGORY_MAPPINGS['sun_exposure'][form_data['sun_exposure']],
                'pregnant': CATEGORY_MAPPINGS['pregnant'][form_data['pregnant']],
                'smoker': CATEGORY_MAPPINGS['smoker'][form_data['smoker']],
                'activity_level': CATEGORY_MAPPINGS['activity_level'][form_data['activity_level']],
                'health_condition': CATEGORY_MAPPINGS['health_condition'][form_data['health_condition']]
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([model_input])
            
            # Make prediction
            prediction = model.predict(input_df)
            vitamin = target_encoder.inverse_transform(prediction)[0]
            
            return render_template('result.html', prediction=vitamin)
            
        except Exception as e:
            return render_template('error.html',
                error=f"Prediction error: {str(e)}")
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))