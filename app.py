from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)

# ======================
# 1. PATH CONFIGURATION
# ======================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

# Model file paths
MODEL_PATH = MODEL_DIR / "vitamin_recommender_nb.pkl"
ENCODER_PATH = MODEL_DIR / "target_encoder.pkl"

# ======================
# 2. MODEL LOADING
# ======================
def load_models():
    """Load model and encoder with error handling"""
    try:
        # Debug: List files in model directory
        print(f"Files in model/: {os.listdir(MODEL_DIR)}")
        
        # Load model (handle both dict-wrapped and direct models)
        loaded = joblib.load(MODEL_PATH)
        model = loaded['model'] if isinstance(loaded, dict) else loaded
        
        # Load encoder
        encoder = joblib.load(ENCODER_PATH)
        
        print("✅ Models loaded successfully!")
        return model, encoder
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return None, None

# Initialize models at startup
model, target_encoder = load_models()

# ======================
# 3. MAPPINGS & DEFAULTS
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
    'diet': CATEGORY_MAPPINGS['diet']['average'],
    'sun_exposure': CATEGORY_MAPPINGS['sun_exposure']['medium'],
    'pregnant': CATEGORY_MAPPINGS['pregnant']['no'],
    'smoker': CATEGORY_MAPPINGS['smoker']['no'],
    'health_condition': CATEGORY_MAPPINGS['health_condition']['good'],
}

# ======================
# 4. FLASK ROUTES
# ======================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not model or not target_encoder:
            return render_template('error.html', 
                error="System not ready. Models failed to load.")
        
        try:
            # Process form data
            input_data = {
                'age': float(request.form['age']),
                'gender': CATEGORY_MAPPINGS['gender'][request.form['gender']],
                'diet': CATEGORY_MAPPINGS['diet'].get(
                    request.form.get('diet', 'average'), 
                    DEFAULTS['diet']),
                'sun_exposure': CATEGORY_MAPPINGS['sun_exposure'].get(
                    request.form.get('sun_exposure', 'medium'), 
                    DEFAULTS['sun_exposure']),
                'pregnant': CATEGORY_MAPPINGS['pregnant'].get(
                    request.form.get('pregnant', 'no'), 
                    DEFAULTS['pregnant']),
                'smoker': CATEGORY_MAPPINGS['smoker'].get(
                    request.form.get('smoker', 'no'), 
                    DEFAULTS['smoker']),
                'activity_level': CATEGORY_MAPPINGS['activity_level'][
                    request.form['activity_level']],
                'health_condition': CATEGORY_MAPPINGS['health_condition'].get(
                    request.form.get('health_condition', 'good'), 
                    DEFAULTS['health_condition']),
            }

            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Predict
            prediction = model.predict(input_df)
            vitamin = target_encoder.inverse_transform(prediction)[0]
            
            return render_template('result.html', 
                prediction=f"Recommended Vitamin: {vitamin}")
            
        except Exception as e:
            return render_template('error.html', 
                error=f"Prediction failed: {str(e)}")
    
    return render_template('form.html')

# ======================
# 5. START APPLICATION
# ======================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))