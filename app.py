from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from pathlib import Path
import sys

app = Flask(__name__)

# ======================
# 1. ENHANCED CONFIGURATION
# ======================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

# Debugging setup
DEBUG = True  # Set to False in production

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, file=sys.stderr, **kwargs)

debug_print("\n=== STARTUP DEBUG INFORMATION ===")
debug_print(f"Python version: {sys.version}")
debug_print(f"Working directory: {os.getcwd()}")
debug_print("\n=== DIRECTORY STRUCTURE ===")

for root, dirs, files in os.walk(BASE_DIR):
    debug_print(f"{root.replace(str(BASE_DIR), '')}/")
    for file in files:
        debug_print(f"  - {file}")

# ======================
# 2. ROBUST MODEL LOADING
# ======================
def load_models():
    """Load models with comprehensive error handling and validation"""
    try:
        # Verify model directory
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model directory missing at {MODEL_DIR}")

        debug_print("\n=== MODEL FILES VERIFICATION ===")
        
        # Define required files with validation
        required_files = {
            "model": {
                "path": MODEL_DIR / "vitamin_recommender_nb.pkl",
                "min_size_kb": 10  # Minimum expected file size
            },
            "encoder": {
                "path": MODEL_DIR / "target_encoder.pkl",
                "min_size_kb": 1
            }
        }
        
        # Check each file
        for name, config in required_files.items():
            path = config["path"]
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} file at {path}")
            
            file_size = os.path.getsize(path) / 1024  # Size in KB
            if file_size < config["min_size_kb"]:
                raise ValueError(f"{name} file too small ({file_size:.1f}KB), possibly corrupted")
            
            debug_print(f"✓ {name.ljust(8)}: {path} ({file_size:.1f}KB)")

        # Load with explicit error handling
        debug_print("\n=== MODEL LOADING ===")
        
        with open(required_files["model"]["path"], 'rb') as f:
            model_data = joblib.load(f)
            model = model_data['model'] if isinstance(model_data, dict) else model_data
            debug_print(f"Model loaded: {type(model)}")
        
        with open(required_files["encoder"]["path"], 'rb') as f:
            encoder = joblib.load(f)
            debug_print(f"Encoder loaded: {type(encoder)}")

        # Validate loaded objects
        if not hasattr(model, 'predict'):
            raise AttributeError("Loaded model object missing predict() method")
        
        debug_print("\n✅ MODELS LOADED SUCCESSFULLY")
        return model, encoder
        
    except Exception as e:
        debug_print("\n❌ LOAD FAILED:", str(e))
        debug_print("Error type:", type(e).__name__)
        return None, None

# Initialize models
model, target_encoder = load_models()

# ======================
# 3. APPLICATION CORE
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

@app.route('/healthcheck')
def health_check():
    """Endpoint for deployment health monitoring"""
    status = {
        'model_loaded': model is not None,
        'encoder_loaded': target_encoder is not None,
        'system_ready': model is not None and target_encoder is not None
    }
    return status

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # System readiness check
        if not all([model, target_encoder]):
            debug_print("WARNING: Model not loaded during prediction request")
            return render_template('error.html',
                error="System not ready. Please try again later.")
        
        try:
            # Process form data with validation
            form_data = {
                'age': max(0, min(120, float(request.form['age']))),
                'gender': request.form['gender'],
                'diet': request.form.get('diet', DEFAULTS['diet']),
                'sun_exposure': request.form.get('sun_exposure', DEFAULTS['sun_exposure']),
                'pregnant': request.form.get('pregnant', DEFAULTS['pregnant']),
                'smoker': request.form.get('smoker', DEFAULTS['smoker']),
                'activity_level': request.form['activity_level'],
                'health_condition': request.form.get('health_condition', DEFAULTS['health_condition'])
            }

            # Convert to model format with validation
            try:
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
            except KeyError as e:
                raise ValueError(f"Invalid form value: {str(e)}")

            # Create DataFrame
            input_df = pd.DataFrame([model_input])
            
            # Make prediction
            prediction = model.predict(input_df)
            vitamin = target_encoder.inverse_transform(prediction)[0]
            
            debug_print(f"Prediction made: {vitamin}")
            return render_template('result.html', prediction=vitamin)
            
        except Exception as e:
            debug_print("Prediction error:", str(e))
            return render_template('error.html',
                error=f"Processing error: {str(e)}")
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))