from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from pathlib import Path
import sys
import traceback

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DEBUG = True  # Set to False in production

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, file=sys.stderr, **kwargs)

# ======================
# SUPER ROBUST MODEL LOADER
# ======================
def load_models():
    """Load models with maximum error handling"""
    try:
        # 1. Verify directory structure
        debug_print("\n=== STARTING MODEL LOAD ===")
        debug_print(f"Base directory: {BASE_DIR}")
        
        if not MODEL_DIR.exists():
            debug_print(f"❌ Model directory missing at {MODEL_DIR}")
            debug_print("Current directory contents:")
            for f in BASE_DIR.iterdir():
                debug_print(f" - {f.name}")
            raise FileNotFoundError("Model directory not found")

        # 2. Verify model files exist
        model_files = {
            'model': MODEL_DIR / "vitamin_recommender_nb.pkl",
            'encoder': MODEL_DIR / "target_encoder.pkl"
        }

        debug_print("\n=== VERIFYING FILES ===")
        for name, path in model_files.items():
            if not path.exists():
                debug_print(f"❌ Missing {name} file at {path}")
                debug_print(f"Model directory contents: {os.listdir(MODEL_DIR)}")
                raise FileNotFoundError(f"Missing {name} file")
            debug_print(f"✓ Found {name} at {path}")

        # 3. Load with validation
        debug_print("\n=== LOADING MODELS ===")
        with open(model_files['model'], 'rb') as f:
            model_data = joblib.load(f)
            model = model_data.get('model') if isinstance(model_data, dict) else model_data
            debug_print(f"Model type: {type(model)}")

        with open(model_files['encoder'], 'rb') as f:
            encoder = joblib.load(f)
            debug_print(f"Encoder type: {type(encoder)}")

        # 4. Validate functionality
        if not hasattr(model, 'predict'):
            raise AttributeError("Model missing predict() method")
        if not hasattr(encoder, 'inverse_transform'):
            raise AttributeError("Encoder missing inverse_transform()")

        debug_print("\n✅ MODELS LOADED SUCCESSFULLY")
        return model, encoder

    except Exception as e:
        debug_print("\n❌ LOAD FAILED:")
        debug_print(traceback.format_exc())
        return None, None

# Initialize models
model, target_encoder = load_models()

# ======================
# DEBUG ENDPOINTS
# ======================
@app.route('/system-check')
def system_check():
    """Comprehensive system diagnostics"""
    status = {
        'model_loaded': model is not None,
        'encoder_loaded': target_encoder is not None,
        'files': {}
    }

    # Check model files
    model_files = ['vitamin_recommender_nb.pkl', 'target_encoder.pkl']
    for f in model_files:
        path = MODEL_DIR / f
        status['files'][f] = {
            'exists': path.exists(),
            'size': f"{os.path.getsize(path)/1024:.1f} KB" if path.exists() else None
        }

    # Add directory structure
    status['directory_structure'] = []
    for root, dirs, files in os.walk(BASE_DIR):
        status['directory_structure'].append({
            'path': str(Path(root).relative_to(BASE_DIR)),
            'files': files
        })

    return status

# ======================
# MAIN APPLICATION
# ======================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not all([model, target_encoder]):
            debug_print("\nCRITICAL: Models not loaded!")
            return render_template('error.html',
                error="System initialization failed. Visit /system-check for details.")
        
        try:
            # [Your existing prediction code here]
            return render_template('result.html', prediction="TEST")
        except Exception as e:
            debug_print("\nPrediction error:", traceback.format_exc())
            return render_template('error.html',
                error="Processing error. Please try again.")
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))