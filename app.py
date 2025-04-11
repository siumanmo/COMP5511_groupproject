from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from pathlib import Path
import sys
import traceback

app = Flask(__name__)

# ======================
# 1. ENHANCED CONFIGURATION
# ======================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

# Debug mode - set to False in production
DEBUG = True

def debug_print(*args, **kwargs):
    """Print debug messages to stderr"""
    if DEBUG:
        print(*args, file=sys.stderr, **kwargs)

# ======================
# 2. SUPER ROBUST MODEL LOADING
# ======================
def load_models(max_retries=3):
    """Load models with retries and comprehensive validation"""
    retry_count = 0
    last_exception = None
    
    while retry_count < max_retries:
        try:
            debug_print("\n=== ATTEMPTING MODEL LOAD ===")
            
            # 1. Verify directory structure
            if not MODEL_DIR.exists():
                raise FileNotFoundError(f"Directory 'model' not found at {MODEL_DIR}")
            
            debug_print("✓ Model directory exists")
            
            # 2. Verify model files
            model_files = {
                'model': 'vitamin_recommender_nb.pkl',
                'encoder': 'target_encoder.pkl'
            }
            
            missing_files = []
            for name, filename in model_files.items():
                path = MODEL_DIR / filename
                if not path.exists():
                    missing_files.append(str(path))
                else:
                    size_kb = os.path.getsize(path) / 1024
                    debug_print(f"✓ {name.ljust(8)}: {path} ({size_kb:.1f} KB)")
            
            if missing_files:
                raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
            
            # 3. Load with validation
            model_path = MODEL_DIR / model_files['model']
            encoder_path = MODEL_DIR / model_files['encoder']
            
            debug_print("\nLoading model...")
            with open(model_path, 'rb') as f:
                model_data = joblib.load(f)
                model = model_data.get('model') if isinstance(model_data, dict) else model_data
            
            debug_print("Loading encoder...")
            with open(encoder_path, 'rb') as f:
                encoder = joblib.load(f)
            
            # 4. Validate loaded objects
            if not hasattr(model, 'predict'):
                raise AttributeError("Model object missing predict() method")
            
            if not hasattr(encoder, 'inverse_transform'):
                raise AttributeError("Encoder missing inverse_transform()")
            
            debug_print("\n✅ MODELS LOADED SUCCESSFULLY")
            debug_print(f"Model type: {type(model)}")
            debug_print(f"Encoder type: {type(encoder)}")
            
            return model, encoder
            
        except Exception as e:
            last_exception = e
            retry_count += 1
            debug_print(f"\n⚠️ Load failed (attempt {retry_count}/{max_retries}):")
            debug_print(traceback.format_exc())
            if retry_count < max_retries:
                debug_print("Retrying...")
    
    debug_print("\n❌ ALL LOAD ATTEMPTS FAILED")
    return None, None

# Initialize models with retries
model, target_encoder = load_models()

# ======================
# 3. APPLICATION CORE
# ======================
@app.route('/debug/files')
def debug_files():
    """Endpoint to check file system"""
    files = []
    for root, dirs, filenames in os.walk(BASE_DIR):
        for f in filenames:
            files.append(f"{root}/{f}")
    return {"files": files, "model_loaded": model is not None}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not all([model, target_encoder]):
            debug_print("\nCRITICAL: Models not loaded during prediction request!")
            return render_template('error.html',
                error="System initialization failed. Technical details have been logged.")
        
        try:
            # [Your existing prediction code here]
            return render_template('result.html', prediction="TEST")
        except Exception as e:
            debug_print("\nPrediction error:", traceback.format_exc())
            return render_template('error.html',
                error="Processing error. Please check your inputs.")
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))