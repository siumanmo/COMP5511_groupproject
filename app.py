from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model_pkg = joblib.load('models/vitamin_recommender_nb.pkl')
model = model_pkg['model']
label_encoders = model_pkg['label_encoders']
target_encoder = model_pkg['target_encoder']
class_names = model_pkg['class_names']

@app.route('/')
def home():
    return render_template('index.html', 
                         vitamins=class_names,
                         features=model_pkg['feature_names'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.json
        
        # Convert to DataFrame with correct feature order
        input_df = pd.DataFrame([form_data], columns=model_pkg['feature_names'])
        
        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        
        # Predict
        proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        
        # Format results
        results = {
            'recommendation': target_encoder.inverse_transform([prediction])[0],
            'probabilities': {
                class_names[i]: f"{p*100:.1f}%" 
                for i, p in enumerate(proba)
            },
            'input_summary': form_data
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)