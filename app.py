from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your model
with open('model/vitamin_recommender_nb.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Return the result
    return render_template('index.html', 
                          prediction_text='Recommended Vitamin: {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)