from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model/vitamin_recommender_nb.pkl')

# Manual mapping (replace with your actual categories)
CATEGORY_MAPPINGS = {
    'gender': {'male': 0, 'female': 1},
    'diet': {'poor': 0, 'average': 1, 'good': 2, 'excellent': 3},
    # Add other columns as needed
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_data = {
                'age': float(request.form['age']),
                'gender': CATEGORY_MAPPINGS['gender'][request.form['gender']],
                'diet': CATEGORY_MAPPINGS['diet'][request.form['diet']],
                # Add other fields
            }
            
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            
            return render_template('result.html', 
                                supplement=prediction[0])
        
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)