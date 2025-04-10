from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")  # Your RF model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    age = int(request.form["age"])
    gender = request.form["gender"]
    diet = request.form["diet"]
    
    # Predict (example)
    input_data = pd.DataFrame([[age, gender, diet]], 
                            columns=["age", "gender", "diet"])
    prediction = model.predict(input_data)[0]
    
    return f"Recommended Vitamin: {prediction}"

if __name__ == "__main__":
    app.run(debug=True)