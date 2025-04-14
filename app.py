@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    logging.info(f"Received input data: {input_data}")

    # Extract health conditions
    health_conditions = request.form.getlist('health_conditions')
    if not health_conditions:
        health_conditions = ['none']  # Default to 'none' if no condition is selected

    # Combine selected health conditions into a string
    input_data['health_condition'] = ', '.join(health_conditions)

    # Set defaults for omitted fields if they are important for the model
    input_data.setdefault('sun_hours_per_week', 0)
    input_data.setdefault('pregnant', 0)
    input_data.setdefault('smoker', 0)  

    # Convert input data to DataFrame
    df_input = pd.DataFrame(input_data, index=[0])
    
    # Ensure numeric columns are in correct dtype
    numeric_columns = ['age', 'sun_hours_per_week', 'vitamin_d_level', 'pregnant', 'smoker']
    
    try:
        # Convert appropriate columns to numeric
        df_input[numeric_columns] = df_input[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Check if required columns are included
        required_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        if required_columns is not None:
            missing_columns = set(required_columns) - set(df_input.columns)
            if missing_columns:
                logging.error(f"Missing expected columns: {missing_columns}")
                return f"Error: Missing expected columns: {', '.join(missing_columns)}."

        # Log the DataFrame before prediction
        logging.info(f"Input DataFrame for prediction:\n{df_input}")

        # Making the prediction
        prediction = model.predict(df_input)
        logging.info(f"Prediction made: {prediction[0]}")
        return render_template('index.html', prediction=prediction[0])
    
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return "Error making prediction. Check logs for details."