import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

try:
    model = joblib.load('fraud_detection_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler files not found. Make sure they are in the same directory.")
    # Handle the error appropriately, maybe exit or use a default model
    model = None
    scaler = None

model_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud',
                 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives user input, preprocesses it, and returns the model's prediction."""
    if model is None or scaler is None:
        return render_template('index.html', prediction_text='Error: Model not loaded.')

    try:

        input_features = {
            'step': int(request.form['step']),
            'amount': float(request.form['amount']),
            'oldbalanceOrg': float(request.form['oldbalanceOrg']),
            'newbalanceOrig': float(request.form['newbalanceOrig']),
            'oldbalanceDest': float(request.form['oldbalanceDest']),
            'newbalanceDest': float(request.form['newbalanceDest']),
            'isFlaggedFraud': int(request.form['isFlaggedFraud']),
            'type': request.form['type']
        }

        input_df = pd.DataFrame([input_features])

        input_df = pd.get_dummies(input_df, columns=['type'], prefix='type')

        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]

        final_features = scaler.transform(input_df)

        prediction = model.predict(final_features)
        prediction_proba = model.predict_proba(final_features)

        if prediction[0] == 1:
            output = f"FRAUD (Confidence: {prediction_proba[0][1]:.2%})"
        else:
            output = f"NOT FRAUD (Confidence: {prediction_proba[0][0]:.2%})"

        return render_template('index.html', prediction_text=f'Transaction is predicted as: {output}')

    except Exception as e:

        return render_template('index.html', prediction_text=f'Error processing input: {e}')


if __name__ == "__main__":
    # Run the app
    app.run(debug=True)
