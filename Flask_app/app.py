from flask import Flask, render_template, request
import pickle
import numpy as np
import math

# Load model
model = pickle.load(open('rf_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Convert to array
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])

        # Prediction
        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f"Recommended Crop: {prediction[0]}")

    except Exception as e:
        return render_template('index.html', prediction_text="Error in input")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)