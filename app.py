from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models
model = joblib.load('heart_model.pkl')  # Changed to match your filename
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Grab data from form
        data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['chestPain']),
            float(request.form['bpm']),
            float(request.form['bloodSugar']),
            float(request.form['discomfort'])
        ]
        
        # Scale and predict
        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)

        result = 'High Risk of Heart Disease' if prediction[0] == 1 else 'Low Risk of Heart Disease'
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)