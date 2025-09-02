from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model & scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Heart Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Input as JSON
    features = np.array([list(data.values())]).reshape(1, -1)
    features = scaler.transform(features)  # Scale input
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

