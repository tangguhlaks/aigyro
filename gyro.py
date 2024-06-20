from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load CSV files for sit, stand, and walk
walk = pd.read_csv('training/walk.csv')
fall = pd.read_csv('training/fall.csv')
sit = pd.read_csv('training/sit.csv')
stand = pd.read_csv('training/stand.csv')

# Combine the datasets for walk, stand, and sit
gyro_data = pd.concat([stand, sit, walk,fall], ignore_index=True)
print(gyro_data)

# Preprocess data
# Assuming the label column is 'label' and features are 'x', 'y', 'z'
X_train = gyro_data[['x', 'y', 'z']].values
y_train = gyro_data['label'].values

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train, y_train)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request body
        data = request.get_json()
        data_dict = json.loads(data)

        # Extract relevant features from the received data
        X_sample = np.array([[data_dict['Accel-X'], data_dict['Accel-Y'], data_dict['Accel-Z']]])

        # Make prediction
        predicted_class = model.predict(X_sample)

        now = datetime.now()
        return jsonify({
            "datetime": now.strftime("%d-%m-%Y %H:%M:%S"),
            "predict": predicted_class[0]
        })
    except KeyError as e:
        return jsonify({"error": f"Missing key in input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
