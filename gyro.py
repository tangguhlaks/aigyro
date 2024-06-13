from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import os
import json
from datetime import datetime


app = Flask(__name__)

# Load CSV files for walk
gyro_back_path_walk = 'training/walk/gyro_back.csv'
gyro_front_path_walk = 'training/walk/gyro_front.csv'

gyro_back_data_walk = pd.read_csv(gyro_back_path_walk)
gyro_front_data_walk = pd.read_csv(gyro_front_path_walk)

# Load CSV files for stand
gyro_back_path_stand = 'training/stand/gyro_back.csv'
gyro_front_path_stand = 'training/stand/gyro_front.csv'

gyro_back_data_stand = pd.read_csv(gyro_back_path_stand)
gyro_front_data_stand = pd.read_csv(gyro_front_path_stand)

# Load CSV files for stairDown
gyro_back_path_stair_down = 'training/stairDown/gyro_back.csv'
gyro_front_path_stair_down = 'training/stairDown/gyro_front.csv'

gyro_back_data_stair_down = pd.read_csv(gyro_back_path_stair_down)
gyro_front_data_stair_down = pd.read_csv(gyro_front_path_stair_down)

# Combine the datasets for walk, stand, and stairDown
gyro_data = pd.concat([gyro_back_data_walk, gyro_front_data_walk, gyro_back_data_stand, gyro_front_data_stand, gyro_back_data_stair_down, gyro_front_data_stair_down], ignore_index=True)

# Preprocess data
# Assuming the label column is 'label' and features are 'x', 'y', 'z'
X_train = gyro_data[['x', 'y', 'z']].values
y_train = gyro_data['label'].values

# Initialize and train the model
model = GaussianNB()
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
            "predict" : predicted_class[0]
        })
    except KeyError as e:
        return jsonify({"error": f"Missing key in input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
