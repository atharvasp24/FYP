from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained RandomForest model
RF_pkl_filename = 'RandomForest.pkl'
with open(RF_pkl_filename, 'rb') as file:
    RF = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    nitrogen = float(request.form['nitrogen'])
    phosphoros = float(request.form['phosphoros'])
    potassium = float(request.form['potassium'])
    temp = float(request.form['temp'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])

    # Make a prediction using the RandomForest model
    data = np.array([[nitrogen, phosphoros, potassium, temp, humidity, ph]])
    prediction = RF.predict(data)[0]

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
