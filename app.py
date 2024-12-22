from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    predicted_price = 0  # or any computed value
    return render_template('index.html', predicted_price=predicted_price)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    square_footage = int(request.form['square_footage'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    
    # Prepare the input features for prediction
    features = np.array([[square_footage, bedrooms, bathrooms]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict the house price
    predicted_price = model.predict(features_scaled)[0]
    
    # Render the result on the webpage
    return render_template('index.html', predicted_price=predicted_price)

@app.route("/clear", methods=["POST"])
def clear():
    # Redirect to reset the form
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)
