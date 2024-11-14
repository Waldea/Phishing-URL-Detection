from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import asyncio
from url_processor import URLFeatureExtractor  

# Load the trained model
model = joblib.load('SupportVectorMachine.joblib')

# Initialize the Flask application
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')

# Define the URL detection route
@app.route('/url_detection')
def url_detection():
    return render_template('url_detection.html')

# Define the route to check the URL
@app.route('/check_url', methods=['POST', 'GET'])
async def check_url():
    if request.method == 'POST':
        url = request.form['url']
        
        # Initialize the feature extractor
        extractor = URLFeatureExtractor(url)
        
        # Extract features asynchronously
        features = await extractor.extract_all_features()
        
        # Convert features to DataFrame for model prediction
        features_df = pd.DataFrame([features])

        # Predict using the loaded model
        prediction = model.predict(features_df)[0]
        
        # Determine if URL is phishing or not
        result = 'Phishing URL' if prediction == 1 else 'Legitimate URL'
        
        # Render the results in the template
        return render_template('url_detection.html', url=url, result=result)
    
    return redirect(url_for('url_detection'))

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)