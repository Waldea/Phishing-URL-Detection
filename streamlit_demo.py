import streamlit as st
import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
import joblib
from bulk_url_processor import URLFeatureExtractor
from deployment_pipeline import MLModelPipeline

# Apply nest_asyncio to allow for re-entrance in the event loop in Streamlit
nest_asyncio.apply()

# Load the trained pipeline
@st.cache(allow_output_mutation=True)
def load_model():
    # Load the trained Random Forest pipeline
    deployment_model = MLModelPipeline(model=RandomForestClassifier())
    deployment_model.load_pipeline('Random Forest_deployment.joblib')
    return deployment_model

# Async function to extract features for a single URL
async def extract_features_for_url(url, ref_urls, session):
    extractor = URLFeatureExtractor(
        url=url,
        ref_urls_csv=ref_urls,
        session=session,
        perform_live_check=True,
        max_retries=3,
        request_timeout=30
    )
    features = await extractor.extract_all_features()
    return features

# Main Streamlit app
def main():
    st.title("Phishing URL Detection with Feature Extraction")
    st.write("Enter a URL to determine if it is likely to be benign or malignant.")
    
    # User input
    user_url = st.text_input("Enter URL:", "")
    
    # Predict button
    if st.button("Predict"):
        if user_url:
            # Prepare for feature extraction
            ref_urls = []  # Replace with the actual list or path to CSV containing reference URLs

            # Extract features asynchronously
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Create a shared aiohttp session for making requests
                async with aiohttp.ClientSession() as session:
                    features = loop.run_until_complete(extract_features_for_url(user_url, ref_urls, session))

                if features:
                    # Convert features to DataFrame
                    features_df = pd.DataFrame([features])

                    # Fill missing values after feature extraction
                    features_df['url_similarity_score'] = features_df['url_similarity_score'].fillna(features_df['url_similarity_score'].median())
                    features_df['registration_duration'] = features_df['registration_duration'].fillna(0)

                    # Load the model and predict
                    model_pipeline = load_model()
                    predictions = model_pipeline.predict(features_df)
                    # Map the predictions to 'benign' and 'malignant'
                    predictions_mapped = ['benign' if pred == 0 else 'malignant' for pred in predictions]

                    # Display result
                    st.write(f"Prediction: **{predictions_mapped[0]}**")

                    # If probabilistic predictions are required
                    if hasattr(model_pipeline.best_pipeline.named_steps['classifier'], 'predict_proba'):
                        prob_predictions = model_pipeline.predict_proba(features_df)
                        st.write(f"Probability (Benign: {prob_predictions[0][0]:.2f}, Malignant: {prob_predictions[0][1]:.2f})")
                else:
                    st.error("Failed to extract features from the URL.")
                    
            except Exception as e:
                st.error(f"Error occurred while making prediction: {e}")
        else:
            st.warning("Please enter a URL to predict.")

if __name__ == "__main__":
    main()
