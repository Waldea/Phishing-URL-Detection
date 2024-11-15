import streamlit as st
import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
import joblib
from bulk_url_processor import URLFeatureExtractor
from deployment_pipeline import MLModelPipeline
from sklearn.ensemble import RandomForestClassifier

# Apply nest_asyncio to allow for re-entrance in the event loop in Streamlit
nest_asyncio.apply()

# Load the trained pipeline
@st.cache(allow_output_mutation=True)
def load_model():
    # Load the trained Random Forest pipeline using the relative path
    deployment_model = MLModelPipeline(model=RandomForestClassifier())
    model_path = "data/models/Random Forest_deployment.joblib"
    deployment_model.load_pipeline(model_path)
    return deployment_model

# Async function to extract features for a batch of URLs
async def extract_features_in_batches(urls, ref_urls, session):
    extracted_features = []

    # Create tasks for each URL and process them concurrently
    tasks = []
    semaphore = asyncio.Semaphore(15)  # Limit concurrency to 15 requests at a time
    for url in urls:
        extractor = URLFeatureExtractor(
            url=url,
            ref_urls_csv=ref_urls,
            session=session,
            perform_live_check=True,
            max_retries=3,
            request_timeout=30
        )
        task = extract_features_for_url(extractor, semaphore)
        tasks.append(task)

    # Gather all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for features in results:
        if isinstance(features, Exception):
            st.error(f"Error during feature extraction: {features}")
        else:
            extracted_features.append(features)

    # Convert the list of feature dictionaries into a DataFrame
    features_df = pd.DataFrame(extracted_features)
    return features_df

# Helper function to extract features for a single URL using the semaphore
async def extract_features_for_url(extractor, semaphore):
    async with semaphore:
        try:
            # Extract features using the asynchronous method
            features = await extractor.extract_all_features()
        except Exception as e:
            st.error(f"Error extracting features for {extractor.url}: {e}")
            features = None
        return features

# Main Streamlit app
def main():
    st.title("Phishing URL Detection with Feature Extraction")
    st.write("Enter one or more URLs (one per line) to determine if they are likely to be benign or malicious.")
    
    # Multi-line text input to accept multiple URLs
    user_urls = st.text_area("Enter URLs (one per line):", height=200)

    # Predict button
    if st.button("Predict"):
        if user_urls.strip():
            # Split the input text into a list of URLs
            urls = [url.strip() for url in user_urls.splitlines() if url.strip()]
            
            # Prepare for feature extraction
            ref_urls = []  # Replace with the actual list or path to CSV containing reference URLs

            # Extract features asynchronously
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Create a shared aiohttp session for making requests
                async with aiohttp.ClientSession() as session:
                    features_df = loop.run_until_complete(extract_features_in_batches(urls, ref_urls, session))

                if not features_df.empty:
                    # Fill missing values after feature extraction
                    features_df['url_similarity_score'] = features_df['url_similarity_score'].fillna(features_df['url_similarity_score'].median())
                    features_df['registration_duration'] = features_df['registration_duration'].fillna(0)

                    # Load the model and predict
                    model_pipeline = load_model()
                    predictions = model_pipeline.predict(features_df)
                    # Map the predictions to 'benign' and 'malignant'
                    predictions_mapped = ['benign' if pred == 0 else 'malignant' for pred in predictions]

                    # Display results for each URL
                    st.write("Predictions:")
                    for url, prediction in zip(urls, predictions_mapped):
                        st.write(f"{url}: **{prediction}**")

                    # If probabilistic predictions are required
                    if hasattr(model_pipeline.best_pipeline.named_steps['classifier'], 'predict_proba'):
                        prob_predictions = model_pipeline.predict_proba(features_df)
                        st.write("Probabilities (Benign, Malignant):")
                        for url, probs in zip(urls, prob_predictions):
                            st.write(f"{url}: (Benign: {probs[0]:.2f}, Malignant: {probs[1]:.2f})")
                else:
                    st.error("Failed to extract features from the URLs.")
                    
            except Exception as e:
                st.error(f"Error occurred while making predictions: {e}")
        else:
            st.warning("Please enter at least one URL to predict.")

if __name__ == "__main__":
    main()
