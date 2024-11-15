import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder
from typing import List

class PreprocessingPipeline:
    def __init__(self, n_components=0.95):
        """
        Initialize the PreprocessingPipeline with different processing steps.

        Args:
            n_components: The number of components for PCA, can be a float (percentage of explained variance)
                          or an integer (number of components). Default is 0.95.
        """
        self.pipeline = None
        self.n_components = n_components
        self._build_pipeline()

    def _build_pipeline(self):
        # Define feature groups
        self.boolean_features = ['has_social_net', 'url_is_random', 'title_is_random', 'description_is_random',
                                 'has_brand_name_in_domain', 'is_expired', 'is_website_live', 'has_redirect']

        self.low_cardinality_categorical_features = ['title_similarity_bin', 'description_similarity_bin',
                                                     'similarity_bin', 'expiration_risk']

        self.high_cardinality_categorical_features = ['url', 'registration_type', 'title', 'description', 'tld', 'domain']

        self.numeric_features = ['url_length', 'domain_length', 'is_https', 'num_subdomains', 'num_subdirectories',
                                 'num_query_params', 'path_length', 'num_slashes', 'domain_entropy', 'char_repetition',
                                 'has_ip_address', 'shortened_url', 'has_hyphen', 'contains_homograph_chars',
                                 'domain_age', 'days_to_expiry', 'total_links', 'external_links', 'digit_ratio_in_url',
                                 'common_phishing_words', 'typosquatting_distance', 'path_suspicious_keywords',
                                 'query_suspicious_keywords', 'title_description_similarity', 'url_title_match_score',
                                 'url_similarity_score', 'registration_duration']

        # Preprocessing for different column types
        preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('target_enc', TargetEncoder(), self.high_cardinality_categorical_features),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), self.low_cardinality_categorical_features),
                #('imputer', SimpleImputer(strategy='mean'), self.numeric_features),  # Impute missing values in numeric features
                ('scaler', StandardScaler(), self.numeric_features)  # Scale numeric features
            ], remainder='passthrough'  # Keep boolean features as-is
        )

        # Building the complete pipeline with PCA
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessing_pipeline),
            ('pca', PCA(n_components=self.n_components))  # Add PCA for dimensionality reduction
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the preprocessing pipeline to the training data."""
        # Fit the pipeline using X_train and y_train for target encoding
        self.pipeline.fit(X_train, y_train)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the given dataset using the fitted pipeline."""
        transformed_data = self.pipeline.transform(X)
        # After PCA, the number of features changes, so we generate new feature names
        feature_names = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
        return pd.DataFrame(transformed_data, columns=feature_names, index=X.index)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Fit and transform the training data."""
        transformed_data = self.pipeline.fit_transform(X_train, y_train)
        # After PCA, the number of features changes, so we generate new feature names
        feature_names = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
        return pd.DataFrame(transformed_data, columns=feature_names, index=X_train.index)

    def get_feature_names_out(self) -> List[str]:
        """Get feature names after transformation (before PCA)."""
        preprocessor = self.pipeline.named_steps['preprocessor']
        if hasattr(preprocessor, 'get_feature_names_out'):
            return list(preprocessor.get_feature_names_out())
        else:
            # Manually extract feature names if get_feature_names_out is not available
            feature_names = []
            for name, transformer, columns in preprocessor.transformers:
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out())
                else:
                    feature_names.extend(columns)
            return feature_names
