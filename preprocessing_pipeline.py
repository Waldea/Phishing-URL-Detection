import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_curve, det_curve, auc
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron
from typing import Dict

# Assuming PreprocessingPipeline is already defined elsewhere in your code

class MLModelPipeline:
    def __init__(self, model, param_grid=None, n_components=0.95):
        """
        Initialize the MLModelPipeline with preprocessing, model, and optional hyperparameter grid.
        
        Args:
            model: The machine learning model to train.
            param_grid: Hyperparameter grid for model tuning.
            n_components: Number of components for PCA, either float (percentage of explained variance) or int.
        """
        self.model = model
        self.param_grid = param_grid
        self.n_components = n_components
        self.pipeline = None
        self.best_pipeline = None
        self._build_pipeline()

    def _build_pipeline(self):
        # Define preprocessing pipeline
        preprocessing = PreprocessingPipeline(n_components=self.n_components)

        # Create full modeling pipeline
        self.pipeline = Pipeline([
            ("preprocessor", preprocessing),
            ("classifier", self.model)
        ])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, cv=5, n_iter=10):
        """Train the model with optional hyperparameter tuning."""
        if self.param_grid:
            print(f"Performing hyperparameter search for {self.model.__class__.__name__}...")
            search = RandomizedSearchCV(
                self.pipeline, param_distributions=self.param_grid, n_iter=n_iter,
                cv=cv, scoring="f1_macro", random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            self.best_pipeline = search.best_estimator_
            print(f"Best Parameters: {search.best_params_}")
        else:
            self.best_pipeline = self.pipeline.fit(X_train, y_train)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, set_name="Test") -> Dict:
        """Evaluate the model on a given dataset and print the classification report."""
        y_pred = self.best_pipeline.predict(X)
        test_f1_score = f1_score(y, y_pred, average="macro")
        print(f"\nClassification Report for {self.model.__class__.__name__} on {set_name} Set:")
        print(classification_report(y, y_pred, zero_division=0))
        print(f"{set_name} F1 Score: {test_f1_score:.4f}")
        return {
            "y_true": y,
            "y_pred": y_pred,
            "test_f1_score": test_f1_score
        }

    def save_pipeline(self, file_name: str):
        """Save the trained pipeline to disk."""
        if self.best_pipeline:
            joblib.dump(self.best_pipeline, file_name)
            print(f"Pipeline saved to {file_name}")
        else:
            print("No trained pipeline to save. Please train the model first.")

    def load_pipeline(self, file_name: str):
        """Load the pipeline from disk."""
        self.best_pipeline = joblib.load(file_name)
        print(f"Pipeline loaded from {file_name}")

    def predict(self, X: pd.DataFrame):
        """Make predictions on new data."""
        if self.best_pipeline:
            return self.best_pipeline.predict(X)
        else:
            raise ValueError("The pipeline is not trained. Train the model or load a trained pipeline first.")

    def predict_proba(self, X: pd.DataFrame):
        """Make probabilistic predictions on new data if supported by the model."""
        if self.best_pipeline and hasattr(self.best_pipeline, "predict_proba"):
            return self.best_pipeline.predict_proba(X)
        else:
            raise ValueError("The pipeline is not trained or model does not support predict_proba.")

#Example Usage
"""
# Models and Hyperparameters
models = {
    "Dummy Classifier": DummyClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(),
    "XGBoost": XGBClassifier(),
    "Simple Perceptron": Perceptron(max_iter=1000, tol=1e-3, random_state=42)
}

param_grids = {
    "Logistic Regression": {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['lbfgs', 'liblinear'],
        'classifier__class_weight': [None, 'balanced']
    },
    "Random Forest": {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__class_weight': [None, 'balanced']
    },
    "Gradient Boosting": {
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
    },
    "Support Vector Machine": {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'poly'],
        'classifier__probability': [True]
    },
    "XGBoost": {
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
    },
    "Simple Perceptron": {
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__penalty': ['l2', 'l1', 'elasticnet'],
    }
}

# Train and Evaluate Models
model_results = {}
roc_curves = {}
det_curves = {}

for model_name, model in models.items():
    print(f"Training {model_name} with PCA...")
    ml_pipeline = MLModelPipeline(model=model, param_grid=param_grids.get(model_name))
    
    # Train the model
    ml_pipeline.train(X_train, y_train_encoded)
    
    # Evaluate the model on the training set to check overfitting/underfitting
    train_results = ml_pipeline.evaluate(X_train, y_train_encoded, set_name="Train")
    
    # Cross-validation score on training data
    cv_score = cross_val_score(ml_pipeline.best_pipeline, X_train, y_train_encoded, cv=3, scoring="f1_macro").mean()
    print(f"{model_name} Cross-Validation F1 Score: {cv_score:.4f}")
    
    # Evaluate the model on the test set
    test_results = ml_pipeline.evaluate(X_test, y_test_encoded, set_name="Test")
    
    # Save results for further analysis (ROC, DET plots, etc.)
    model_results[model_name] = {
        "train_f1_score": train_results["test_f1_score"],
        "test_f1_score": test_results["test_f1_score"],
        "cv_score": cv_score,
        "y_true": test_results["y_true"],
        "y_pred": test_results["y_pred"]
    }
    
    # Compare train and test F1 scores to detect overfitting/underfitting
    if train_results["test_f1_score"] > test_results["test_f1_score"] + 0.1:
        print(f"Warning: Potential Overfitting in {model_name} (Train F1: {train_results['test_f1_score']:.4f}, Test F1: {test_results['test_f1_score']:.4f})")
    elif train_results["test_f1_score"] < test_results["test_f1_score"] - 0.1:
        print(f"Warning: Potential Underfitting in {model_name} (Train F1: {train_results['test_f1_score']:.4f}, Test F1: {test_results['test_f1_score']:.4f})")
    else:
        print(f"{model_name} appears to have a good fit.")
    
    # ROC and DET data collection
    if hasattr(ml_pipeline.best_pipeline.named_steps['classifier'], 'predict_proba'):
        y_prob = ml_pipeline.best_pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(test_results['y_true'], y_prob)
        fnr, fpr_det, _ = det_curve(test_results['y_true'], y_prob)
        roc_curves[model_name] = (fpr, tpr)
        det_curves[model_name] = (fpr_det, fnr)

    print("\n" + "="*50 + "\n")

# Combined ROC Plot
plt.figure(figsize=(10, 8))
for model_name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves for All Models')
plt.legend(loc='best')
plt.show()

# Combined DET Plot
plt.figure(figsize=(10, 8))
for model_name, (fpr_det, fnr) in det_curves.items():
    plt.plot(fpr_det, fnr, label=f'{model_name}')
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title('Combined DET Curves for All Models')
plt.legend(loc='best')
plt.show()

# Bar Chart Comparing F1 Scores
train_f1_scores = [results['train_f1_score'] for results in model_results.values()]
test_f1_scores = [results['test_f1_score'] for results in model_results.values()]
cv_scores = [results['cv_score'] for results in model_results.values()]

model_names = list(model_results.keys())

x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, train_f1_scores, width, label='Train F1 Score')
rects2 = ax.bar(x, test_f1_scores, width, label='Test F1 Score')
rects3 = ax.bar(x + width, cv_scores, width, label='CV F1 Score')

ax.set_xlabel('Models')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Scores by Model and Dataset')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()

fig.tight_layout()
plt.show()


# Example usage of deployment with a dataset called label_df
print("\nDeployment Example with label_df:\n")
deployment_model = MLModelPipeline(model=RandomForestClassifier())
deployment_model.load_pipeline("/content/Random Forest_deployment.joblib")  # Load the trained pipeline

# Predicting using the deployment pipeline
predictions = deployment_model.predict(label_df)
# Map the predictions to 'benign' and 'malignant'
predictions_mapped = ['benign' if pred == 0 else 'malignant' for pred in predictions]
print(f"Predictions: {predictions_mapped}")

# If probabilistic predictions are required
if hasattr(deployment_model.best_pipeline.named_steps['classifier'], 'predict_proba'):
    prob_predictions = deployment_model.predict_proba(label_df)
    print(f"Probabilistic Predictions: {prob_predictions}")
'''