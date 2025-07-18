import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import os

# Load training data
train_df = pd.read_csv("data/splits/train.csv")

# Combine relevant text columns
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    train_df[col] = train_df[col].fillna('')
train_df['combined_text'] = train_df[text_cols].agg(' '.join, axis=1)

X = train_df['combined_text']
y = train_df['fraudulent']

# Pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Hyperparameter grid
param_grid = {
    'clf__C': [0.1, 1.0, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs']
}

# Grid search with 3-fold cross-validation
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
grid.fit(X, y)

print("✅ Best Hyperparameters:")
print(grid.best_params_)

print(f"\n✅ Best F1 Score from GridSearchCV: {grid.best_score_:.4f}")

print("\n✅ Best Model Pipeline:")
print(grid.best_estimator_)
