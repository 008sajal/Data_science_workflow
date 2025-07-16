# final_test_random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Load train and test data
train_df = pd.read_csv("data/splits/train.csv")
test_df = pd.read_csv("data/splits/test.csv")

# Fill missing values and create combined_text
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    train_df[col] = train_df[col].fillna('')
    test_df[col] = test_df[col].fillna('')

train_df['combined_text'] = train_df[text_cols].agg(' '.join, axis=1)
test_df['combined_text'] = test_df[text_cols].agg(' '.join, axis=1)

# Features and labels
X_train = train_df['combined_text']
y_train = train_df['fraudulent']

X_test = test_df['combined_text']
y_test = test_df['fraudulent']

# TF-IDF + Random Forest (pipeline-like approach)
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)

print("✅ Final Test Set Evaluation (Random Forest):")
print(classification_report(y_test, y_pred, digits=4))
