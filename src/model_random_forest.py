# src/model_random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Paths
TRAIN_PATH = "Data/splits/train.csv"
VAL_PATH = "Data/splits/val.csv"

# Load training and validation data
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

# Create combined text column
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    train_df[col] = train_df[col].fillna('')
    val_df[col] = val_df[col].fillna('')

train_df['combined_text'] = train_df[text_columns].agg(' '.join, axis=1)
val_df['combined_text'] = val_df[text_columns].agg(' '.join, axis=1)

# Define text and label columns
TEXT_COL = "combined_text"
LABEL_COL = "fraudulent"

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train = vectorizer.fit_transform(train_df[TEXT_COL])
y_train = train_df[LABEL_COL]

X_val = vectorizer.transform(val_df[TEXT_COL])
y_val = val_df[LABEL_COL]

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on validation set
y_pred = rf_model.predict(X_val)

# Evaluate
print(" Random Forest Evaluation on Validation Set:\n")
print(classification_report(y_val, y_pred, digits=4))
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
