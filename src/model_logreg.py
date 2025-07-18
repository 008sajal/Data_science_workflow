import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/val.csv")

# Fill missing values
train_df['description'].fillna('', inplace=True)
val_df['description'].fillna('', inplace=True)

train_df['combined_text'] = (
    train_df['title'].fillna('') + ' ' +
    train_df['company_profile'].fillna('') + ' ' +
    train_df['description'].fillna('') + ' ' +
    train_df['requirements'].fillna('') + ' ' +
    train_df['benefits'].fillna('')
)

val_df['combined_text'] = (
    val_df['title'].fillna('') + ' ' +
    val_df['company_profile'].fillna('') + ' ' +
    val_df['description'].fillna('') + ' ' +
    val_df['requirements'].fillna('') + ' ' +
    val_df['benefits'].fillna('')
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train = vectorizer.fit_transform(train_df['combined_text'])
print(X_train.shape)
X_val = vectorizer.transform(val_df['combined_text'])

# Labels
y_train = train_df['fraudulent']
y_val = val_df['fraudulent']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)

# Evaluate
accuracy= accuracy_score(y_val,y_pred)
print(f"Accuracy: {accuracy: .2f}")

print("Logistic Regression Performance:")
print(classification_report(y_val, y_pred))

