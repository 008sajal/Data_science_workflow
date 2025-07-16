import pandas as pd
from sklearn.metrics import classification_report

# Load the validation set
val_df = pd.read_csv("data/splits/val.csv")

# Predict majority class
most_common = val_df['fraudulent'].mode()[0]
val_df['prediction'] = most_common

# Evaluate
print("Naive Baseline Performance:")
print(classification_report(val_df['fraudulent'], val_df['prediction']))
