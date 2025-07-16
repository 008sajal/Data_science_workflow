import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the cleaned dataset
data_path = 'Data/cleaned_fake_jobs.csv'
df = pd.read_csv(data_path)

# First split: Train vs Test+Val
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['fraudulent'])

# Second split: Validation vs Test (from the 30%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['fraudulent'])

# Save splits to CSV
os.makedirs("Data/splits", exist_ok=True)
train_df.to_csv("Data/splits/train.csv", index=False)
val_df.to_csv("Data/splits/val.csv", index=False)
test_df.to_csv("Data/splits/test.csv", index=False)

print(f"Train: {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows")
print(f"Test: {len(test_df)} rows")

print("✅ Data split complete: train, val, test saved in Data/splits/")
