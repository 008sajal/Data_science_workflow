import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("data/fake_job_postings.csv")

print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head(2))

# drop irrelevant column
drop_cols = ['job_id', 'telecommuting', 'has_company_logo', 'has_questions']
df.drop(columns=drop_cols, inplace=True)

# Handle missing values
print(df.isnull().sum())

df.drop(columns=['salary_range'], inplace=True)
df.dropna(subset=['title', 'location', 'description'], inplace=True)

# Fill other fields
df['company_profile'].fillna('', inplace=True)
df['requirements'].fillna('', inplace=True)
df['benefits'].fillna('', inplace=True)

#Remove HTML tags and normalize text.

def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()  # Remove HTML
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip().lower()

text_cols = ['description', 'company_profile', 'requirements', 'benefits']

for col in text_cols:
    df[col] = df[col].apply(clean_text)


# Convert columns like employment_type, required_experience, industry, etc. into numerical format.

cat_cols = ['employment_type', 'required_experience', 'required_education',
            'industry', 'function', 'location', 'department']

for col in cat_cols:
    df[col] = df[col].fillna('Unknown')  # Fill missing with placeholder
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# Remove exact duplicates
df.drop_duplicates(inplace=True)

# Remove empty or nonsense descriptions
df = df[df['description'].str.len() > 50]

df.to_csv("data/cleaned_fake_jobs.csv", index=False)
print("Cleaned dataset saved.")

