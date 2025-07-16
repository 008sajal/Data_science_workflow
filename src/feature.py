from sklearn.preprocessing import LabelEncoder

def encode_categoricals(df):
    le = LabelEncoder()
    for col in ["employment_type", "required_experience", "required_education", "industry", "function"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = le.fit_transform(df[col])
    return df
