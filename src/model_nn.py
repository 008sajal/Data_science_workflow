import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np

# 1. Load and preprocess the data
train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/val.csv")

# Fill missing values and create combined_text column
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    train_df[col] = train_df[col].fillna('')
    val_df[col] = val_df[col].fillna('')

train_df['combined_text'] = train_df[text_columns].agg(' '.join, axis=1)
val_df['combined_text'] = val_df[text_columns].agg(' '.join, axis=1)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_df['combined_text']).toarray()
X_val = vectorizer.transform(val_df['combined_text']).toarray()

y_train = train_df['fraudulent'].values.astype(np.float32)
y_val = val_df['fraudulent'].values.astype(np.float32)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# 2. PyTorch Dataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Define the PyTorch model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5000, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleNN()

# 4. Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
model.train()
for epoch in range(5):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

# 6. Evaluation
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val_tensor)
    y_pred_class = (y_pred_val > 0.5).int().numpy()

print("🔍 PyTorch Neural Network Performance on Validation Set:")
print(classification_report(y_val, y_pred_class, digits=4))
