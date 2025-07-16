# 🕵️‍♀️ Fake Job Postings Prediction

# Objective

This project tackles the problem of detecting **fake job postings** using machine learning. Given a non-standard, real-world dataset, we cleaned, preprocessed, and trained multiple models to classify whether a job posting is fraudulent or not.

---

## 👨‍💻 Team Members

- Sajal Vatsayan
- Ashritha Shreedhara Udupa
- Farzana

---

## 📂 Project Structure

fake-job-classification/
│
├── data/ # Raw data from Kaggle
│ └── fake_job_postings.csv
├── Data/splits/ # Train/Val/Test split files
│ ├── train.csv
│ ├── val.csv
│ └── test.csv
│

│
├── notebooks/
│ ├── eda.ipynb # Data exploration & cleaning
│ 
│
 
│
├── src/
│ ├── split_data.py # Script to split the dataset
│ ├── model_logistic.py # Logistic regression training
│ ├── model_nn.py # Neural network 
│ └── model_random_forest.py# Random Forest training
│
├── venv/ # Virtual environment
├── requirements.txt
└── README.md


---

## 📊 Dataset

- **Source:** [Kaggle - Fake Job Postings Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Description:** A real-world dataset of job postings with labels indicating whether they are real or fraudulent.
- **Non-standardness:** The dataset is noisy, unbalanced, and requires text cleaning, making it suitable for real-world classification tasks.

---

## 🧼 Data Preprocessing

We performed the following:
- Removed missing values from `description` and `fraudulent` columns
- Combined `title`, `location`, and `description` into a single `text` field 
- Removed punctuation and lowercased text
- Tokenized and applied TF-IDF vectorization

---

## 🧪 Models Trained

| Model              | Description                         |
|-------------------|-------------------------------------|
| Logistic Regression | Baseline linear model               |
| Neural Network (PyTorch) | Feedforward network with ReLU, Adam |
| Random Forest      | Non-linear ensemble tree-based model|

Each model was:
- Trained on the `train.csv` split
- Tuned on `val.csv`
- Evaluated finally on `test.csv` (only once)

---

## 🔧 Hyperparameter Tuning

- Done manually using validation set.
- Learning rate and hidden layers adjusted for the neural network.
- Number of estimators tuned for Random Forest.

---

## 📈 Evaluation Metrics

Each model was evaluated using:

- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

See outputs in the `output/` folder.

---

## 🥇 Results

| Model              | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Logistic Regression | 0.98     | 0.69   | 0.77     |
| Neural Network      | 0.94     | 0.84   | 0.89     |
| Random Forest       | 0.97     | 0.74   | 0.82     |

📌 **Best Model**: Random Forest

---


## Final Model Selection
After comparing Logistic Regression, a Neural Network (PyTorch), and Random Forest, we selected Random Forest for final evaluation because it consistently outperformed the others on F1-score, precision, and recall on the validation set. It also handled the TF-IDF features more robustly and was less sensitive to hyperparameters and training noise than the Neural Network.

## 📂 How to Run

### 1. Install Dependencies

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Prepare data
jupyter notebook notebooks/eda.ipynb

# Train models
python src/model_logistic_regression.py
python src/model_nn.py
python src/model_random_forest.py

# Final test evaluation
python src/final_test_evaluation.py


