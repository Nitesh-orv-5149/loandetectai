import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample dataset (Replace with real data)
data = {
    'income': [40000, 60000, 80000, 30000, 75000, 50000],
    'credit_score': [650, 700, 750, 600, 720, 680],
    'loan_amount': [10000, 20000, 30000, 5000, 25000, 15000],
    'loan_term': [5, 10, 15, 3, 12, 7],
    'debt_to_income': [35, 25, 20, 50, 22, 30],
    'loan_approved': [0, 1, 1, 0, 1, 0]  # 1 = Approved, 0 = Rejected
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Features & target
X = df[['income', 'credit_score', 'loan_amount', 'loan_term', 'debt_to_income']]
y = df['loan_approved']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ AI Model Trained & Saved as loan_model.pkl")
