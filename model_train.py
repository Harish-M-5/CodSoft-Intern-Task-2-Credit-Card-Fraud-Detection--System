import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load datasets
train_df = pd.read_csv("datasets/fraudTrain.csv")
test_df = pd.read_csv("datasets/fraudTest.csv")

print("Train columns:", train_df.columns)
print("Test columns:", test_df.columns)

# Target column (fraud dataset la usually 'is_fraud')
if "is_fraud" in train_df.columns:
    target_col = "is_fraud"
else:
    target_col = train_df.columns[-1]  # fallback

print("Target column detected:", target_col)

# Features & target
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]

X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

# ⚠️ Remove non-numeric columns (important for this dataset)
X_train = X_train.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully")
print("🎯 Accuracy:", acc)

# Save model & scaler
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("💾 Model & Scaler saved successfully")
