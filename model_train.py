import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("datasets/fraudTrain.csv")
test_df = pd.read_csv("datasets/fraudTest.csv")

print("Train columns:", train_df.columns)
print("Test columns:", test_df.columns)


if "is_fraud" in train_df.columns:
    target_col = "is_fraud"
else:
    target_col = train_df.columns[-1]

print("Target column detected:", target_col)


X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]

X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

X_train = X_train.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully")
print("🎯 Accuracy:", acc)

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("💾 Model & Scaler saved successfully")
