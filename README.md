## CodSoft-Intern-Task-2-Credit-Card-Fraud-Detection--System
 
---

## 🧾 Introduction
With the rapid growth of online transactions, credit card fraud has become a serious concern for financial institutions. Detecting fraudulent transactions manually is inefficient and error-prone. This project aims to automate fraud detection using machine learning, ensuring faster and more accurate identification of suspicious activities.

---

## 🔍 Overview
- Uses real-world credit card transaction datasets
- Performs data preprocessing and feature scaling
- Trains a Logistic Regression model
- Evaluates model performance on test data
- Deploys the trained model using Streamlit UI

---

## 🛠️ Technology Used

| Category        | Tools / Libraries |
|-----------------|------------------|
| Programming     | Python |
| Data Handling   | Pandas, NumPy |
| Machine Learning| Scikit-learn |
| Model Deployment| Streamlit |
| Version Control | Git, GitHub |

---
## Dataset

⚠️
Due to GitHub file size limitations, the datasets used in this project are not uploaded directly to this repository.

📌 Reason
The datasets are very large and exceed GitHub’s recommended file size limits.
Uploading large datasets can slow down repository cloning and is not considered a best practice.

🔗 Dataset Download Links
Please download the datasets manually from the following sources and place them inside the dataset/ folder:
fraudTrain.csv – Training dataset
fraudTest.csv – Testing dataset

https://www.kaggle.com/datasets/kartik2112/fraud-detection

---

## ⚙️ Project Setup

credit_card_fraud_detection/

│

├── datasets/

│ ├── fraudTrain.csv

│ └── fraudTest.csv

│

├── model_train.py

├── app.py

├── requirements.txt

└── README.md

---

## 📦 Installation Setup

- 1. Clone the repository:
   ```bash
   git clone https://github.com/Harish-M-5/credit-card-fraud-detection.git


- 2. Navigate to the project directory:
 
 ``bash

cd credit-card-fraud-detection


- 3. Install required dependencies:

 ``bash

pip install -r requirements.txt



---

## 📊 Dataset Details

- fraudTrain.csv: Training dataset containing historical transaction records

🔹 Purpose
Used to train the machine learning model.

🔹 What it contains
Historical credit card transactions
Both fraudulent (1) and non-fraudulent (0) records

Features like:
- Transaction amount
- Transaction time
- Merchant information
- User details
- Category, location, etc.


🔹 Role in ML Model
The model learns patterns of fraud
Understands relationships between features and the target variabl


Used for:
- Feature learning
- Model fitting
- Parameter optimization



- fraudTest.csv: Testing dataset used for evaluation

🔹 Purpose
Used to evaluate the trained model.

🔹 What it contains
New transactions not seen during training
Same structure as training dataset
Represents real-world data


🔹 Role in ML Model
Tests how well the model performs on unseen data

Measures:
- Accuracy
- Precision
- Recall
- Fraud detection performance


others:

- Target column: is_fraud

- Contains numerical and categorical features related to transactions

---

##  🧠 Model Architecture

- Algorithm: Logistic Regression

- Feature Scaling: StandardScaler

- Input: Transaction-related numerical features

- Output: Binary classification (Fraud / Legit)

---

## 🎯 Project Objectives

- Detect fraudulent credit card transactions

- Reduce financial risk and losses

- Build an end-to-end machine learning pipeline

- Deploy model using a user-friendly web interface

---


## 💡 Use Case & Problem Solved

- Problem:
Manual fraud detection is slow and unreliable for large-scale transactions.

- Solution:
This system automates fraud detection using machine learning, providing quick and accurate predictions, helping financial institutions take preventive action.

---

## 🔐 Security

- No sensitive customer information is stored

- Model files are used only for prediction

- Application runs locally without exposing data externally

---

## 🔄 Process Explained

Step 1: Load and preprocess datasets
Step 2: Train the machine learning model
Step 3: Evaluate model performance
Step 4: Deploy model using Streamlit UI

---
## 📘 Learning Outcomes

- Understanding real-world fraud detection problems

-Hands-on experience with supervised ML models

-Data preprocessing and feature scaling

- Model deployment using Streamlit

- End-to-end ML project development

---

## 📂 Data Information

-High-dimensional transaction data

- Imbalanced dataset (fraud cases are rare)

- Numerical features used for training

---
## 🔑 Key Concepts

- Supervised Machine Learning

- Logistic Regression

- Data Scaling

- Model Serialization (Pickle)

- Streamlit Deployment

---

## 👨‍🎓 Ideal For

- Students learning Machine Learning

- Data Science beginners

- Internship and academic projects

- Portfolio and resume projects

---

## 🚀 Future Enhancements

- Use advanced models 

- Handle class imbalance using SMOTE

- Add categorical feature encoding

- Improve UI with charts and metrics



---

## ⚙️ Configuration

- Python version: 3.8+

- Dataset path configurable inside model_train.py

- Streamlit runs on default port 8501

---
## output
<img width="1920" height="1080" alt="Screenshot 2025-12-27 131139" src="https://github.com/user-attachments/assets/0ee79d56-f484-482c-a10c-c3038a2fd391" />

<img width="1920" height="1080" alt="Screenshot 2025-12-27 131210" src="https://github.com/user-attachments/assets/6d444348-f27f-4106-9361-6481d7674e1a" />


---

## 📄 License

This project is licensed under the MIT License.


--- 

## 🙏 Acknowledgments

- Dataset source: Kaggle

- Internship guidance and learning resources

- Open-source Python and ML community

---
## ✅ Conclusion

This Credit Card Fraud Detection System demonstrates how machine learning can be effectively used to solve real-world financial problems. The project successfully integrates data preprocessing, model training, evaluation, and deployment into a complete, practical solution.


