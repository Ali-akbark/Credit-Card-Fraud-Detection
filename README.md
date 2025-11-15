


# 📌 **Credit Card Fraud Detection**

**Machine Learning Project — Classification | Imbalanced Dataset | End-to-End Pipeline**

---

## 🚀 **Project Overview**

Credit card fraud is a major concern for banks and financial institutions. Fraudulent transactions are rare but extremely costly.
This project builds a machine learning model that **detects fraudulent transactions** using an imbalanced dataset from Kaggle.

The goal is to build a **production-ready fraud detection system** using best practices:
✔ Exploratory Data Analysis
✔ Handling class imbalance
✔ Model training
✔ Evaluation with correct metrics
✔ Threshold tuning
✔ Saving the final model (pickle format)



## 📊 **Dataset Information**

This dataset contains:

* **284,807 transactions**
* **492 fraud cases** (only **0.172%**) → **highly imbalanced**
* Features `V1` to `V28` are PCA-transformed components (to protect sensitive customer info)
* Additional features:

  * `Time` — seconds elapsed between transactions
  * `Amount` — transaction value
* `Class` → Target variable

  * `0` = Legitimate
  * `1` = Fraud



## 🧠 **Problem Definition**

> **Given anonymized credit card transaction data, the task is to classify whether a transaction is fraudulent or not.**

Because fraudulent transactions are extremely rare, this is an **imbalanced classification problem**.



## 🛠️ **Project Workflow**

### **1️⃣ Data Loading & Basic Checks**

* Shape, missing values, duplicates
* Distribution of target variable (`Class`)

### **2️⃣ Exploratory Data Analysis**

* Class imbalance visualization
* Distribution of `Amount` and `Time`
* Correlation of features with `Class`
* Identifying most important PCA components

### **3️⃣ Data Preprocessing**

* Scaling numerical features (StandardScaler)
* Train-test split (stratified)

### **4️⃣ Handling Class Imbalance**

* Used **class weights** to penalize misclassification of minority class
* (Alternative techniques: SMOTE, undersampling — but class weights performed best)

### **5️⃣ Model Training**

Models tried:

* Logistic Regression
* Random Forest Classifier (final chosen model)

Random Forest performed best in recall & ROC-AUC.

### **6️⃣ Model Evaluation**

Correct metrics used:

* **Confusion Matrix**
* **Precision, Recall, F1-score**
* **ROC-AUC Score**
* **Threshold Tuning** to catch more frauds

### **7️⃣ Saving the Final Model**

Saved using `pickle`:

```
models/
│-- scaler.pkl
│-- random_forest_model.pkl
```

---

## 🧪 **Model Performance**

### 📌 **Confusion Matrix (Best Threshold)**

```
[[56644     7]
 [   24    71]]
```

### 📌 **Classification Report**

* Legit (0): **Precision 1.00 | Recall 1.00**
* Fraud (1): **Precision 0.91 | Recall 0.75**

### 📌 **ROC-AUC Score**

```
0.973
```

Random Forest + Class Weights + Threshold tuning gives **strong recall** on fraud cases while keeping false positives very low.



## 💾 **Files in This Repository**

```
Credit-Card-Fraud-Detection/
│
├── models/
│   ├── scaler.pkl                → StandardScaler object
│   └── random_forest_model.pkl   → Final trained model
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb  → Full EDA + Model training code
│
├── README.md
└── requirements.txt
```

---

## 🧰 **Technologies Used**

* **Python**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Scikit-learn**
* **Imbalanced-learn (for class imbalance handling)**
* **Pickle**
* **Jupyter Notebook**

---

## 🔥 **Key Highlights of This Project**

✔ Solved a **highly imbalanced** real-world problem
✔ Demonstrated **EDA, visualization, and preprocessing**
✔ Implemented class balancing using **class weights**
✔ Built multiple ML models and selected the best one
✔ Optimized recall using **threshold tuning**
✔ Saved a production-ready model with scaler
✔ Clean & professional GitHub structure

---

## 🙌 **Author**

**Ali Akbark Kanorewala**
Data Science Enthusiast | ML | Python




