# =========================
# 🧠 PARKINSON MODEL TRAINING (FULL PIPELINE)
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# =========================
# 📌 LOAD DATA
# =========================
df = pd.read_csv("Parkinsson disease.csv")

print("\n===== ORIGINAL DATA =====")
print(df.head())

# =========================
# 🧹 CLEANING
# =========================

# Remove non-numeric column
if "name" in df.columns:
    df.drop("name", axis=1, inplace=True)

# Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# =========================
# 📊 EDA
# =========================

# Class distribution
print("\n===== CLASS DISTRIBUTION =====")
print(df["status"].value_counts())

plt.figure()
sns.countplot(x="status", data=df)
plt.title("Target Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Feature distributions
df.hist(figsize=(15,12))
plt.suptitle("Feature Distributions")
plt.show()

# =========================
# 🧠 PREPROCESSING
# =========================

X = df.drop("status", axis=1)
y = df["status"]

print("\n===== FEATURES =====")
print(X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 🤖 MODEL (XGBOOST)
# =========================
base_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

# Calibration (important for realistic probabilities)
model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)

# Train
model.fit(X_train_scaled, y_train)

# =========================
# 📈 EVALUATION
# =========================

y_pred = model.predict(X_test_scaled)

print("\n===== ACCURACY =====")
print(accuracy_score(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# 📊 FEATURE IMPORTANCE
# =========================

# Fit base model for importance
base_model.fit(X_train_scaled, y_train)

importance = base_model.feature_importances_
feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,5))
feat_imp.plot(kind='bar')
plt.title("Feature Importance")
plt.show()

print("\n===== TOP FEATURES =====")
print(feat_imp.head(10))

# =========================
# 💾 SAVE MODELS
# =========================

joblib.dump(model, "models/parkinson_model.pkl")        # calibrated model
joblib.dump(base_model, "models/parkinson_base.pkl")    # SHAP model
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Model and scaler saved successfully!")

# =========================
# 🧪 TEST SAMPLE
# =========================

sample = X_test.iloc[0:1]
scaled_sample = scaler.transform(sample)

print("\n===== SAMPLE TEST =====")
print("Prediction:", model.predict(scaled_sample))
print("Probability:", model.predict_proba(scaled_sample))