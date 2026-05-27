# ==========================================
# HEART DISEASE MODEL - COMPLETE PIPELINE
# ==========================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

# ==================================
# 1. LOAD DATA
# ==================================
df = pd.read_csv("dataset/heart.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ==================================
# 2. CLEAN COLUMN NAMES (FINAL STANDARD)
# ==================================
df = df.rename(columns={
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "thalach": "max_heart_rate",
    "exang": "exercise_angina",
    "oldpeak": "st_depression",
    "ca": "num_major_vessels"
})

print("\nRenamed Columns:", df.columns.tolist())

# ==================================
# 3. CHECK MISSING VALUES
# ==================================
print("\nMissing Values:\n", df.isnull().sum())

# ==================================
# 4. EDA VISUALIZATION
# ==================================

# Target distribution
plt.figure()
df["target"].value_counts().plot(kind="bar", title="Target Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==================================
# 5. FEATURES & TARGET
# ==================================
X = df.drop("target", axis=1)
y = df["target"]

print("\nFinal Features Used:")
print(X.columns.tolist())

# ==================================
# 6. TRAIN TEST SPLIT
# ==================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================================
# 7. HANDLE IMBALANCE (SMOTE)
# ==================================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n", pd.Series(y_train).value_counts())

# ==================================
# 8. TRAIN MODEL (OPTIMIZED)
# ==================================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==================================
# 9. PREDICTION
# ==================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==================================
# 10. EVALUATION
# ==================================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", roc_auc)

# ==================================
# 11. ROC CURVE
# ==================================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# ==================================
# 12. FEATURE IMPORTANCE
# ==================================
importances = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:\n", feature_df.head(10))

# Plot importance
plt.figure(figsize=(10,6))
plt.barh(feature_df["Feature"][:10], feature_df["Importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importance")
plt.show()

# ==================================
# 13. SAVE MODEL + COLUMNS
# ==================================
joblib.dump(model, "heart_model.pkl")
joblib.dump(X.columns.tolist(), "heart_columns.pkl")

print("\n✅ MODEL TRAINED & SAVED SUCCESSFULLY")