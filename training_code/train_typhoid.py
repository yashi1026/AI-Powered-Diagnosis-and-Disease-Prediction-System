import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import shap

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("typhoid_clean_dataset.csv")

print("Shape:", df.shape)
print(df.head())

# =========================
# CLEANING
# =========================
df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# =========================
# VISUALIZATION 1: CLASS DISTRIBUTION
# =========================
plt.figure()
df['Typhoid_Status'].value_counts().plot(kind='bar')
plt.title("Typhoid Class Distribution")
plt.xlabel("Class (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# =========================
# VISUALIZATION 2: CORRELATION HEATMAP
# =========================
plt.figure()
corr = df.corr()

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()

# =========================
# FEATURES / TARGET
# =========================
X = df.drop("Typhoid_Status", axis=1)
y = df["Typhoid_Status"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL (NO SCALING 🔥)
# =========================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================
importances = model.feature_importances_

plt.figure()
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# =========================
# =========================
# =========================
# 🔥 SHAP FINAL FIX (CORRECT)
# =========================
explainer = shap.TreeExplainer(model)

sample = X_train.iloc[:100]

shap_values = explainer.shap_values(sample)

# ✅ HANDLE ALL CASES PROPERLY
values = np.array(shap_values)

print("Raw SHAP shape:", values.shape)

# Case 1: list → old version
if isinstance(shap_values, list):
    values = shap_values[1]

# Case 2: 3D array → new version
elif values.ndim == 3:
    values = values[:, :, 1]   # take class 1

# Case 3: already correct → do nothing

print("Final SHAP shape:", values.shape)
print("Sample shape:", sample.shape)

# ✅ FINAL PLOT
shap.summary_plot(values, sample)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/typhoid_model.pkl")

print("\n✅ Model saved successfully!")