import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
from xgboost import XGBClassifier
import shap

# ----------------------
# 📂 CREATE FOLDERS
# ----------------------
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ----------------------
# 📂 LOAD DATASET
# ----------------------
df = pd.read_csv("liver_health.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Drop missing
df = df.dropna()

# Encode Gender
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Convert target
df['Dataset'] = df['Dataset'].apply(lambda x: 0 if x == 1 else 1)

print("Class Distribution BEFORE:\n", df['Dataset'].value_counts())

# ----------------------
# ⚖️ BALANCE DATASET
# ----------------------
df_majority = df[df.Dataset == 1]
df_minority = df[df.Dataset == 0]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df = pd.concat([df_majority, df_minority_upsampled])

print("Class Distribution AFTER:\n", df['Dataset'].value_counts())

# ----------------------
# 📊 VISUALIZATION
# ----------------------
sns.countplot(x='Dataset', data=df)
plt.title("Balanced Target Distribution")
plt.savefig("outputs/target_distribution.png")
plt.close()

# ----------------------
# 🧪 FEATURES & TARGET
# ----------------------
X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Save feature names
feature_names = X.columns.tolist()
pickle.dump(feature_names, open("model/feature_names.pkl", "wb"))

# ----------------------
# ✂️ TRAIN TEST SPLIT
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# 🤖 MODEL (OPTIMIZED)
# ----------------------
model = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ----------------------
# 📈 EVALUATION
# ----------------------
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------
# 🔥 SHAP (CORRECT WAY)
# ----------------------
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.savefig("outputs/shap_summary.png", bbox_inches='tight')
plt.close()

# ----------------------
# 💾 SAVE MODEL
# ----------------------
pickle.dump(model, open("model/liver_model.pkl", "wb"))
pickle.dump(explainer, open("model/explainer.pkl", "wb"))

print("\n✅ Model + Explainer saved successfully!")