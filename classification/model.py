# ===============================
# Smart Packing Assistant 🧳 (Multi-label Classification)
# ===============================

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score,precision_score,recall_score,f1_score, classification_report, confusion_matrix
)

# ------------------------------
# Load & Inspect Dataset
# ------------------------------
df = pd.read_csv("C://Final_Project//Mini_Project//classification//smart_packing_assistant_dataset.csv")
print(df)
print(df.head())
print(df.info())
print(df.describe())
df.columns = df.columns.str.strip()
df['Essential_Item_Categories'] = df['Essential_Item_Categories'].apply(lambda x: [i.strip() for i in x.split(",")])

# ------------------------------
# Label Encode Categorical Features
# ------------------------------
categorical_cols = ['Trip_Type', 'Destination_Climate', 'Gender', 'Planned_Activity',
                    'Travel_Type', 'Accommodation_Type']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ------------------------------
# Prepare Features and Labels
# ------------------------------
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Essential_Item_Categories'])
X = df.drop('Essential_Item_Categories', axis=1)

# ------------------------------
# Normalize Features
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# Random Forest + MultiOutput Wrapper (multi-label)
# ------------------------------
base_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,             # further limited to reduce overfitting
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',     # limit features per split
    random_state=42,
    n_jobs=-1
)

multi_rf = MultiOutputClassifier(base_rf)
multi_rf.fit(X_train, y_train)

# ------------------------------
# Evaluation
# ------------------------------
y_pred = multi_rf.predict(X_test)



# Optional: Micro-averaged accuracy (approx for multi-label)
print("✅ Subset Accuracy:", round(accuracy_score(y_test, y_pred), 4))
# ------------------------------
# Evaluation
# ------------------------------
y_pred = multi_rf.predict(X_test)

# Use micro-average (or another suitable average) for multi-label classification
print("✅ Subset Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision score:", precision_score(y_test, y_pred, average='micro'))
print("Recall :", recall_score(y_test, y_pred, average='micro'))
print("F1 Score :", f1_score(y_test, y_pred, average='micro'))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=mlb.classes_))

# ------------------------------
# Confusion Matrix (for ONE LABEL ONLY)
# ------------------------------
label_index = 0  # You can change this
label_name = mlb.classes_[label_index]

cm = confusion_matrix(y_test[:, label_index], y_pred[:, label_index], labels=[0, 1])
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix for '{label_name}'")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ------------------------------
# Feature Importance (average across labels)
# ------------------------------
# Note: MultiOutputClassifier doesn't directly expose feature_importances_
# We'll average them manually

importances = np.mean([
    estimator.feature_importances_ for estimator in multi_rf.estimators_
], axis=0)

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Average Feature Importances Across Labels")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
