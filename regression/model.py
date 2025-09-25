# Battery Life Regression Model using Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
df = pd.read_csv("C:\\Final_Project\\Mini_Project\\regression\\battery_life_dataset.csv")
print("First 5 rows:\n", df.head())
print("\nDataset description:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# ----------------------------
# Step 2: Fill missing values (numeric columns)
# ----------------------------
df.fillna(df.mean(), inplace=True)

# ----------------------------
# Step 3: Feature Engineering
# ----------------------------
df['battery_usage_rate'] = df['battery_percentage'] / df['screen_on_time_hr']
df['cpu_load_per_app'] = df['cpu_usage_percent'] / (df['apps_running'] + 1)

# ----------------------------
# Step 4: Features and target
# ----------------------------
target_column = 'battery_life_remaining_hr'
X = df.drop(columns=[target_column])
y = df[target_column]

print("\nFeature columns:\n", X.columns)

# ----------------------------
# Step 5: Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# ----------------------------
# Step 6: Hyperparameter tuning for Random Forest
# ----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='r2'
)
grid.fit(X_train, y_train)

# Best Random Forest model
rf_model = grid.best_estimator_
print("\nBest Random Forest parameters:", grid.best_params_)
print("Best cross-validation score (R2):", grid.best_score_)

# ----------------------------
# Step 7: Predictions & Evaluation
# ----------------------------
y_pred_rf = rf_model.predict(X_test)

# Random Forest performance
print("\nRandom Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# ----------------------------
# Step 8: Visualization
# ----------------------------
plt.figure(figsize=(18,5))

# Subplot 1: Actual vs Predicted
plt.subplot(1,3,1)
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Battery Life (hr)")
plt.ylabel("Predicted Battery Life (hr)")
plt.title("Actual vs Predicted Battery Life")

# Subplot 2: Residuals
plt.subplot(1,3,2)
residuals_rf = y_test - y_pred_rf
sns.histplot(residuals_rf, bins=30, kde=True, color='blue', alpha=0.7)
plt.xlabel("Residuals")
plt.title("Residuals Distribution")

# Subplot 3: Feature Importance
plt.subplot(1,3,3)
importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True)
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='purple', alpha=0.7)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()
plt.show()


