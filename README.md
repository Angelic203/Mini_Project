# Mini_Project

A machine learning mini-project demonstrating classification and regression tasks using Python, scikit-learn, and pandas.

## Project Structure

```
Mini_Project/
├── README.md
├── classification/
│   └── model.py
└── regression/
    ├── model.py
    └── battery_life_dataset.csv
```

## Regression Task: Battery Life Prediction

### Overview
This task predicts the remaining battery life (in hours) of a smartphone based on various usage patterns and hardware metrics. The model uses Random Forest regression to handle complex, non-linear relationships in the data.

### Features Used
- **battery_percentage**: Current battery level (0-100%)
- **screen_on_time_hr**: Hours the screen has been on
- **cpu_usage_percent**: CPU utilization percentage
- **apps_running**: Number of active applications
- **brightness_level**: Screen brightness (0-100)
- **battery_temperature_c**: Battery temperature in Celsius
- **charging_status**: Whether the device is charging (0/1)
- **network_usage_mb**: Data usage in MB
- **battery_health_degraded**: Battery health status (0/1)

### Engineered Features
- **battery_usage_rate**: battery_percentage / screen_on_time_hr
- **cpu_load_per_app**: cpu_usage_percent / (apps_running + 1)

### Model Details
- **Algorithm**: Random Forest Regressor
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Evaluation Metrics**: Mean Squared Error (MSE), R² Score
- **Visualization**: Actual vs Predicted scatter plot, residuals distribution, feature importance

### Dataset
- **Source**: battery_life_dataset.csv
- **Size**: 1000 samples
- **Target**: battery_life_remaining_hr (hours)

### Usage
1. Ensure Python 3.x is installed with required packages:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```

2. Run the model:
   ```bash
   python regression/model.py
   ```

### Results
- Best hyperparameters are selected via grid search
- Model performance is evaluated on test set
- Visualizations provide insights into predictions and feature importance

## Classification Task
(TBD - Documentation for classification model to be added)

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

## License
This project is for educational purposes.
