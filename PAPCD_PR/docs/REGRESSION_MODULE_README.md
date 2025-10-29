# 🧠 Macroeconomic Regression Analysis Module

A comprehensive Python module for training multivariable regression models to analyze macroeconomic relationships and predict GDP growth.

## 📊 Overview

This module provides a complete pipeline for macroeconomic regression analysis, including:

- **Data preprocessing** and feature engineering
- **Multiple regression models** (Linear, Ridge, Lasso, ElasticNet)
- **Model evaluation** and performance metrics
- **Visualization** of results and coefficients
- **Export functionality** for further analysis

## 🎯 Key Features

### 📈 Models Implemented
- **LinearRegression**: Baseline linear regression
- **Ridge**: L2 regularization to prevent overfitting
- **Lasso**: L1 regularization for feature selection
- **ElasticNet**: Combination of L1 and L2 regularization

### 🔧 Features Analyzed
- **investment_GDP**: Investment as percentage of GDP
- **exports_GDP**: Exports as percentage of GDP
- **inflation**: Inflation rate (US proxy)
- **population**: Log-transformed population
- **oil_price**: Oil price (Brent crude)
- **fed_rate**: Federal Reserve interest rate
- **diversification_HHI**: Economic diversification index

### 📊 Target Variable
- **GDP_growth**: Annual GDP growth rate (%)

## 🚀 Quick Start

### Basic Usage

```python
from src.macroeconomic_regression import MacroeconomicRegression

# Initialize the regression class
regression = MacroeconomicRegression()

# Run complete analysis
summary_df, coef_df = regression.run_full_analysis(
    start_year=1970,
    end_year=2021,
    test_size=0.2,
    scaler_type='standard',
    optimize_hyperparameters=True
)

# View results
print(summary_df)
print(coef_df)
```

### Step-by-Step Usage

```python
# 1. Load and preprocess data
X, y = regression.preprocess_data(start_year=2000, end_year=2021)

# 2. Split data
regression.split_data(X, y, test_size=0.2)

# 3. Scale features
regression.scale_features('standard')

# 4. Train models
regression.train_models(optimize_hyperparameters=True)

# 5. Get results
summary = regression.get_model_summary()
coefficients = regression.get_coefficients()

# 6. Create visualizations
regression.plot_coefficients()
regression.plot_model_performance()
regression.plot_predictions('LinearRegression')
```

## 📋 Class Methods

### Data Loading and Preprocessing

#### `load_data()`
Loads the macroeconomic dataset from CSV file.

#### `preprocess_data(start_year, end_year, min_years_per_country)`
- Filters data by year range and country data availability
- Engineers features for regression analysis
- Returns feature matrix (X) and target variable (y)

**Parameters:**
- `start_year`: Start year for analysis (default: 1970)
- `end_year`: End year for analysis (default: 2021)
- `min_years_per_country`: Minimum years required per country (default: 10)

### Model Training

#### `split_data(X, y, test_size, random_state)`
Splits data into training and validation sets.

#### `scale_features(scaler_type)`
Scales features using StandardScaler or RobustScaler.

#### `train_models(use_scaled, optimize_hyperparameters)`
Trains all regression models with optional hyperparameter optimization.

### Results and Analysis

#### `get_model_summary()`
Returns DataFrame with performance metrics for all models.

#### `get_coefficients()`
Returns DataFrame with coefficients for all models.

#### `plot_coefficients(figsize)`
Creates heatmap visualization of model coefficients.

#### `plot_model_performance(figsize)`
Creates bar charts comparing model performance metrics.

#### `plot_predictions(model_name, figsize)`
Creates scatter plot of actual vs predicted values.

### Export and Utilities

#### `export_results(output_dir)`
Exports all results to CSV files in specified directory.

#### `run_full_analysis(...)`
Runs the complete analysis pipeline and returns results.

## 📊 Model Performance Metrics

The module calculates comprehensive performance metrics:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Square Error

## 🎨 Visualization Features

### Coefficient Analysis
- Heatmap showing coefficients across models
- Feature importance ranking
- Coefficient comparison charts

### Model Performance
- R² score comparison
- RMSE and MAE comparison
- Actual vs predicted scatter plots
- Residuals analysis

### Comprehensive Reports
- Multi-panel visualization reports
- Summary statistics
- Economic interpretation guides

## 📈 Economic Interpretation

### Key Economic Insights

The module provides economic interpretation of regression results:

1. **Investment/GDP Ratio**
   - Positive coefficient indicates investment drives growth
   - Magnitude shows economic impact

2. **Exports/GDP Ratio**
   - Positive coefficient suggests export-led growth
   - Trade openness effect on growth

3. **Inflation Rate**
   - Negative coefficient indicates inflation hampers growth
   - Price stability importance

4. **Federal Funds Rate**
   - Monetary policy impact on growth
   - Interest rate transmission mechanism

### Regularization Effects

- **Linear Regression**: Baseline model, no regularization
- **Ridge**: Prevents overfitting, maintains all features
- **Lasso**: Feature selection, sets some coefficients to zero
- **ElasticNet**: Balance between Ridge and Lasso approaches

## 🔧 Configuration Options

### Data Filtering
```python
# Filter by time period
X, y = regression.preprocess_data(start_year=2000, end_year=2021)

# Filter by minimum data availability
X, y = regression.preprocess_data(min_years_per_country=15)
```

### Model Training
```python
# Use different scalers
regression.scale_features('standard')  # StandardScaler
regression.scale_features('robust')    # RobustScaler

# Enable/disable hyperparameter optimization
regression.train_models(optimize_hyperparameters=True)
```

### Data Splitting
```python
# Custom train/validation split
regression.split_data(X, y, test_size=0.25)  # 75% train, 25% validation
```

## 📁 Output Files

The module generates several output files:

### Results Directory (`results/`)
- `model_performance_summary.csv`: Performance metrics for all models
- `model_coefficients.csv`: Coefficients for all models
- `linearregression_predictions.csv`: Predictions from Linear Regression
- `ridge_predictions.csv`: Predictions from Ridge Regression
- `lasso_predictions.csv`: Predictions from Lasso Regression
- `elasticnet_predictions.csv`: Predictions from ElasticNet

### Visualization Files
- `regression_analysis_report.png`: Comprehensive visualization report

## 🧪 Testing and Validation

### Test Script
Run the test script to validate the module:

```bash
python test_regression_module.py
```

### Example Analysis
Run the comprehensive example:

```bash
python example_regression_analysis.py
```

## 📊 Sample Results

### Model Performance (Example)
```
              Model  Train_R2  Val_R2  Train_RMSE  Val_RMSE  Train_MAE  Val_MAE
0  LinearRegression    0.0918  0.1029      5.4756    4.4955     3.2389   3.0281
1             Ridge    0.0918  0.1029      5.4756    4.4955     3.2388   3.0280
2             Lasso    0.0077  0.0097      5.7235    4.7234     3.3876   3.1184
3        ElasticNet    0.0374  0.0454      5.6372    4.6373     3.3143   3.0383
```

### Feature Importance (Example)
```
                LinearRegression   Ridge   Lasso  ElasticNet
investment_GDP            1.1136  1.1132  0.1195      0.4154
exports_GDP               0.6238  0.6235  0.0000      0.0389
inflation                -0.6615 -0.6612 -0.0000     -0.1200
population                0.6816  0.6812  0.0000      0.0000
oil_price                 0.5764  0.5760  0.0000      0.0000
fed_rate                  0.6938  0.6936  0.0000      0.1876
```

## 🔬 Research Applications

### Academic Research
- Economic growth determinants analysis
- Policy impact assessment
- Cross-country comparative studies
- Time series macroeconomic modeling

### Policy Analysis
- Investment policy evaluation
- Trade policy impact assessment
- Monetary policy effectiveness
- Economic diversification strategies

### Business Applications
- Market size estimation
- Economic forecasting
- Risk assessment
- Investment decision support

## 🛠️ Technical Requirements

### Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Installation
```bash
pip install -r requirements.txt
```

## 📚 Advanced Usage

### Custom Feature Engineering
```python
# Add custom features to the regression class
regression.feature_mapping['custom_feature'] = 'your_column_name'
```

### Model Customization
```python
# Add custom models
from sklearn.ensemble import RandomForestRegressor
regression.models['RandomForest'] = RandomForestRegressor()
```

### Cross-Validation
```python
# Enable cross-validation in model training
regression.train_models(use_scaled=True, optimize_hyperparameters=True)
```

## 🎯 Best Practices

1. **Data Quality**: Ensure sufficient data points per country
2. **Feature Scaling**: Always scale features for regularized models
3. **Hyperparameter Tuning**: Use grid search for optimal performance
4. **Model Selection**: Compare multiple models for robustness
5. **Economic Interpretation**: Consider economic theory when interpreting results

## 📞 Support and Documentation

For technical issues or feature requests, please refer to:
- Module documentation in `src/macroeconomic_regression.py`
- Example scripts in `example_regression_analysis.py`
- Test validation in `test_regression_module.py`

---

**Happy Analyzing! 🧠📊**

