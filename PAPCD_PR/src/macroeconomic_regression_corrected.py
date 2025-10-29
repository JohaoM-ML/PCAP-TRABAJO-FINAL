"""
CORRECTED Macroeconomic Regression Analysis Module

This module addresses the circular causality issue by using proper independent variables
that don't create endogeneity problems when predicting GDP growth.

Key Corrections:
1. Use absolute investment levels instead of investment/GDP ratios
2. Use absolute export levels instead of export/GDP ratios  
3. Add lagged variables to capture temporal relationships
4. Use external/policy variables that are truly independent

Author: Macroeconomic Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class MacroeconomicRegressionCorrected:
    """
    CORRECTED version of the macroeconomic regression analysis class.
    
    This class addresses endogeneity issues by using proper independent variables
    that don't create circular causality when predicting GDP growth.
    """
    
    def __init__(self, data_path='data/external/global_merged_all.csv'):
        """
        Initialize the corrected MacroeconomicRegression class.
        
        Parameters:
        -----------
        data_path : str
            Path to the merged macroeconomic dataset
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.feature_names = []
        
        # CORRECTED feature mapping - using absolute values and external variables
        self.feature_mapping = {
            'investment_absolute': 'Gross_capital_formation_main',  # Absolute investment, not ratio
            'exports_absolute': 'Exports_main',  # Absolute exports, not ratio
            'inflation': 'US_Inflation_Index',  # External inflation proxy
            'population': 'Population_wb',  # Population level
            'oil_price': 'Oil_Price_Brent',  # External oil price
            'fed_rate': 'Federal_Funds_Rate',  # External monetary policy
            'exchange_rate': 'AMA_exchange_rate',  # External exchange rate
            'diversification_HHI': 'diversification_HHI'  # Calculated diversification
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize regression models with default parameters."""
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=1.0, random_state=42, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
        }
    
    def load_data(self):
        """
        Load the macroeconomic dataset.
        
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        print("Loading macroeconomic dataset...")
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.data.shape[0]:,} rows, {self.data.shape[1]} columns")
        return self.data
    
    def preprocess_data(self, start_year=1970, end_year=2021, min_years_per_country=10, 
                       use_lagged_variables=True, lag_years=1):
        """
        Preprocess the data for regression analysis with corrected features.
        
        Parameters:
        -----------
        start_year : int
            Start year for analysis
        end_year : int
            End year for analysis
        min_years_per_country : int
            Minimum number of years required per country
        use_lagged_variables : bool
            Whether to include lagged variables to avoid endogeneity
        lag_years : int
            Number of years to lag variables
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataset
        """
        print("Preprocessing data for CORRECTED regression analysis...")
        print("Addressing endogeneity issues with proper independent variables...")
        
        if self.data is None:
            self.load_data()
        
        # Filter by year range
        df = self.data[
            (self.data['Year'] >= start_year) & 
            (self.data['Year'] <= end_year)
        ].copy()
        
        print(f"Data after year filtering: {df.shape[0]:,} rows")
        
        # Filter countries with sufficient data
        country_counts = df.groupby('Country').size()
        countries_with_sufficient_data = country_counts[country_counts >= min_years_per_country].index
        df = df[df['Country'].isin(countries_with_sufficient_data)]
        
        print(f"Data after country filtering: {df.shape[0]:,} rows")
        print(f"Countries with sufficient data: {len(countries_with_sufficient_data)}")
        
        # Create corrected feature engineering
        df = self._engineer_features_corrected(df, use_lagged_variables, lag_years)
        
        # Select features and target
        feature_columns = list(self.feature_mapping.keys())
        target_column = 'GDP_growth'
        
        # Create feature matrix
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset after removing missing values: {X.shape[0]:,} rows")
        print(f"Features: {list(X.columns)}")
        
        # Store processed data
        self.processed_data = df[mask].copy()
        self.feature_names = list(X.columns)
        
        return X, y
    
    def _engineer_features_corrected(self, df, use_lagged_variables=True, lag_years=1):
        """
        Engineer CORRECTED features for regression analysis.
        
        This method addresses endogeneity by:
        1. Using absolute values instead of GDP ratios
        2. Adding lagged variables to capture temporal relationships
        3. Using external/policy variables that are truly independent
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        use_lagged_variables : bool
            Whether to include lagged variables
        lag_years : int
            Number of years to lag variables
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with corrected engineered features
        """
        print("Engineering CORRECTED features (addressing endogeneity)...")
        
        # Sort by country and year for lag calculations
        df = df.sort_values(['Country', 'Year']).reset_index(drop=True)
        
        # 1. ABSOLUTE INVESTMENT (not investment/GDP ratio)
        if 'Gross_capital_formation_main' in df.columns:
            df['investment_absolute'] = df['Gross_capital_formation_main']
        else:
            # Fallback calculation
            df['investment_absolute'] = df['Gross_capital_formation_percent_GDP'] * df['GDP_main'] / 100
        
        # 2. ABSOLUTE EXPORTS (not exports/GDP ratio)
        if 'Exports_main' in df.columns:
            df['exports_absolute'] = df['Exports_main']
        else:
            # Fallback calculation
            df['exports_absolute'] = df['Exports_wb']
        
        # 3. EXTERNAL VARIABLES (truly independent)
        df['inflation'] = df['US_Inflation_Index']  # External inflation proxy
        df['population'] = np.log(df['Population_wb'] + 1)  # Log population
        df['oil_price'] = df['Oil_Price_Brent']  # External oil price
        df['fed_rate'] = df['Federal_Funds_Rate']  # External monetary policy
        df['exchange_rate'] = df['AMA_exchange_rate']  # External exchange rate
        
        # 4. DIVERSIFICATION INDEX (calculated from sectoral data)
        df['diversification_HHI'] = self._calculate_diversification_index(df)
        
        # 5. LAGGED VARIABLES (to avoid endogeneity)
        if use_lagged_variables:
            print(f"Adding lagged variables (lag = {lag_years} years)...")
            
            # Create lagged versions of potentially endogenous variables
            lagged_vars = ['investment_absolute', 'exports_absolute']
            
            for var in lagged_vars:
                if var in df.columns:
                    df[f'{var}_lag{lag_years}'] = df.groupby('Country')[var].shift(lag_years)
                    # Add to feature mapping
                    self.feature_mapping[f'{var}_lag{lag_years}'] = f'{var}_lag{lag_years}'
            
            # Remove original potentially endogenous variables from feature mapping
            for var in lagged_vars:
                if var in self.feature_mapping:
                    del self.feature_mapping[var]
        
        # 6. ADDITIONAL EXTERNAL VARIABLES
        # Add trade openness (imports + exports) as absolute values
        if 'Imports_main' in df.columns:
            df['trade_volume'] = df['Exports_main'] + df['Imports_main']
            self.feature_mapping['trade_volume'] = 'trade_volume'
        
        # Add government consumption as absolute value
        if 'Government_consumption' in df.columns:
            df['government_consumption'] = df['Government_consumption']
            self.feature_mapping['government_consumption'] = 'government_consumption'
        
        print("CORRECTED feature engineering complete!")
        print("Key corrections made:")
        print("  - Using absolute investment instead of investment/GDP ratio")
        print("  - Using absolute exports instead of exports/GDP ratio")
        print("  - Adding lagged variables to avoid endogeneity")
        print("  - Focusing on external/policy variables")
        
        return df
    
    def _calculate_diversification_index(self, df):
        """
        Calculate a simplified diversification index based on sectoral composition.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.Series
            Diversification index values
        """
        # Calculate sector shares
        sectors = [
            'Agriculture_value_added',
            'Manufacturing_value_added',
            'Construction_value_added',
            'Transport_communication',
            'Wholesale_retail_trade',
            'Other_activities'
        ]
        
        diversification = []
        for idx, row in df.iterrows():
            if pd.notna(row['Total_value_added']) and row['Total_value_added'] > 0:
                shares = []
                for sector in sectors:
                    if sector in df.columns and pd.notna(row[sector]):
                        share = row[sector] / row['Total_value_added']
                        shares.append(share)
                
                if shares:
                    # Calculate HHI (sum of squared shares)
                    hhi = sum(s**2 for s in shares)
                    diversification.append(hhi)
                else:
                    diversification.append(np.nan)
            else:
                diversification.append(np.nan)
        
        return pd.Series(diversification, index=df.index)
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and validation sets.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        test_size : float
            Proportion of data for validation
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_val, y_train, y_val)
        """
        print(f"Splitting data: {int((1-test_size)*100)}% training, {int(test_size*100)}% validation")
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape[0]:,} samples")
        print(f"Validation set: {self.X_val.shape[0]:,} samples")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def scale_features(self, scaler_type='standard'):
        """
        Scale features for better model performance.
        
        Parameters:
        -----------
        scaler_type : str
            Type of scaler ('standard' or 'robust')
        """
        print(f"Scaling features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        # Fit scaler on training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        
        print("Features scaled successfully")
    
    def train_models(self, use_scaled=True, optimize_hyperparameters=True):
        """
        Train all regression models.
        
        Parameters:
        -----------
        use_scaled : bool
            Whether to use scaled features
        optimize_hyperparameters : bool
            Whether to optimize hyperparameters using grid search
        """
        print("Training CORRECTED regression models...")
        
        # Select features
        if use_scaled:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
        else:
            X_train = self.X_train
            X_val = self.X_val
        
        # Hyperparameter grids for optimization
        param_grids = {
            'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
            'ElasticNet': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        }
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if optimize_hyperparameters and name in param_grids:
                # Grid search for hyperparameter optimization
                grid_search = GridSearchCV(
                    model, param_grids[name], 
                    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train, self.y_train)
                best_model = grid_search.best_estimator_
                print(f"  Best parameters: {grid_search.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train, self.y_train)
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_val_pred = best_model.predict(X_val)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(self.y_train, y_train_pred)
            val_metrics = self._calculate_metrics(self.y_val, y_val_pred)
            
            # Store results
            self.results[name] = {
                'model': best_model,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'predictions': {
                    'train': y_train_pred,
                    'val': y_val_pred
                }
            }
            
            print(f"  Training R²: {train_metrics['r2']:.4f}")
            print(f"  Validation R²: {val_metrics['r2']:.4f}")
            print(f"  Validation RMSE: {val_metrics['rmse']:.4f}")
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
            
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    def get_coefficients(self):
        """
        Extract coefficients from trained models.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with coefficients for each model
        """
        coefficients = {}
        
        for name, result in self.results.items():
            model = result['model']
            if hasattr(model, 'coef_'):
                coefficients[name] = model.coef_
            else:
                coefficients[name] = np.nan
        
        coef_df = pd.DataFrame(coefficients, index=self.feature_names)
        return coef_df
    
    def get_model_summary(self):
        """
        Get a summary of all model performances.
        
        Returns:
        --------
        pd.DataFrame
            Summary of model performances
        """
        summary_data = []
        
        for name, result in self.results.items():
            train_metrics = result['train_metrics']
            val_metrics = result['val_metrics']
            
            summary_data.append({
                'Model': name,
                'Train_R2': train_metrics['r2'],
                'Val_R2': val_metrics['r2'],
                'Train_RMSE': train_metrics['rmse'],
                'Val_RMSE': val_metrics['rmse'],
                'Train_MAE': train_metrics['mae'],
                'Val_MAE': val_metrics['mae']
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_coefficients(self, figsize=(12, 8)):
        """
        Plot coefficients for all models.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        coef_df = self.get_coefficients()
        
        plt.figure(figsize=figsize)
        
        # Create heatmap of coefficients
        sns.heatmap(coef_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.4f', cbar_kws={'label': 'Coefficient Value'})
        
        plt.title('CORRECTED Regression Coefficients Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance(self, figsize=(15, 5)):
        """
        Plot model performance comparison.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        summary_df = self.get_model_summary()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # R² comparison
        axes[0].bar(summary_df['Model'], summary_df['Val_R2'], alpha=0.7)
        axes[0].set_title('Validation R² Score', fontweight='bold')
        axes[0].set_ylabel('R² Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[1].bar(summary_df['Model'], summary_df['Val_RMSE'], alpha=0.7, color='orange')
        axes[1].set_title('Validation RMSE', fontweight='bold')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[2].bar(summary_df['Model'], summary_df['Val_MAE'], alpha=0.7, color='green')
        axes[2].set_title('Validation MAE', fontweight='bold')
        axes[2].set_ylabel('MAE')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, model_name='LinearRegression', figsize=(10, 6)):
        """
        Plot actual vs predicted values for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot
        figsize : tuple
            Figure size
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        result = self.results[model_name]
        y_val_pred = result['predictions']['val']
        
        plt.figure(figsize=figsize)
        
        # Scatter plot
        plt.scatter(self.y_val, y_val_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(self.y_val.min(), y_val_pred.min())
        max_val = max(self.y_val.max(), y_val_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual GDP Growth (%)', fontsize=12)
        plt.ylabel('Predicted GDP Growth (%)', fontsize=12)
        plt.title(f'CORRECTED {model_name}: Actual vs Predicted GDP Growth', fontweight='bold')
        
        # Add R² score
        r2 = result['val_metrics']['r2']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def export_results(self, output_dir='results_corrected/'):
        """
        Export results to CSV files.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export model summary
        summary_df = self.get_model_summary()
        summary_df.to_csv(f'{output_dir}corrected_model_performance_summary.csv', index=False)
        
        # Export coefficients
        coef_df = self.get_coefficients()
        coef_df.to_csv(f'{output_dir}corrected_model_coefficients.csv')
        
        # Export predictions
        for name, result in self.results.items():
            pred_df = pd.DataFrame({
                'actual': self.y_val,
                'predicted': result['predictions']['val']
            })
            pred_df.to_csv(f'{output_dir}corrected_{name.lower()}_predictions.csv', index=False)
        
        print(f"CORRECTED results exported to {output_dir}")
    
    def run_full_analysis(self, start_year=1970, end_year=2021, 
                         test_size=0.2, scaler_type='standard', 
                         optimize_hyperparameters=True, use_lagged_variables=True):
        """
        Run the complete CORRECTED regression analysis pipeline.
        
        Parameters:
        -----------
        start_year : int
            Start year for analysis
        end_year : int
            End year for analysis
        test_size : float
            Proportion of data for validation
        scaler_type : str
            Type of scaler ('standard' or 'robust')
        optimize_hyperparameters : bool
            Whether to optimize hyperparameters
        use_lagged_variables : bool
            Whether to use lagged variables to avoid endogeneity
            
        Returns:
        --------
        tuple
            (summary_df, coef_df)
        """
        print("=" * 70)
        print("CORRECTED MACROECONOMIC REGRESSION ANALYSIS")
        print("=" * 70)
        print("Addressing endogeneity issues with proper independent variables")
        print("=" * 70)
        
        # Load and preprocess data
        X, y = self.preprocess_data(start_year, end_year, use_lagged_variables=use_lagged_variables)
        
        # Split data
        self.split_data(X, y, test_size)
        
        # Scale features
        self.scale_features(scaler_type)
        
        # Train models
        self.train_models(optimize_hyperparameters=optimize_hyperparameters)
        
        # Display results
        print("\n" + "=" * 70)
        print("CORRECTED MODEL PERFORMANCE SUMMARY")
        print("=" * 70)
        summary_df = self.get_model_summary()
        print(summary_df.round(4))
        
        print("\n" + "=" * 70)
        print("CORRECTED COEFFICIENTS SUMMARY")
        print("=" * 70)
        coef_df = self.get_coefficients()
        print(coef_df.round(4))
        
        # Export results
        self.export_results()
        
        print("\n" + "=" * 70)
        print("CORRECTED ANALYSIS COMPLETE")
        print("=" * 70)
        print("Key improvements made:")
        print("  - Used absolute investment instead of investment/GDP ratio")
        print("  - Used absolute exports instead of exports/GDP ratio")
        print("  - Added lagged variables to avoid endogeneity")
        print("  - Focused on external/policy variables")
        print("=" * 70)
        
        return summary_df, coef_df


def main():
    """Main function to run the CORRECTED regression analysis."""
    
    # Initialize the corrected regression analysis
    regression = MacroeconomicRegressionCorrected()
    
    # Run full corrected analysis
    summary_df, coef_df = regression.run_full_analysis(
        start_year=1970,
        end_year=2021,
        test_size=0.2,
        scaler_type='standard',
        optimize_hyperparameters=True,
        use_lagged_variables=True
    )
    
    # Create visualizations
    regression.plot_coefficients()
    regression.plot_model_performance()
    regression.plot_predictions('LinearRegression')
    
    return regression, summary_df, coef_df


if __name__ == "__main__":
    regression, summary, coefficients = main()



