"""
Comparison script to demonstrate the difference between the original
and corrected regression approaches, highlighting the endogeneity issue.
"""

import sys
import os
sys.path.append('src')

from macroeconomic_regression import MacroeconomicRegression
from macroeconomic_regression_corrected import MacroeconomicRegressionCorrected
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_approaches():
    """Compare original vs corrected regression approaches."""
    
    print("=" * 80)
    print("COMPARISON: ORIGINAL vs CORRECTED REGRESSION APPROACHES")
    print("=" * 80)
    print("This comparison highlights the endogeneity issue in the original approach")
    print("=" * 80)
    
    # Run original analysis
    print("\n1. RUNNING ORIGINAL ANALYSIS (with endogeneity issues)")
    print("-" * 60)
    
    original_regression = MacroeconomicRegression()
    original_summary, original_coef = original_regression.run_full_analysis(
        start_year=2000,  # Use recent data for faster processing
        end_year=2021,
        test_size=0.2,
        scaler_type='standard',
        optimize_hyperparameters=False  # Skip for speed
    )
    
    # Run corrected analysis
    print("\n2. RUNNING CORRECTED ANALYSIS (addressing endogeneity)")
    print("-" * 60)
    
    corrected_regression = MacroeconomicRegressionCorrected()
    corrected_summary, corrected_coef = corrected_regression.run_full_analysis(
        start_year=2000,  # Use recent data for faster processing
        end_year=2021,
        test_size=0.2,
        scaler_type='standard',
        optimize_hyperparameters=False,  # Skip for speed
        use_lagged_variables=True
    )
    
    # Compare results
    print("\n3. COMPARING RESULTS")
    print("-" * 60)
    
    print("\nModel Performance Comparison:")
    print("\nORIGINAL APPROACH:")
    print(original_summary.round(4))
    
    print("\nCORRECTED APPROACH:")
    print(corrected_summary.round(4))
    
    # Feature comparison
    print("\nFeature Comparison:")
    print("\nORIGINAL FEATURES (with endogeneity issues):")
    print("  - investment_GDP: Investment/GDP ratio")
    print("  - exports_GDP: Exports/GDP ratio")
    print("  - inflation: US inflation proxy")
    print("  - population: Log population")
    print("  - oil_price: Oil price")
    print("  - fed_rate: Federal funds rate")
    
    print("\nCORRECTED FEATURES (addressing endogeneity):")
    print("  - investment_absolute_lag1: Lagged absolute investment")
    print("  - exports_absolute_lag1: Lagged absolute exports")
    print("  - inflation: US inflation proxy")
    print("  - population: Log population")
    print("  - oil_price: Oil price")
    print("  - fed_rate: Federal funds rate")
    print("  - exchange_rate: Exchange rate")
    print("  - trade_volume: Total trade volume")
    print("  - government_consumption: Government consumption")
    print("  - diversification_HHI: Economic diversification")
    
    # Coefficient comparison
    print("\nCoefficient Comparison (Linear Regression):")
    print("\nORIGINAL COEFFICIENTS:")
    if 'LinearRegression' in original_coef.columns:
        for feature, coef in original_coef['LinearRegression'].items():
            print(f"  {feature}: {coef:.4f}")
    
    print("\nCORRECTED COEFFICIENTS:")
    if 'LinearRegression' in corrected_coef.columns:
        for feature, coef in corrected_coef['LinearRegression'].items():
            print(f"  {feature}: {coef:.4f}")
    
    # Economic interpretation
    print("\n4. ECONOMIC INTERPRETATION")
    print("-" * 60)
    
    print("\nPROBLEMS WITH ORIGINAL APPROACH:")
    print("1. Circular Causality:")
    print("   - investment_GDP = Investment / GDP")
    print("   - Using investment/GDP to predict GDP growth creates endogeneity")
    print("   - GDP appears on both sides of the equation")
    
    print("\n2. Statistical Issues:")
    print("   - Violates independence assumption of regression")
    print("   - Coefficients may be biased and inconsistent")
    print("   - R² scores may be artificially inflated")
    
    print("\n3. Economic Logic:")
    print("   - Investment/GDP ratio is endogenous to GDP growth")
    print("   - High growth periods may have high investment ratios")
    print("   - This creates reverse causality")
    
    print("\nIMPROVEMENTS IN CORRECTED APPROACH:")
    print("1. Absolute Values:")
    print("   - Use absolute investment levels instead of ratios")
    print("   - Use absolute export levels instead of ratios")
    print("   - Avoids GDP appearing in both dependent and independent variables")
    
    print("\n2. Lagged Variables:")
    print("   - Use lagged investment and exports")
    print("   - Captures temporal relationships")
    print("   - Reduces endogeneity concerns")
    
    print("\n3. External Variables:")
    print("   - Focus on truly independent variables")
    print("   - Oil prices, exchange rates, Fed rates")
    print("   - Variables that are exogenous to GDP growth")
    
    # Performance comparison
    print("\n5. PERFORMANCE COMPARISON")
    print("-" * 60)
    
    original_best = original_summary.loc[original_summary['Val_R2'].idxmax()]
    corrected_best = corrected_summary.loc[corrected_summary['Val_R2'].idxmax()]
    
    print(f"\nORIGINAL BEST MODEL: {original_best['Model']}")
    print(f"  R² Score: {original_best['Val_R2']:.4f}")
    print(f"  RMSE: {original_best['Val_RMSE']:.4f}")
    
    print(f"\nCORRECTED BEST MODEL: {corrected_best['Model']}")
    print(f"  R² Score: {corrected_best['Val_R2']:.4f}")
    print(f"  RMSE: {corrected_best['Val_RMSE']:.4f}")
    
    print(f"\nR² DIFFERENCE: {original_best['Val_R2'] - corrected_best['Val_R2']:.4f}")
    if original_best['Val_R2'] > corrected_best['Val_R2']:
        print("  Note: Lower R² in corrected model is expected and more reliable")
        print("  Original model may have artificially inflated R² due to endogeneity")
    
    # Create comparison visualization
    create_comparison_visualization(original_summary, corrected_summary, 
                                  original_coef, corrected_coef)
    
    return original_regression, corrected_regression

def create_comparison_visualization(original_summary, corrected_summary, 
                                  original_coef, corrected_coef):
    """Create visualization comparing the two approaches."""
    
    print("\n6. CREATING COMPARISON VISUALIZATION")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. R² Comparison
    ax1 = axes[0, 0]
    models = original_summary['Model']
    original_r2 = original_summary['Val_R2']
    corrected_r2 = corrected_summary['Val_R2']
    
    x = range(len(models))
    width = 0.35
    
    ax1.bar([xi - width/2 for xi in x], original_r2, width, label='Original', alpha=0.7, color='red')
    ax1.bar([xi + width/2 for xi in x], corrected_r2, width, label='Corrected', alpha=0.7, color='blue')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE Comparison
    ax2 = axes[0, 1]
    original_rmse = original_summary['Val_RMSE']
    corrected_rmse = corrected_summary['Val_RMSE']
    
    ax2.bar([xi - width/2 for xi in x], original_rmse, width, label='Original', alpha=0.7, color='red')
    ax2.bar([xi + width/2 for xi in x], corrected_rmse, width, label='Corrected', alpha=0.7, color='blue')
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Coefficient Comparison (Linear Regression)
    ax3 = axes[1, 0]
    if 'LinearRegression' in original_coef.columns and 'LinearRegression' in corrected_coef.columns:
        # Get common features
        common_features = set(original_coef.index) & set(corrected_coef.index)
        if common_features:
            common_features = list(common_features)
            original_coefs = [original_coef.loc[feature, 'LinearRegression'] for feature in common_features]
            corrected_coefs = [corrected_coef.loc[feature, 'LinearRegression'] for feature in common_features]
            
            x_coef = range(len(common_features))
            ax3.bar([xi - width/2 for xi in x_coef], original_coefs, width, label='Original', alpha=0.7, color='red')
            ax3.bar([xi + width/2 for xi in x_coef], corrected_coefs, width, label='Corrected', alpha=0.7, color='blue')
            
            ax3.set_xlabel('Features')
            ax3.set_ylabel('Coefficient Value')
            ax3.set_title('Coefficient Comparison (Linear Regression)', fontweight='bold')
            ax3.set_xticks(x_coef)
            ax3.set_xticklabels(common_features, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    COMPARISON SUMMARY
    
    ORIGINAL APPROACH:
    • Features: {len(original_coef.index)} variables
    • Best R²: {original_summary['Val_R2'].max():.4f}
    • Best RMSE: {original_summary['Val_RMSE'].min():.4f}
    • Issues: Endogeneity, circular causality
    
    CORRECTED APPROACH:
    • Features: {len(corrected_coef.index)} variables
    • Best R²: {corrected_summary['Val_R2'].max():.4f}
    • Best RMSE: {corrected_summary['Val_RMSE'].min():.4f}
    • Improvements: Lagged variables, absolute values
    
    KEY DIFFERENCES:
    • Original uses GDP ratios (endogenous)
    • Corrected uses absolute values + lags
    • Corrected approach is more econometrically sound
    • Lower R² in corrected model is expected and reliable
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('regression_approaches_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison visualization saved as 'regression_approaches_comparison.png'")

def main():
    """Main function to run the comparison."""
    
    # Run comparison
    original_regression, corrected_regression = compare_approaches()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("Key Takeaways:")
    print("1. Original approach has endogeneity issues")
    print("2. Corrected approach uses proper independent variables")
    print("3. Lower R² in corrected model is more reliable")
    print("4. Economic interpretation is more valid in corrected approach")
    print("=" * 80)
    
    return original_regression, corrected_regression

if __name__ == "__main__":
    original, corrected = main()

