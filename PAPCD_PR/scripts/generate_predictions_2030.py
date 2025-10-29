"""
Generador de Predicciones hasta 2030
Usa el mejor modelo (Ridge) para predecir GDP_growth de todos los países hasta 2030
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data():
    """Carga y prepara los datos para predicciones."""
    print("Cargando datos...")
    df = pd.read_csv('../data/external/global_merged_all.csv')
    print(f"Datos cargados: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    
    # Filtrar datos válidos
    df = df[(df['Year'] >= 1970) & (df['Year'] <= 2021)].copy()
    df = df.dropna(subset=['GDP_growth', 'Country', 'Year'])
    
    return df

def engineer_features_for_prediction(df):
    """Feature engineering para predicciones."""
    df = df.copy()
    
    # Variables absolutas
    if 'Gross_capital_formation_percent_GDP' in df.columns and 'GDP_main' in df.columns:
        df['investment_absolute'] = df['Gross_capital_formation_percent_GDP'] * df['GDP_main'] / 100.0
    
    if 'Gross_fixed_capital_formation' in df.columns:
        df['fixed_capital_formation'] = df['Gross_fixed_capital_formation']
    
    # Usar fuentes World Bank
    if 'Exports_wb' in df.columns:
        df['exports_absolute'] = df['Exports_wb']
    elif 'Exports_main' in df.columns:
        df['exports_absolute'] = df['Exports_main']
    
    if 'Imports_wb' in df.columns:
        df['imports_absolute'] = df['Imports_wb']
    elif 'Imports_main' in df.columns:
        df['imports_absolute'] = df['Imports_main']
    
    # Variables estructurales
    structural_mapping = {
        'Government_consumption': 'government_expenditure',
        'Household_consumption': 'household_consumption',
        'Manufacturing_value_added': 'manufacturing_value_added',
        'Agriculture_value_added': 'agriculture_value_added'
    }
    
    for src, tgt in structural_mapping.items():
        if src in df.columns:
            df[tgt] = df[src]
    
    # Variables externas
    external_vars = ['Oil_Price_Brent', 'Federal_Funds_Rate', 'US_Inflation_Index']
    for var in external_vars:
        if var in df.columns:
            df[var.lower()] = df[var]
    
    # Diversificación (HHI simplificado)
    sectors = ['Agriculture_value_added', 'Manufacturing_value_added', 'Construction_value_added',
               'Transport_communication', 'Wholesale_retail_trade', 'Other_activities']
    
    def calc_hhi(row):
        if 'Total_value_added' in df.columns and pd.notna(row.get('Total_value_added', np.nan)) and row.get('Total_value_added', 0) > 0:
            shares = []
            for s in sectors:
                if s in df.columns and pd.notna(row.get(s, np.nan)):
                    shares.append(row[s] / row['Total_value_added'])
            return np.sum(np.square(shares)) if shares else np.nan
        return np.nan
    
    df['diversification_HHI'] = df.apply(calc_hhi, axis=1)
    
    # Términos de intercambio
    if 'exports_absolute' in df.columns and 'imports_absolute' in df.columns:
        df['terms_of_trade'] = df['exports_absolute'] / (df['imports_absolute'] + 1e-8)
        df['terms_of_trade'] = df['terms_of_trade'].clip(upper=df['terms_of_trade'].quantile(0.99))
    
    # Variables lagged
    lag_vars = ['investment_absolute', 'exports_absolute', 'imports_absolute', 
               'government_expenditure', 'household_consumption']
    
    for var in lag_vars:
        if var in df.columns:
            df[f'{var}_lag1'] = df.groupby('Country')[var].shift(1)
            df[f'{var}_lag2'] = df.groupby('Country')[var].shift(2)
    
    # Diferencias temporales
    for var in ['GDP_growth', 'investment_absolute', 'exports_absolute']:
        if var in df.columns:
            df[f'{var}_diff'] = df.groupby('Country')[var].diff()
    
    # Promedios móviles
    for var in ['GDP_growth', 'investment_absolute']:
        if var in df.columns:
            df[f'{var}_ma3'] = df.groupby('Country')[var].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    return df

def train_best_model(df):
    """Entrena el mejor modelo (Ridge) con regularización fuerte."""
    print("Entrenando modelo Ridge...")
    
    # Seleccionar features
    feature_cols = [col for col in df.columns if col not in 
                   ['Country', 'Year', 'GDP_growth', 'CountryID', 'Currency']]
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['GDP_growth']
    
    # Limpiar NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Eliminar variables altamente correlacionadas
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_vars = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    X = X.drop(columns=high_corr_vars)
    
    print(f"Features finales: {len(X.columns)}")
    print(f"Filas de entrenamiento: {len(X):,}")
    
    # Entrenar modelo Ridge con regularización fuerte
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=50.0, max_iter=2000))
    ])
    
    model.fit(X, y)
    
    # Evaluación
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    print(f"R² del modelo: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    return model, X.columns.tolist()

def generate_predictions_2030(model, feature_names, df):
    """Genera predicciones hasta 2030 para todos los países con tendencias realistas."""
    print("Generando predicciones hasta 2030...")
    
    predictions = []
    countries = df['Country'].unique()
    
    for country in countries:
        print(f"Procesando {country}...")
        country_data = df[df['Country'] == country].sort_values('Year').copy()
        
        if len(country_data) < 5:  # Mínimo 5 observaciones
            continue
        
        # Obtener datos históricos recientes (últimos 5 años)
        recent_data = country_data[country_data['Year'] >= 2017].copy()
        if len(recent_data) < 3:
            recent_data = country_data.tail(5)
        
        # Calcular tendencias históricas
        if 'GDP_growth' in recent_data.columns:
            gdp_trend = recent_data['GDP_growth'].mean()
            gdp_volatility = recent_data['GDP_growth'].std()
        else:
            gdp_trend = 3.0  # Valor por defecto
            gdp_volatility = 2.0
        
        # Generar predicciones para 2022-2030 con tendencias
        for i, year in enumerate(range(2022, 2031)):
            # Crear fila de predicción basada en el último año disponible
            last_data = recent_data.iloc[-1].copy()
            pred_row = last_data.copy()
            pred_row['Year'] = year
            
            # Aplicar tendencias y variaciones realistas
            years_ahead = i + 1
            
            # 1. Ajustar variables que cambian con el tiempo
            if 'investment_absolute' in pred_row.index and pd.notna(pred_row['investment_absolute']):
                # Inversión con tendencia ligeramente decreciente a largo plazo
                pred_row['investment_absolute'] *= (0.98 ** years_ahead)
            
            if 'exports_absolute' in pred_row.index and pd.notna(pred_row['exports_absolute']):
                # Exportaciones con crecimiento moderado
                pred_row['exports_absolute'] *= (1.02 ** years_ahead)
            
            if 'imports_absolute' in pred_row.index and pd.notna(pred_row['imports_absolute']):
                # Importaciones con crecimiento similar
                pred_row['imports_absolute'] *= (1.02 ** years_ahead)
            
            # 2. Ajustar variables externas (petróleo, tasas, inflación)
            if 'oil_price_brent' in pred_row.index and pd.notna(pred_row['oil_price_brent']):
                # Precio del petróleo con volatilidad
                oil_trend = 1.0 + (np.random.normal(0, 0.1) * years_ahead)
                pred_row['oil_price_brent'] *= max(0.5, oil_trend)
            
            if 'federal_funds_rate' in pred_row.index and pd.notna(pred_row['federal_funds_rate']):
                # Tasas de interés con ciclos
                rate_cycle = 0.5 + 0.5 * np.sin(years_ahead * 0.5)
                pred_row['federal_funds_rate'] = max(0, pred_row['federal_funds_rate'] * rate_cycle)
            
            # 3. Ajustar variables lagged (usar valores del año anterior)
            lag_vars = [col for col in feature_names if '_lag1' in col or '_lag2' in col]
            for lag_var in lag_vars:
                if lag_var in pred_row.index and pd.notna(pred_row[lag_var]):
                    # Mantener valores lagged del año anterior
                    base_var = lag_var.replace('_lag1', '').replace('_lag2', '')
                    if base_var in pred_row.index:
                        pred_row[lag_var] = pred_row[base_var]
            
            # 4. Ajustar diferencias temporales
            diff_vars = [col for col in feature_names if '_diff' in col]
            for diff_var in diff_vars:
                if diff_var in pred_row.index:
                    # Diferencias con variación aleatoria pequeña
                    pred_row[diff_var] = np.random.normal(0, 0.5)
            
            # 5. Ajustar promedios móviles
            ma_vars = [col for col in feature_names if '_ma3' in col]
            for ma_var in ma_vars:
                if ma_var in pred_row.index and pd.notna(pred_row[ma_var]):
                    # Promedio móvil con tendencia
                    pred_row[ma_var] *= (1.0 + np.random.normal(0, 0.02))
            
            # Preparar features para predicción
            pred_features = pred_row[feature_names].values.reshape(1, -1)
            
            # Convertir a float y verificar que no haya NaN
            pred_features = pred_features.astype(float)
            if not np.isnan(pred_features).any():
                try:
                    gdp_growth_pred = model.predict(pred_features)[0]
                    
                    # Aplicar ajustes finales basados en tendencias históricas
                    # Tendencia de convergencia hacia la media global
                    convergence_factor = 0.1 * years_ahead
                    global_mean = 3.0  # Crecimiento global promedio
                    gdp_growth_pred = gdp_growth_pred * (1 - convergence_factor) + global_mean * convergence_factor
                    
                    # Añadir variabilidad realista
                    noise = np.random.normal(0, gdp_volatility * 0.3)
                    gdp_growth_pred += noise
                    
                    # Limitar valores extremos
                    gdp_growth_pred = np.clip(gdp_growth_pred, -15, 25)
                    
                    predictions.append({
                        'Country': country,
                        'Year': year,
                        'GDP_growth_predicted': gdp_growth_pred,
                        'Region': 'Global',
                        'Income_Level': 'Mixed',
                        'Trend_Component': gdp_trend,
                        'Volatility_Component': gdp_volatility
                    })
                except Exception as e:
                    print(f"Error prediciendo {country} {year}: {str(e)}")
                    continue
    
    return pd.DataFrame(predictions)

def main():
    """Función principal."""
    print("=== GENERADOR DE PREDICCIONES HASTA 2030 ===")
    
    # 1. Cargar datos
    df = load_and_prepare_data()
    
    # 2. Feature engineering
    df = engineer_features_for_prediction(df)
    
    # 3. Entrenar modelo
    model, feature_names = train_best_model(df)
    
    # 4. Generar predicciones
    predictions_df = generate_predictions_2030(model, feature_names, df)
    
    # 5. Guardar resultados
    predictions_df.to_csv('../results/predictions_2030.csv', index=False)
    print(f"\nPredicciones guardadas: {len(predictions_df):,} registros")
    print(f"Países: {predictions_df['Country'].nunique()}")
    print(f"Años: {predictions_df['Year'].min()}-{predictions_df['Year'].max()}")
    
    # Estadísticas
    print(f"\nEstadísticas de predicciones:")
    print(f"GDP Growth promedio: {predictions_df['GDP_growth_predicted'].mean():.2f}%")
    print(f"GDP Growth mediano: {predictions_df['GDP_growth_predicted'].median():.2f}%")
    print(f"Rango: {predictions_df['GDP_growth_predicted'].min():.2f}% - {predictions_df['GDP_growth_predicted'].max():.2f}%")
    
    return predictions_df

if __name__ == "__main__":
    predictions = main()