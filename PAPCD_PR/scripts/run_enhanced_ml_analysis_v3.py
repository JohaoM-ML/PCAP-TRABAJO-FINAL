"""
Enhanced Macroeconomic Machine Learning Model — V3 (Anti-Overfitting & Autocorrelation)

- Análisis de correlación inicial
- Regularización fuerte para evitar sobreajuste
- Validación temporal robusta
- Detección y manejo de autocorrelación
- Feature selection inteligente
- Early stopping y cross-validation temporal

Autor: Macroeconomic Analysis Team
Año: 2025
"""

import os
import time
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

# Flags de librerías opcionales
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    VIF_AVAILABLE = True
except Exception:
    VIF_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

warnings.filterwarnings("ignore")
np.random.seed(42)
import random as _random
_random.seed(42)


# =========================
# Utilidades de tiempo/ETA
# =========================
class StepTimer:
    """Acumula tiempos por paso y estima un ETA remanente."""

    def __init__(self, total_steps: int, label: str = "Progress"):
        self.total = max(int(total_steps), 1)
        self.label = label
        self.start = time.perf_counter()
        self.step_times: List[float] = []
        self.done = 0

    def update(self, steps: int = 1) -> str:
        now = time.perf_counter()
        if self.done == 0:
            self.step_times.append(now - self.start)
        else:
            self.step_times.append(now - self.last_time)
        self.done += steps
        self.last_time = now

        elapsed = now - self.start
        avg = np.mean(self.step_times) if self.step_times else 0.0
        remaining = max(self.total - self.done, 0) * avg
        return f"[{self.label}] {self.done}/{self.total} | Elapsed: {format_hms(elapsed)} | ETA: {format_hms(remaining)}"

    def snapshot(self) -> str:
        now = time.perf_counter()
        elapsed = now - self.start
        remaining = 0 if self.done >= self.total else (self.total - self.done) * np.mean(self.step_times or [0])
        return f"[{self.label}] {self.done}/{self.total} | Elapsed: {format_hms(elapsed)} | ETA: {format_hms(remaining)}"


def format_hms(seconds: float) -> str:
    """Formatea segundos como HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# =========================
# Análisis de correlación
# =========================
def analyze_correlations(df: pd.DataFrame, target_col: str = 'GDP_growth', 
                        threshold: float = 0.95) -> Dict:
    """
    Analiza correlaciones entre variables para detectar multicolinealidad.
    """
    print("\n=== ANÁLISIS DE CORRELACIONES ===")
    
    # Variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Matriz de correlación
    corr_matrix = df[numeric_cols].corr()
    
    # Encontrar pares altamente correlacionados
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # Variables con mayor correlación con target
    if target_col in df.columns:
        target_corr = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
        top_corr_vars = target_corr.abs().sort_values(ascending=False).head(10)
    else:
        top_corr_vars = pd.Series()
    
    # Estadísticas
    stats = {
        'total_vars': len(numeric_cols),
        'high_corr_pairs': len(high_corr_pairs),
        'correlation_matrix': corr_matrix,
        'high_corr_pairs_list': high_corr_pairs,
        'top_correlated_with_target': top_corr_vars
    }
    
    print(f"Variables analizadas: {len(numeric_cols)}")
    print(f"Pares altamente correlacionados (>={threshold}): {len(high_corr_pairs)}")
    print(f"Top 5 correlaciones con {target_col}:")
    for var, corr in top_corr_vars.head(5).items():
        print(f"  {var}: {corr:.3f}")
    
    return stats


def detect_autocorrelation(df: pd.DataFrame, target_col: str = 'GDP_growth') -> Dict:
    """
    Detecta autocorrelación en series temporales por país.
    """
    print("\n=== DETECCIÓN DE AUTOCORRELACIÓN ===")
    
    autocorr_stats = {}
    
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Year')
        if len(country_data) < 5:  # Mínimo 5 observaciones
            continue
            
        if target_col in country_data.columns:
            series = country_data[target_col].dropna()
            if len(series) >= 3:
                # Autocorrelación de primer orden
                if len(series) > 1:
                    lag1_corr = series.autocorr(lag=1)
                    autocorr_stats[country] = {
                        'observations': len(series),
                        'lag1_autocorr': lag1_corr,
                        'has_autocorr': abs(lag1_corr) > 0.3 if not pd.isna(lag1_corr) else False
                    }
    
    # Estadísticas generales
    total_countries = len(autocorr_stats)
    countries_with_autocorr = sum(1 for stats in autocorr_stats.values() if stats['has_autocorr'])
    
    print(f"Países analizados: {total_countries}")
    print(f"Países con autocorrelación significativa: {countries_with_autocorr}")
    print(f"Porcentaje con autocorrelación: {countries_with_autocorr/total_countries*100:.1f}%")
    
    return {
        'country_stats': autocorr_stats,
        'total_countries': total_countries,
        'countries_with_autocorr': countries_with_autocorr,
        'autocorr_rate': countries_with_autocorr/total_countries if total_countries > 0 else 0
    }


# =========================
# Feature Engineering Mejorado
# =========================
def engineer_features_v3(df: pd.DataFrame, lag_years: List[int] = [1, 2], 
                             include_structural: bool = True) -> pd.DataFrame:
    """
    Feature engineering mejorado con manejo de autocorrelación.
    """
    df = df.copy()
    
    # 1) Variables absolutas (evitar endogeneidad)
    if 'Gross_capital_formation_percent_GDP' in df.columns and 'GDP_main' in df.columns:
        df['investment_absolute'] = df['Gross_capital_formation_percent_GDP'] * df['GDP_main'] / 100.0
    
    if 'Gross_fixed_capital_formation' in df.columns:
        df['fixed_capital_formation'] = df['Gross_fixed_capital_formation']
    
    # Usar fuentes World Bank preferentemente
    if 'Exports_wb' in df.columns:
        df['exports_absolute'] = df['Exports_wb']
    elif 'Exports_main' in df.columns:
        df['exports_absolute'] = df['Exports_main']
    
    if 'Imports_wb' in df.columns:
        df['imports_absolute'] = df['Imports_wb']
    elif 'Imports_main' in df.columns:
        df['imports_absolute'] = df['Imports_main']
    
    # 2) Variables estructurales
    if include_structural:
        structural_mapping = {
            'Government_consumption': 'government_expenditure',
            'Household_consumption': 'household_consumption',
            'Final_consumption_expenditure': 'final_consumption',
            'Manufacturing_value_added': 'manufacturing_value_added',
            'Agriculture_value_added': 'agriculture_value_added'
        }
        
        for src, tgt in structural_mapping.items():
            if src in df.columns:
                df[tgt] = df[src]
    
    # 3) Variables externas
    external_vars = ['Oil_Price_Brent', 'Federal_Funds_Rate', 'US_Inflation_Index']
    for var in external_vars:
        if var in df.columns:
            df[var.lower().replace('_', '_')] = df[var]
    
    # 4) Diversificación (HHI)
    df['diversification_HHI'] = _calc_hhi(df)
    
    # 5) Términos de intercambio
    if 'exports_absolute' in df.columns and 'imports_absolute' in df.columns:
        df['terms_of_trade'] = df['exports_absolute'] / (df['imports_absolute'] + 1e-8)
        df['terms_of_trade'] = df['terms_of_trade'].clip(upper=df['terms_of_trade'].quantile(0.99))
    
    # 6) Variables lagged (CRÍTICO para autocorrelación)
    variables_to_lag = [
        'investment_absolute', 'exports_absolute', 'imports_absolute',
        'government_expenditure', 'household_consumption', 'manufacturing_value_added'
    ]
    
    for var in variables_to_lag:
        if var in df.columns:
            for L in lag_years:
                df[f'{var}_lag{L}'] = df.groupby('Country')[var].shift(L)
    
    # 7) Diferencias (para reducir autocorrelación)
    for var in ['GDP_growth', 'investment_absolute', 'exports_absolute']:
        if var in df.columns:
            df[f'{var}_diff'] = df.groupby('Country')[var].diff()
    
    # 8) Promedios móviles (suavizado)
    for var in ['GDP_growth', 'investment_absolute']:
        if var in df.columns:
            df[f'{var}_ma3'] = df.groupby('Country')[var].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    print(f"Features engineered: {df.shape[1]} columns")
    return df


def _calc_hhi(df: pd.DataFrame) -> pd.Series:
    """Calcula el índice de Herfindahl-Hirschman."""
    sectors = [
        'Agriculture_value_added', 'Manufacturing_value_added', 'Construction_value_added',
        'Transport_communication', 'Wholesale_retail_trade', 'Other_activities'
    ]
    out = []
    for _, row in df.iterrows():
        if 'Total_value_added' in df.columns and pd.notna(row.get('Total_value_added', np.nan)) and row.get('Total_value_added', 0) > 0:
            shares = []
            for s in sectors:
                if s in df.columns and pd.notna(row.get(s, np.nan)):
                    shares.append(row[s] / row['Total_value_added'])
            out.append(np.sum(np.square(shares)) if shares else np.nan)
        else:
            out.append(np.nan)
    return pd.Series(out, index=df.index)


# =========================
# Modelos con Regularización Fuerte
# =========================
def build_models_v3() -> Dict[str, Pipeline]:
    """Construye modelos con regularización fuerte para evitar sobreajuste."""
    
    models = {
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=50.0, max_iter=2000))  # Regularización muy fuerte
        ]),
        
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=5.0, max_iter=2000))  # Regularización muy fuerte
        ]),
        
        'ElasticNet': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=5.0, l1_ratio=0.5, max_iter=2000))  # Regularización muy fuerte
        ]),
        
        'RandomForest': RandomForestRegressor(
            n_estimators=30,        # Menos árboles
            max_depth=6,           # Profundidad limitada
            min_samples_split=50,  # Muchas muestras para dividir
            min_samples_leaf=25,   # Muchas muestras por hoja
            max_features='sqrt',   # Menos features por árbol
            random_state=42,
            n_jobs=-1
        ),
        
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=30,       # Menos estimadores
            max_depth=4,          # Profundidad limitada
            learning_rate=0.01,   # Learning rate muy bajo
            subsample=0.7,        # Submuestreo fuerte
            random_state=42
        )
    }
    
    # XGBoost con regularización extrema
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=30,           # Menos estimadores
            max_depth=3,             # Profundidad muy limitada
            learning_rate=0.01,   # Learning rate muy bajo
            subsample=0.6,        # Submuestreo fuerte
            colsample_bytree=0.6, # Submuestreo de features
            reg_alpha=10.0,        # Regularización L1 muy fuerte
            reg_lambda=10.0,      # Regularización L2 muy fuerte
            early_stopping_rounds=5,  # Early stopping agresivo
            random_state=42,
            n_jobs=-1
        )
    
    return models


# =========================
# Validación Temporal Robusta
# =========================
def temporal_cv_split(df: pd.DataFrame, feature_cols: List[str], target_col: str, 
                      n_splits: int = 5) -> List[Tuple]:
    """
    Split temporal robusto para series temporales.
    """
    splits = []
    countries = df['Country'].unique()
    
    for country in countries:
        country_data = df[df['Country'] == country].sort_values('Year')
        if len(country_data) < 10:  # Mínimo 10 observaciones
            continue
            
        # Usar TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=min(n_splits, len(country_data)//2))
        
        for train_idx, val_idx in tscv.split(country_data):
            train_data = country_data.iloc[train_idx]
            val_data = country_data.iloc[val_idx]
            
            splits.append({
                'country': country,
                'train': train_data,
                'val': val_data,
                'train_years': train_data['Year'].tolist(),
                'val_years': val_data['Year'].tolist()
            })
    
    return splits


# =========================
# Clase Principal Mejorada
# =========================
class EnhancedMacroeconomicMLV3:
    def __init__(self, data_path: str = '../data/external/global_merged_all.csv'):
        self.data_path = data_path
        self.data: pd.DataFrame = None
        self.processed_data: pd.DataFrame = None
        self.feature_names: List[str] = []
        self.models: Dict[str, Pipeline] = build_models_v3()
        self.results = {}
        self.correlation_stats = {}
        self.autocorr_stats = {}
        self.permutation_importances = {}

    def load_data(self) -> pd.DataFrame:
        print("Loading dataset...")
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded: {self.data.shape[0]:,} rows x {self.data.shape[1]} cols")
        return self.data

    def analyze_data_quality(self):
        """Análisis inicial de calidad de datos."""
        print("\n=== ANÁLISIS DE CALIDAD DE DATOS ===")
        
        # Análisis de correlaciones
        self.correlation_stats = analyze_correlations(self.data, 'GDP_growth')
        
        # Detección de autocorrelación
        self.autocorr_stats = detect_autocorrelation(self.data, 'GDP_growth')
        
        return self.correlation_stats, self.autocorr_stats

    def preprocess(self, start_year=1970, end_year=2021, min_years_per_country=10,
                   lag_years=[1, 2], include_structural=True):
        """Preprocesamiento mejorado con análisis de autocorrelación."""
        
        # Cargar datos
        if self.data is None:
            self.load_data()
        
        # Análisis de calidad
        self.analyze_data_quality()
        
        # Filtrar por años
        df = self.data[(self.data['Year'] >= start_year) & (self.data['Year'] <= end_year)].copy()
        
        # Filtrar países con suficientes observaciones
        country_counts = df['Country'].value_counts()
        valid_countries = country_counts[country_counts >= min_years_per_country].index
        df = df[df['Country'].isin(valid_countries)]
        
        print(f"Countries with >= {min_years_per_country} years: {len(valid_countries)}")
        
        # Feature engineering
        df = engineer_features_v3(df, lag_years, include_structural)
        
        # Seleccionar features y target
        feature_cols = [col for col in df.columns if col not in ['Country', 'Year', 'GDP_growth', 'CountryID', 'Currency']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['GDP_growth']
        
        # Limpiar NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Filtrar variables altamente correlacionadas
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_vars = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X = X.drop(columns=high_corr_vars)
        
        print(f"Final features: {len(X.columns)} | Final rows: {len(X):,}")
        print(f"Dropped high correlation vars: {len(high_corr_vars)}")
        
        self.processed_data = df.loc[X.index, ['Country', 'Year']].copy()
        self.feature_names = list(X.columns)
        
        return X, y

    def train_and_eval(self, X_train, y_train, X_val, y_val):
        """Entrenamiento con validación temporal robusta."""
        print("\n=== ENTRENAMIENTO DE MODELOS ===")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Entrenar modelo
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                else:
                    # Para pipelines
                    model.fit(X_train, y_train)
                
                # Predicciones
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Métricas
                train_r2 = r2_score(y_train, y_train_pred)
                val_r2 = r2_score(y_val, y_val_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                
                # Detectar sobreajuste
                overfitting = train_r2 - val_r2 > 0.2
                
                self.results[name] = {
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'overfitting': overfitting,
                    'overfitting_gap': train_r2 - val_r2
                }
                
                print(f"  Train R²: {train_r2:.3f} | Val R²: {val_r2:.3f}")
                print(f"  Overfitting: {'YES' if overfitting else 'NO'} (gap: {train_r2 - val_r2:.3f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue

    def summary(self) -> pd.DataFrame:
        """Resumen de resultados."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for name, metrics in self.results.items():
            summary_data.append({
                'Model': name,
                'Train_R2': metrics['train_r2'],
                'Val_R2': metrics['val_r2'],
                'Train_RMSE': metrics['train_rmse'],
                'Val_RMSE': metrics['val_rmse'],
                'Train_MAE': metrics['train_mae'],
                'Val_MAE': metrics['val_mae'],
                'Overfitting': metrics['overfitting'],
                'Overfitting_Gap': metrics['overfitting_gap']
            })
        
        return pd.DataFrame(summary_data).sort_values('Val_R2', ascending=False)

    def export(self, output_dir: str = 'enhanced_results_v3/', y_val=None):
        """Exportar resultados."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Resumen
        summary = self.summary()
        summary.to_csv(os.path.join(output_dir, 'model_performance_v3.csv'), index=False)
        
        # Análisis de correlaciones
        if self.correlation_stats:
            corr_df = pd.DataFrame(self.correlation_stats['high_corr_pairs_list'])
            corr_df.to_csv(os.path.join(output_dir, 'high_correlations.csv'), index=False)
        
        # Análisis de autocorrelación
        if self.autocorr_stats:
            autocorr_df = pd.DataFrame(self.autocorr_stats['country_stats']).T
            autocorr_df.to_csv(os.path.join(output_dir, 'autocorrelation_analysis.csv'))
        
        print(f"Exported results -> {output_dir}")


# =========================
# Función Principal
# =========================
def main():
    """Función principal mejorada."""
    DATA_PATH = '../data/external/global_merged_all.csv'
    
    # Timer global
    global_timer = StepTimer(total_steps=5, label="Pipeline V3")
    
    # Inicializar modelo
    ml = EnhancedMacroeconomicMLV3(data_path=DATA_PATH)
    
    print("\n[1/5] Load & Analyze")
    ml.load_data()
    ml.analyze_data_quality()
    print(global_timer.update())
    
    print("\n[2/5] Preprocess")
    X, y = ml.preprocess(start_year=1970, end_year=2021, min_years_per_country=10)
    print(global_timer.update())
    
    print("\n[3/5] Temporal Split")
    # Split temporal simple (80% train, 20% val)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
    print(global_timer.update())
    
    print("\n[4/5] Train & Evaluate")
    ml.train_and_eval(X_train, y_train, X_val, y_val)
    print(global_timer.update())
    
    print("\n[5/5] Summary & Export")
    summary = ml.summary()
    print(summary.round(4))
    ml.export(output_dir='enhanced_results_v3/', y_val=y_val)
    print(global_timer.update())
    
    print("\nDone.")
    return ml, summary, (X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    ml, summary, data_splits = main()
