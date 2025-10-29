"""
Generador de Predicciones hasta 2030 (V2 Corregido)
Usa un modelo Ridge para predecir GDP_growth de todos los países hasta 2030,
evitando fuga del objetivo, con rezagos correctos y simulación secuencial.

Salida: predictions_2030.csv en la carpeta padre de este script (BASE.parent),
coincidiendo con la ruta que usa el dashboard.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("default")  # no ocultamos avisos útiles

import os
from pathlib import Path
import random
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ---------------------------
# Semillas para reproducibilidad
# ---------------------------
random.seed(42)
np.random.seed(42)


# ---------------------------
# Utilidades de ruta
# ---------------------------
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE.parent / "data" / "external" / "global_merged_all.csv"
OUT_PATH = BASE.parent / "predictions_2030.csv"


# ---------------------------
# Carga y preparación
# ---------------------------
def load_and_prepare_data() -> pd.DataFrame:
    """Carga y prepara los datos históricos para entrenamiento."""
    print("Cargando datos...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Datos cargados: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # Filtrar rango y columnas mínimas
    if "Year" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'Year'.")
    if "Country" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'Country'.")
    if "GDP_growth" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'GDP_growth'.")

    df = df[(df["Year"] >= 1970) & (df["Year"] <= 2021)].copy()
    # No tiramos filas a ciegas aún: dejaremos que el pipeline impute.
    # Pero sí garantizamos Country/Year/GDP_growth presentes.
    df = df.dropna(subset=["Country", "Year", "GDP_growth"])
    # Orden correcto para lags
    df = df.sort_values(["Country", "Year"]).reset_index(drop=True)
    return df


# ---------------------------
# Feature engineering (sin fuga del objetivo)
# ---------------------------
def engineer_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features macro evitando usar transformaciones del objetivo (p.ej., no usar GDP_growth_ma/diff).
    Incluye rezagos para variables potencialmente endógenas y externas.
    """
    df = df.copy().sort_values(["Country", "Year"]).reset_index(drop=True)

    # 1) Absolutas con fallback
    if {"Gross_capital_formation_percent_GDP", "GDP_main"}.issubset(df.columns):
        df["investment_absolute"] = df["Gross_capital_formation_percent_GDP"] * df["GDP_main"] / 100.0
    if "Gross_fixed_capital_formation" in df.columns:
        df["fixed_capital_formation"] = df["Gross_fixed_capital_formation"]

    # Exportaciones/Importaciones absolutas
    if "Exports_wb" in df.columns:
        df["exports_absolute"] = df["Exports_wb"]
    elif "Exports_main" in df.columns:
        df["exports_absolute"] = df["Exports_main"]

    if "Imports_wb" in df.columns:
        df["imports_absolute"] = df["Imports_wb"]
    elif "Imports_main" in df.columns:
        df["imports_absolute"] = df["Imports_main"]

    # 2) Variables estructurales (mapeo simple)
    structural_mapping = {
        "Government_consumption": "government_expenditure",
        "Household_consumption": "household_consumption",
        "Manufacturing_value_added": "manufacturing_value_added",
        "Agriculture_value_added": "agriculture_value_added",
        "Construction_value_added": "construction_value_added",
        "Transport_communication": "transport_communication",
        "Wholesale_retail_trade": "wholesale_retail_trade",
        "Other_activities": "other_activities",
        "Total_value_added": "total_value_added",
    }
    for src, tgt in structural_mapping.items():
        if src in df.columns:
            df[tgt] = df[src]

    # 3) Diversificación (HHI) vectorizado
    sectors = [
        "agriculture_value_added", "manufacturing_value_added", "construction_value_added",
        "transport_communication", "wholesale_retail_trade", "other_activities"
    ]
    # Asegura columnas existentes
    sector_cols = [c for c in sectors if c in df.columns]
    if "total_value_added" in df.columns and sector_cols:
        # shares = sector / total
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = df[sector_cols].div(df["total_value_added"].replace(0, np.nan), axis=0)
        df["diversification_HHI"] = (shares ** 2).sum(axis=1)
    else:
        df["diversification_HHI"] = np.nan

    # 4) Términos de intercambio
    if {"exports_absolute", "imports_absolute"}.issubset(df.columns):
        df["terms_of_trade"] = df["exports_absolute"] / (df["imports_absolute"].replace(0, np.nan))
        # Cap para outliers
        df["terms_of_trade"] = df["terms_of_trade"].clip(upper=df["terms_of_trade"].quantile(0.99))

    # 5) Variables externas (en minúsculas + rezagos)
    # Creamos versiones lower y lags para evitar endogeneidad contemporánea
    external_pairs = [
        ("Oil_Price_Brent", "oil_price_brent"),
        ("Federal_Funds_Rate", "federal_funds_rate"),
        ("US_Inflation_Index", "us_inflation_index"),
    ]
    for cap, low in external_pairs:
        if cap in df.columns:
            df[low] = df[cap]
            df[f"{low}_lag1"] = df.groupby("Country")[low].shift(1)
            df[f"{low}_lag2"] = df.groupby("Country")[low].shift(2)

    # 6) Rezagos económicos plausibles
    lag_vars = [
        "investment_absolute", "exports_absolute", "imports_absolute",
        "fixed_capital_formation", "government_expenditure", "household_consumption",
        "manufacturing_value_added", "agriculture_value_added",
    ]
    for var in lag_vars:
        if var in df.columns:
            df[f"{var}_lag1"] = df.groupby("Country")[var].shift(1)
            df[f"{var}_lag2"] = df.groupby("Country")[var].shift(2)

    # ⚠️ No crear diffs/rolling de GDP_growth (fuga del objetivo)
    return df


# ---------------------------
# Entrenamiento del modelo
# ---------------------------
def train_best_model(df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """
    Entrena un Ridge con imputación y escalado.
    Selecciona columnas numéricas seguras (lista blanca vía exclusión) y quita duplicadas caps/lower.
    """
    print("Entrenando modelo Ridge...")

    exclude = {"Country", "Year", "GDP_growth", "CountryID", "Currency"}
    whitelist = [c for c in df.columns if c not in exclude]

    X = df[whitelist].select_dtypes(include=[np.number]).copy()
    y = df["GDP_growth"].astype(float).copy()

    # Elimina duplicados obvios (si coexisten caps/lower de externas)
    dup_pairs = [
        ("Oil_Price_Brent", "oil_price_brent"),
        ("Federal_Funds_Rate", "federal_funds_rate"),
        ("US_Inflation_Index", "us_inflation_index"),
    ]
    for cap, low in dup_pairs:
        if cap in X.columns and low in X.columns:
            X.drop(columns=[cap], inplace=True)

    # Filtro simple de correlación muy alta
    if X.shape[1] > 1:
        corr = X.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        if to_drop:
            X.drop(columns=to_drop, inplace=True)

    print(f"Features finales: {X.shape[1]}")
    print(f"Filas de entrenamiento: {len(X):,}")

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=50.0, max_iter=5000, random_state=42)),
    ])
    model.fit(X, y)

    # Métricas in-sample (solo diagnósticas; sin validación temporal aquí)
    y_hat = model.predict(X)
    r2 = model.score(X, y)
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    print(f"R² (train): {r2:.3f}")
    print(f"RMSE (train): {rmse:.3f}")

    return model, X.columns.tolist()


# ---------------------------
# Predicción 2022–2030 secuencial (sin fuga y con imputación)
# ---------------------------
def generate_predictions_2030(model: Pipeline, feature_names: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Simula 2022–2030 por país, actualizando lags correctamente y
    utilizando el imputador del pipeline para evitar NaN (no perder países).
    """
    print("Generando predicciones hasta 2030...")
    rng = np.random.default_rng(42)

    preds = []
    countries = df["Country"].unique()

    # Medianas de entrenamiento del imputador (si está en el pipeline)
    train_medians = None
    try:
        train_medians = pd.Series(model.named_steps["imputer"].statistics_, index=feature_names)
    except Exception:
        pass

    # Variables base a actualizar (si existen en feature_names)
    base_vars = [v for v in [
        "investment_absolute", "exports_absolute", "imports_absolute",
        "government_expenditure", "household_consumption",
        "fixed_capital_formation", "manufacturing_value_added", "agriculture_value_added",
        "oil_price_brent", "federal_funds_rate", "us_inflation_index",
    ] if v in feature_names]

    for country in countries:
        cdf = df[df["Country"] == country].sort_values("Year").copy()
        if len(cdf) < 5:
            continue

        recent = cdf.tail(5).copy()
        # Tendencia/volatilidad seguras
        gdp_trend = float(recent["GDP_growth"].mean()) if "GDP_growth" in recent else 3.0
        if np.isnan(gdp_trend):
            gdp_trend = 3.0
        gdp_vol = float(recent["GDP_growth"].std()) if "GDP_growth" in recent else 2.0
        if (gdp_vol is None) or np.isnan(gdp_vol) or gdp_vol == 0:
            gdp_vol = 2.0

        # Estado de features (último año histórico)
        # state: vector completo en el espacio de feature_names para el año base
        last_row = recent.iloc[-1]
        state = pd.Series(index=feature_names, dtype="float64")
        for col in feature_names:
            state[col] = last_row[col] if col in last_row.index else np.nan

        # Base subyacente para evolucionar año a año
        base_state = {k: last_row.get(k, np.nan) for k in base_vars}

        for year in range(2022, 2031):
            # 1) Evolución simple de bases
            if "investment_absolute" in base_state and pd.notna(base_state["investment_absolute"]):
                base_state["investment_absolute"] *= 0.98  # ligera convergencia
            if "exports_absolute" in base_state and pd.notna(base_state["exports_absolute"]):
                base_state["exports_absolute"] *= 1.02
            if "imports_absolute" in base_state and pd.notna(base_state["imports_absolute"]):
                base_state["imports_absolute"] *= 1.02
            if "fixed_capital_formation" in base_state and pd.notna(base_state["fixed_capital_formation"]):
                base_state["fixed_capital_formation"] *= 1.01

            if "oil_price_brent" in base_state and pd.notna(base_state["oil_price_brent"]):
                base_state["oil_price_brent"] *= max(0.5, 1.0 + rng.normal(0, 0.1))
            if "federal_funds_rate" in base_state and pd.notna(base_state["federal_funds_rate"]):
                k = year - 2021
                base_state["federal_funds_rate"] = max(0.0, base_state["federal_funds_rate"] * (0.5 + 0.5 * np.sin(0.5 * k)))
            if "us_inflation_index" in base_state and pd.notna(base_state["us_inflation_index"]):
                base_state["us_inflation_index"] *= (1.0 + rng.normal(0, 0.02))

            # 2) Construye la fila de features para este año a partir del estado previo
            row = state.copy()

            # Actualiza bases contemporáneas con base_state
            for b in base_vars:
                if b in row.index:
                    row[b] = base_state.get(b, row.get(b, np.nan))
                # Lags correctos: t lag1 = valor t-1; t lag2 = lag1 de t-1
                if f"{b}_lag1" in row.index:
                    row[f"{b}_lag1"] = state.get(b, np.nan)
                if f"{b}_lag2" in row.index:
                    row[f"{b}_lag2"] = state.get(f"{b}_lag1", np.nan)

            # 3) Imputación previa a predicción (por si queda algún NaN)
            if train_medians is not None:
                row = row.fillna(train_medians)
            else:
                row = row.fillna(row.median())

            Xf = row.values.astype(float).reshape(1, -1)
            try:
                pred = float(model.predict(Xf)[0])
            except Exception as e:
                print(f"[WARN] Error prediciendo {country} {year}: {e}")
                continue

            # 4) Convergencia global más suave (máx 50% hacia media global ~3%)
            k = year - 2021
            conv = min(0.05 * k, 0.5)
            global_mean = 3.0
            pred = pred * (1 - conv) + global_mean * conv

            # 5) Ruido razonable según volatilidad histórica
            pred += float(rng.normal(0, gdp_vol * 0.2))

            # 6) Límite plausible
            pred = float(np.clip(pred, -15, 25))

            preds.append({
                "Country": country,
                "Year": year,
                "GDP_growth_predicted": pred,
                "Trend_Component": gdp_trend,
                "Volatility_Component": gdp_vol,
            })

            # 7) Avanza estado para el próximo año
            state = row.copy()

    return pd.DataFrame(preds)


# ---------------------------
# Main
# ---------------------------
def main() -> pd.DataFrame:
    print("=== GENERADOR DE PREDICCIONES HASTA 2030 (V2) ===")
    print(f"Dataset histórico: {DATA_PATH}")
    print(f"Salida de predicciones: {OUT_PATH}")

    # 1. Cargar datos
    df = load_and_prepare_data()

    # 2. Feature engineering
    df_fe = engineer_features_for_prediction(df)

    # 3. Entrenar modelo
    model, feature_names = train_best_model(df_fe)

    # 4. Generar predicciones
    predictions_df = generate_predictions_2030(model, feature_names, df_fe)

    # 5. Guardar resultados
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(OUT_PATH, index=False)

    # Reporte
    print(f"\nPredicciones guardadas: {len(predictions_df):,} registros")
    if not predictions_df.empty:
        print(f"Países: {predictions_df['Country'].nunique()}")
        print(f"Años: {int(predictions_df['Year'].min())}-{int(predictions_df['Year'].max())}")
        print("\nEstadísticas de predicciones:")
        print(f"  GDP Growth promedio: {predictions_df['GDP_growth_predicted'].mean():.2f}%")
        print(f"  GDP Growth mediano : {predictions_df['GDP_growth_predicted'].median():.2f}%")
        print(f"  Rango              : {predictions_df['GDP_growth_predicted'].min():.2f}%  —  {predictions_df['GDP_growth_predicted'].max():.2f}%")
    else:
        print("⚠️ No se generaron predicciones (revisar datos de entrada).")

    return predictions_df


if __name__ == "__main__":
    _preds = main()
