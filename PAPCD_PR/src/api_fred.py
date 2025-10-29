# ==========================================================
# 🪙 API: Reserva Federal (FRED)
# ==========================================================
# Descarga datos macrofinancieros globales:
#   - Precio del petróleo Brent (DCOILBRENTEU)
#   - Tasa de Fondos Federales (FEDFUNDS)
#   - Índice de Precios al Consumidor (CPIAUCSL)
# ==========================================================

import os
import requests
import pandas as pd
from dotenv import load_dotenv
import warnings
import logging
import urllib3

# ----------------------------------------------------------
# 🔇 Limpieza total de logs y warnings
# ----------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

# ----------------------------------------------------------
# 📊 Función principal
# ----------------------------------------------------------
def fetch_fred_data(save_path="data/external/fred_data.csv"):
    """
    Descarga datos globales desde la API REST oficial de FRED.
    Incluye variables macroeconómicas clave:
        - DCOILBRENTEU : Precio del petróleo Brent (USD/barril)
        - FEDFUNDS      : Tasa de fondos federales (%)
        - CPIAUCSL      : Índice de precios al consumidor (CPI)
    """

    print("🪙 Descargando datos desde la Reserva Federal (FRED, vía REST)...")

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("⚠️ No se encontró FRED_API_KEY en el archivo .env")

    # Series oficiales FRED
    series = {
        "DCOILBRENTEU": "Oil_Price_Brent",      # Petróleo Brent
        "FEDFUNDS": "Federal_Funds_Rate",       # Tasa de interés
        "CPIAUCSL": "US_Inflation_Index"        # Índice de precios (CPI)
    }

    frames = []
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    # ------------------------------------------------------
    # 🔁 Función auxiliar: descarga y anualiza una serie
    # ------------------------------------------------------
    def get_series(code, name):
        params = {
            "series_id": code,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": "1970-01-01"
        }
        r = requests.get(base_url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"❌ Error HTTP {r.status_code} para {code}")
            return None
        data = r.json().get("observations", [])
        if not data:
            print(f"⚠️ {code} sin datos válidos.")
            return None
        df = pd.DataFrame(data)[["date", "value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date", "value"], inplace=True)
        df["Year"] = df["date"].dt.year
        df = df.groupby("Year")["value"].mean().reset_index()
        df.rename(columns={"value": name}, inplace=True)
        return df

    # ------------------------------------------------------
    # 🚀 Descargar todas las series
    # ------------------------------------------------------
    for code, name in series.items():
        print(f"   🔹 Descargando {name} ({code})...")
        df = get_series(code, name)
        if df is not None and not df.empty:
            frames.append(df)
            print(f"      ✅ {name}: {len(df)} registros")
        else:
            print(f"      ⚠️ {name} no disponible.")

    if not frames:
        raise ValueError("❌ No se pudo descargar ninguna serie válida desde FRED.")

    # ------------------------------------------------------
    # 📈 Combinar y guardar
    # ------------------------------------------------------
    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="Year", how="outer")

    df["Country"] = "Global"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"✅ Datos FRED guardados en {save_path}")
    print(f"📊 Años disponibles: {df['Year'].min()}–{df['Year'].max()}")
    print(f"📁 Columnas: {', '.join(df.columns)}")
    return df


# ----------------------------------------------------------
# 🧪 Ejecución directa
# ----------------------------------------------------------
if __name__ == "__main__":
    fetch_fred_data()
