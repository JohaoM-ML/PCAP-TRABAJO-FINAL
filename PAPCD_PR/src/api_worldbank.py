# ==========================================================
# 🌍 API: Banco Mundial (World Development Indicators)
# Limpieza total de warnings, logs y caché
# ==========================================================

import os
import warnings
import logging
import urllib3

# ----------------------------------------------------------
# 🔇 1. Silenciar todos los warnings de Python
# ----------------------------------------------------------
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# 🔇 2. Desactivar logs globales (incluye wbdata, urllib, cache)
# ----------------------------------------------------------
logging.getLogger().setLevel(logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

# ----------------------------------------------------------
# 🔇 3. Silenciar warnings SSL y HTTP de urllib / requests
# ----------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

# ----------------------------------------------------------
# 🔇 4. Desactivar caché interna de wbdata y sus logs
# ----------------------------------------------------------
import wbdata
wbdata.cache = None
logging.getLogger("shelved_cache.persistent_cache").setLevel(logging.CRITICAL)
logging.getLogger("wbdata.api").setLevel(logging.CRITICAL)
logging.getLogger("wbdata").setLevel(logging.CRITICAL)

# ----------------------------------------------------------
# Librerías principales
# ----------------------------------------------------------
import datetime
import pandas as pd
import time


def fetch_worldbank_data(countries, save_path="data/external/worldbank_data.csv", start_year=1970):
    """
    Descarga datos macroeconómicos del Banco Mundial (WDI).
    Totalmente silencioso: sin warnings, sin logs ni caché.
    """

    indicators = {
        "NY.GDP.MKTP.CD": "GDP_usd",
        "NY.GDP.MKTP.KD.ZG": "GDP_growth",
        "NY.GNP.PCAP.CD": "GNI_per_capita",
        "SP.POP.TOTL": "Population",
        "NE.EXP.GNFS.CD": "Exports",
        "NE.IMP.GNFS.CD": "Imports",
        "FP.CPI.TOTL.ZG": "Inflation",
        "FR.INR.RINR": "Interest_rate_real",
        "NE.GDI.FTOT.ZS": "Gross_capital_formation_percent_GDP"
    }

    start_date = datetime.datetime(start_year, 1, 1)
    end_date = datetime.datetime.now()
    print(f"🌍 Descargando datos del Banco Mundial para {len(countries)} países ({start_date.year}-{end_date.year})...")

    all_data = []
    step = 50  # cantidad de países por bloque

    for i in range(0, len(countries), step):
        subset = countries[i:i + step]
        print(f"🔹 Procesando bloque {i // step + 1} ({len(subset)} países)...")

        try:
            df_temp = wbdata.get_dataframe(indicators, country=subset, date=(start_date, end_date))
            if df_temp is not None and not df_temp.empty:
                df_temp = df_temp.reset_index().rename(columns={"country": "Country", "date": "Year"})
                df_temp["Year"] = pd.to_datetime(df_temp["Year"]).dt.year
                all_data.append(df_temp)
            else:
                print(f"⚠️ Bloque {i // step + 1} sin datos válidos.")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error al procesar bloque {i // step + 1}: {e}")

    if not all_data:
        print("❌ No se descargaron datos de ningún bloque.")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"✅ Datos descargados desde {start_year} hasta {end_date.year}.")
    print(f"✅ Guardados en: {save_path}")
    print(f"📊 Registros totales: {len(df):,} filas.")
    print(f"🌐 Cobertura de países: {df['Country'].nunique()} únicos.")
    return df
