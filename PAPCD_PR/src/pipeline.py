# ============================================================ 
# 🌍 Pipeline Económico Global — World Bank + FRED
# ============================================================
# Descarga automática de datos macroeconómicos anuales:
#  - Banco Mundial (World Development Indicators)
#  - Reserva Federal de EE.UU. (FRED)
# ============================================================

# ------------------------------------------------------------
# 🔇 Limpieza global de logs y warnings
# ------------------------------------------------------------
import os
import sys
import warnings
import logging
import urllib3

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # oculta logs C++/TensorFlow
os.environ["PYTHONWARNINGS"] = "ignore"

# ------------------------------------------------------------
# 🧭 Configuración de rutas para ejecución directa o modular
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("✅ Entorno inicializado. Ejecutando pipeline global...")

# ------------------------------------------------------------
# 🔧 Importación de módulos del pipeline
# ------------------------------------------------------------
from src.utils_worldbank import get_worldbank_country_list
from src.api_worldbank import fetch_worldbank_data
from src.api_fred import fetch_fred_data
# ------------------------------------------------------------

def update_all_data():
    """
    Ejecuta la actualización completa del pipeline:
    1️⃣ Descarga lista oficial de países y datos del Banco Mundial
    2️⃣ Descarga datos globales del FRED (commodities y tasas)
    3️⃣ Devuelve ambos DataFrames
    """
    # ======================================================
    # 1️⃣ Banco Mundial
    # ======================================================
    df_countries = get_worldbank_country_list()
    iso_codes = df_countries["ISO"].tolist()
    print(f"🌍 {len(iso_codes)} países válidos detectados en la API del Banco Mundial.")
    df_wb = fetch_worldbank_data(iso_codes, start_year=1970)

    # ======================================================
    # 2️⃣ Reserva Federal (FRED)
    # ======================================================
    df_fred = fetch_fred_data()


    # ======================================================
    # 4️⃣ Resumen general
    # ======================================================
    print("\n📊 RESUMEN GENERAL:")
    print(f"   🌍 Banco Mundial → {len(df_wb):,} filas, {df_wb['Country'].nunique()} países.")
    print(f"   🪙 FRED → {len(df_fred):,} filas, {df_fred['Year'].nunique()} años de datos globales.")
    print("\n✅ Pipeline ejecutado correctamente.")
    print("📁 Archivos generados en carpeta 'data/external/'")

    return df_wb, df_fred


# ------------------------------------------------------------
# 🚀 Ejecución directa
# ------------------------------------------------------------
if __name__ == "__main__":
    update_all_data()
