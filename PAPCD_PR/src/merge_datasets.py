# ================================================================
# 🔗 MERGE PRINCIPAL — Data.csv + WorldBank.csv
# ================================================================
import pandas as pd
import os

# Paths
DATA_PATH = "data/external/global_merged_all.csv"
WB_PATH = "data/external/worldbank_data.csv"
OUTPUT_PATH = "data/external/data_merged_worldbank.csv"

# ---------------------------
# 1️⃣ Cargar datasets
# ---------------------------
print("📂 Cargando datasets...")
data = pd.read_csv(DATA_PATH)
wb = pd.read_csv(WB_PATH)

# Limpieza básica
data.columns = data.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
wb.columns = wb.columns.str.strip().str.replace(" ", "_")

# ---------------------------
# 2️⃣ Renombrar columnas clave
# ---------------------------
data.rename(columns={
    "Per_capita_GNI": "GNI_per_capita_data",
    "Gross_National_IncomeGNI_in_USD": "GNI_total_USD",
    "Gross_Domestic_Product_GDP": "GDP_data",
    "AMA_exchange_rate": "AMA_exchange_rate",
    "IMF_based_exchange_rate": "IMF_exchange_rate"
}, inplace=True)

# ---------------------------
# 3️⃣ Seleccionar columnas relevantes del World Bank
# ---------------------------
wb_subset = wb[
    ["Country", "Year", "GDP_usd", "GDP_growth", "GNI_per_capita",
     "Population", "Exports", "Imports", "Inflation",
     "Interest_rate_real", "Gross_capital_formation_percent_GDP"]
]

# ---------------------------
# 4️⃣ Unir datasets
# ---------------------------
df_merged = data.merge(
    wb_subset,
    on=["Country", "Year"],
    how="left",
    suffixes=("", "_WB")
)

print(f"✅ Merge completado: {len(df_merged):,} filas, {len(df_merged.columns)} columnas")

# ---------------------------
# 5️⃣ Guardar resultado
# ---------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"📊 Dataset combinado guardado en {OUTPUT_PATH}")
