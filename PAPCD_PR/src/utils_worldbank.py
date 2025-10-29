# src/utils_worldbank.py
import wbdata
import pandas as pd
import os


def get_worldbank_country_list(save_path="data/reference/worldbank_countries.csv"):
    """
    Descarga la lista oficial de países válidos de la API del Banco Mundial.
    Retorna un DataFrame con columnas: ISO, Name, Region, IncomeLevel.
    Excluye agregados y grupos regionales (solo países reales).
    Compatible con wbdata >= 0.3.0 (usa get_countries()).
    """
    print("🌍 Descargando lista oficial de países del Banco Mundial...")

    # ✅ usar get_countries() en lugar de get_country()
    countries = wbdata.get_countries()

    data = []
    for c in countries:
        region_id = c["region"]["id"]
        # Excluir regiones agregadas o sin datos reales
        if region_id not in ["NA", "", "WLD"]:
            data.append({
                "ISO": c["id"],
                "Name": c["name"],
                "Region": c["region"]["value"],
                "IncomeLevel": c["incomeLevel"]["value"]
            })

    df = pd.DataFrame(data).drop_duplicates(subset=["ISO"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"✅ Lista oficial: {len(df)} países guardados en {save_path}")
    return df
