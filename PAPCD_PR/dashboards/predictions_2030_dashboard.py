"""
Dashboard de Predicciones de Crecimiento Económico hasta 2030
Visualización interactiva de predicciones (Base/Optimista/Pesimista) para todos los países.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import unicodedata
import pycountry
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------------------------------------------------------
# Configuración de página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicciones Económicas 2030",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Utilidades: normalización y mapeo ISO-3 robusto (corrige lectura de países)
# -----------------------------------------------------------------------------
ALIASES_ISO3 = {
    # Correcciones comunes/alias
    "bolivia (plurinational state of)": "BOL",
    "bolivia": "BOL",
    "brunei darussalam": "BRN",
    "cape verde": "CPV",
    "congo, democratic republic of the": "COD",
    "congo, republic of the": "COG",
    "cote d'ivoire": "CIV",
    "côte d’ivoire": "CIV",
    "cote d’ivoire": "CIV",
    "czechia": "CZE",
    "czech republic": "CZE",
    "eswatini": "SWZ",
    "holy see": "VAT",
    "hong kong": "HKG",
    "iran, islamic republic of": "IRN",
    "iran": "IRN",
    "korea, republic of": "KOR",
    "south korea": "KOR",
    "korea, democratic people's republic of": "PRK",
    "north korea": "PRK",
    "lao people's democratic republic": "LAO",
    "laos": "LAO",
    "macedonia, the former yugoslav republic of": "MKD",
    "north macedonia": "MKD",
    "moldova, republic of": "MDA",
    "palestine, state of": "PSE",
    "russian federation": "RUS",
    "russia": "RUS",
    "syrian arab republic": "SYR",
    "taiwan": "TWN",
    "tanzania, united republic of": "TZA",
    "united kingdom": "GBR",
    "united states of america": "USA",
    "united states": "USA",
    "usa": "USA",
    "venezuela (bolivarian republic of)": "VEN",
    "vietnam": "VNM",
    "myanmar (burma)": "MMR",
    "burma": "MMR",
    # Acentos / español
    "peru": "PER",
    "méxico": "MEX",
    "mexico": "MEX",
    "españa": "ESP",
    "cabo verde": "CPV",
    "cote d ivoire": "CIV",
    "côte d ivoire": "CIV",
}


def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    # Normaliza y elimina acentos/tildes
    nfkd = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def normalize_country_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = strip_accents(name).lower().strip()
    # Limpiezas frecuentes
    s = s.replace("  ", " ")
    s = s.replace("'", "’")  # homogeniza apóstrofos
    s = s.replace("’", "'")
    s = s.replace("democratic republic of", "democratic republic of")
    s = s.replace("&", "and")
    return s


def country_to_iso3(name: str) -> str | None:
    """
    Mapea nombre de país a ISO-3 de forma robusta:
    - Normaliza tildes/espacios.
    - Usa alias conocidos.
    - Fallback a pycountry.lookup (maneja variantes).
    """
    if not isinstance(name, str) or not name.strip():
        return None
    norm = normalize_country_name(name)

    # Alias manual primero
    if norm in ALIASES_ISO3:
        return ALIASES_ISO3[norm]

    # Intento 1: lookup directo
    try:
        return pycountry.countries.lookup(name).alpha_3
    except Exception:
        pass

    # Intento 2: lookup con nombre normalizado (capitalizado)
    try:
        cap = " ".join(w.capitalize() for w in strip_accents(name).split())
        return pycountry.countries.lookup(cap).alpha_3
    except Exception:
        pass

    # Intento 3: lookup con alias normalizado capitalizado
    try:
        cap2 = " ".join(w.capitalize() for w in norm.split())
        return pycountry.countries.lookup(cap2).alpha_3
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Carga de datos (cache + rutas robustas)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Carga predicciones y, opcionalmente, histórico.
    Acepta varios nombres de columnas (flexible).
    """
    BASE = Path(__file__).resolve().parent
    pred_path = BASE.parent / "results" / "predictions_2030.csv"
    hist_path = BASE.parent / "data" / "external" / "global_merged_all.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"No se encontró {pred_path}")

    predictions_df = pd.read_csv(pred_path)

    # Detecta/renombra columnas esenciales
    # Country
    country_col = None
    for c in ["Country", "country", "pais", "País", "PAIS"]:
        if c in predictions_df.columns:
            country_col = c
            break
    if country_col is None:
        raise ValueError("No se encuentra columna de país en predictions_2030.csv")

    # Year
    year_col = None
    for c in ["Year", "year", "anio", "Año"]:
        if c in predictions_df.columns:
            year_col = c
            break
    if year_col is None:
        raise ValueError("No se encuentra columna de año en predictions_2030.csv")

    # Base / Low / High (muchos alias)
    def find_first(cols):
        for c in cols:
            if c in predictions_df.columns:
                return c
        return None

    base_aliases = ["GDP_growth_predicted", "pred_base", "base", "prediction", "yhat", "growth_pred"]
    low_aliases = ["GDP_growth_predicted_low", "pred_low", "low", "pessimistic", "pesimista"]
    high_aliases = ["GDP_growth_predicted_high", "pred_high", "high", "optimistic", "optimista"]

    base_col = find_first(base_aliases)
    low_col = find_first(low_aliases)
    high_col = find_first(high_aliases)

    df = predictions_df.rename(columns={
        country_col: "Country",
        year_col: "Year"
    }).copy()

    # Crea/renombra escenarios
    if base_col is None:
        raise ValueError("No se encuentra la columna base de predicción (ej. 'GDP_growth_predicted').")
    df = df.rename(columns={base_col: "pred_base"})

    if low_col is not None:
        df = df.rename(columns={low_col: "pred_low"})
    if high_col is not None:
        df = df.rename(columns={high_col: "pred_high"})

    # Si faltan low/high, generar banda sintética ±σ por país
    if "pred_low" not in df.columns or "pred_high" not in df.columns:
        sigma_country = df.groupby("Country")["pred_base"].transform("std")
        sigma_global = df["pred_base"].std()
        sigma = sigma_country.fillna(sigma_global if pd.notna(sigma_global) else 1.0)
        df["pred_low"] = df["pred_base"] - sigma
        df["pred_high"] = df["pred_base"] + sigma

    # Clip básico para valores extremos
    df["pred_low"] = df["pred_low"].clip(lower=-20, upper=30)
    df["pred_high"] = df["pred_high"].clip(lower=-20, upper=30)

    # ISO-3 (si no viene ya)
    iso_col = None
    for c in ["iso3", "ISO3", "country_code", "Country_Code"]:
        if c in df.columns:
            iso_col = c
            break

    if iso_col is None:
        df["iso3"] = df["Country"].apply(country_to_iso3)
    else:
        df = df.rename(columns={iso_col: "iso3"})

    # Señala los países que no se pudieron mapear
    missing_iso = df["iso3"].isna().sum()
    if missing_iso > 0:
        st.warning(f"⚠️ {missing_iso} filas no se pudieron mapear a ISO-3 (revisa nombres de país en predictions_2030.csv).")

    # Histórico (opcional)
    historical_df = pd.DataFrame()
    if hist_path.exists():
        hist_raw = pd.read_csv(hist_path)
        if {"Country", "Year", "GDP_growth"}.issubset(hist_raw.columns):
            historical_df = hist_raw[["Country", "Year", "GDP_growth"]].dropna()

    return df, historical_df


# -----------------------------------------------------------------------------
# Helpers de UI
# -----------------------------------------------------------------------------
def scenario_col_key(name: str) -> str:
    n = (name or "").lower()
    if n.startswith("opt") or "optimista" in n:
        return "pred_high"
    if n.startswith("pes") or "pesi" in n:
        return "pred_low"
    return "pred_base"


def create_time_series_plot(df, selected_countries, title, show_band=True):
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    colors = px.colors.qualitative.Set3

    for i, country in enumerate(selected_countries):
        g = df[df["Country"] == country].sort_values("Year")
        if g.empty:
            continue

        # Cenefa: low-high
        if show_band and {"pred_low", "pred_high"}.issubset(g.columns):
            fig.add_trace(go.Scatter(
                x=g["Year"], y=g["pred_high"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip", name=f"{country} (High)"
            ))
            fig.add_trace(go.Scatter(
                x=g["Year"], y=g["pred_low"], mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(100,100,100,0.15)",
                showlegend=False, hoverinfo="skip", name=f"{country} (Low)"
            ))

        # Línea base
        fig.add_trace(go.Scatter(
            x=g["Year"], y=g["pred_base"], mode="lines+markers",
            name=f"{country}", line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Año",
        yaxis_title="Crecimiento del PIB (%)",
        hovermode="x unified",
        height=500
    )
    return fig


def create_geographical_plot(df, year, col_name):
    if df.empty:
        return go.Figure()
    year_data = df[df["Year"] == year].copy()
    if year_data.empty:
        return go.Figure()

    # Filtra filas con iso3 válido
    year_data = year_data.dropna(subset=["iso3"])
    if year_data.empty:
        return go.Figure()

    fig = px.choropleth(
        year_data,
        locations="iso3",
        color=col_name,
        hover_name="Country",
        hover_data=[col_name],
        color_continuous_scale="RdYlGn",
        title=f"Mapa de Crecimiento Económico Predicho - {year} ({col_name})"
    )
    fig.update_layout(height=500)
    return fig


def create_heatmap_plot(df, year, col_name):
    if df.empty:
        return go.Figure()
    year_data = df[df["Year"] == year].copy()
    if year_data.empty:
        return go.Figure()
    year_data = year_data.sort_values(col_name, ascending=False)

    fig = go.Figure(data=go.Heatmap(
        z=[year_data[col_name].values],
        x=year_data["Country"].values,
        y=[f"{year}"],
        colorscale="RdYlGn",
        showscale=True,
        hoverongaps=False
    ))
    fig.update_layout(
        title=f"Predicciones de Crecimiento Económico - {year} ({col_name})",
        xaxis_title="Países",
        yaxis_title="Año",
        height=400
    )
    return fig


def create_ranking_plot(df, year, col_name, top_n=20):
    if df.empty:
        return go.Figure()
    year_data = df[df["Year"] == year].copy()
    if year_data.empty:
        return go.Figure()

    top_countries = year_data.nlargest(top_n, col_name)
    bottom_countries = year_data.nsmallest(top_n, col_name)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Top {top_n} Países", f"Bottom {top_n} Países"),
        horizontal_spacing=0.1
    )

    fig.add_trace(
        go.Bar(
            y=top_countries["Country"], x=top_countries[col_name],
            orientation="h", name="Top", marker_color="green", showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            y=bottom_countries["Country"], x=bottom_countries[col_name],
            orientation="h", name="Bottom", marker_color="red", showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f"Ranking de Crecimiento Económico - {year} ({col_name})",
        height=600,
        showlegend=False
    )
    return fig


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main():
    st.title("📈 Predicciones de Crecimiento Económico hasta 2030")
    st.markdown("**Modelo**: Ridge (u otros) | *Los indicadores se basan en predicciones y están sujetas a incertidumbre.*")
    st.markdown("---")

    # Cargar datos
    try:
        predictions_df, historical_df = load_data()
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

    if predictions_df.empty:
        st.error("No se pudieron cargar las predicciones.")
        st.stop()

    # Sidebar
    st.sidebar.header("🔍 Filtros y Controles")

    # Escenario principal para mapa/rankings
    scenario_for_aggregates = st.sidebar.selectbox(
        "Escenario para mapa/rankings/tablas",
        options=["Base", "Optimista", "Pesimista"],
        index=0
    )
    show_band = st.sidebar.checkbox("Mostrar banda (opt/base/pes) en series", value=True)

    countries = sorted(predictions_df["Country"].unique().tolist())
    default_c = countries[:10] if len(countries) > 10 else countries
    selected_countries = st.sidebar.multiselect("Seleccionar países", countries, default=default_c)

    years = sorted(predictions_df["Year"].unique().tolist())
    selected_year = st.sidebar.selectbox("Seleccionar año", years, index=len(years) - 1)

    pred_col = scenario_col_key(scenario_for_aggregates)

    # Estadísticas generales del escenario seleccionado
    st.sidebar.markdown("### 📊 Estadísticas Generales")
    subset_for_stats = predictions_df[predictions_df["Year"] == selected_year] if "Year" in predictions_df else predictions_df
    st.sidebar.metric("Crecimiento Promedio", f"{subset_for_stats[pred_col].mean():.2f}%")
    st.sidebar.metric("Crecimiento Mediano", f"{subset_for_stats[pred_col].median():.2f}%")
    st.sidebar.metric("Crecimiento Máximo", f"{subset_for_stats[pred_col].max():.2f}%")
    st.sidebar.metric("Crecimiento Mínimo", f"{subset_for_stats[pred_col].min():.2f}%")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Series Temporales",
        "🗺️ Mapa Geográfico",
        "🏆 Rankings",
        "📊 Análisis Comparativo",
        "📋 Datos Detallados"
    ])

    # Tab 1: Series
    with tab1:
        st.header("📈 Evolución Temporal de Predicciones")
        if selected_countries:
            fig = create_time_series_plot(
                predictions_df, selected_countries,
                f"Predicciones de Crecimiento Económico ({', '.join(selected_countries[:5])})",
                show_band=show_band
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabla de stats por país (base + 2030 si existe)
            st.subheader("📊 Estadísticas por País (escenario base)")
            stats_rows = []
            for c in selected_countries:
                g = predictions_df[predictions_df["Country"] == c]
                if g.empty:
                    continue
                val_2030 = g.loc[g["Year"] == 2030, "pred_base"]
                stats_rows.append({
                    "País": c,
                    "Crecimiento Promedio (base)": f"{g['pred_base'].mean():.2f}%",
                    "Crecimiento 2030 (base)": f"{val_2030.iloc[0]:.2f}%" if not val_2030.empty else "N/A",
                    "Volatilidad (base)": f"{g['pred_base'].std():.2f}%"
                })
            if stats_rows:
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
        else:
            st.warning("Selecciona al menos un país para ver las series.")

    # Tab 2: Mapa
    with tab2:
        st.header("🗺️ Mapa de Crecimiento Económico")
        fig_map = create_geographical_plot(predictions_df, selected_year, pred_col)
        if fig_map.data:
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Mapa no disponible para el año seleccionado.")

        st.subheader("🔥 Heatmap de Países")
        fig_heatmap = create_heatmap_plot(predictions_df, selected_year, pred_col)
        if fig_heatmap.data:
            st.plotly_chart(fig_heatmap, use_container_width=True)

    # Tab 3: Rankings
    with tab3:
        st.header("🏆 Rankings de Crecimiento Económico")
        fig_rank = create_ranking_plot(predictions_df, selected_year, pred_col, top_n=15)
        st.plotly_chart(fig_rank, use_container_width=True)

        # Tabla de ranking
        st.subheader("📋 Ranking Completo")
        year_data = predictions_df[predictions_df["Year"] == selected_year].copy()
        if not year_data.empty:
            year_data = year_data.sort_values(pred_col, ascending=False)
            year_data["Ranking"] = range(1, len(year_data) + 1)
            year_data = year_data[["Ranking", "Country", pred_col]].rename(
                columns={pred_col: "Crecimiento Predicho (%)"}
            )
            st.dataframe(year_data, use_container_width=True)

    # Tab 4: Comparativo
    with tab4:
        st.header("📊 Análisis Comparativo")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Distribución de Crecimiento")
            fig = px.histogram(
                predictions_df[predictions_df["Year"] == selected_year],
                x=pred_col, nbins=30,
                title=f"Distribución de Predicciones - {selected_year} ({pred_col})"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Box Plot por Año")
            fig = px.box(
                predictions_df,
                x="Year", y=pred_col,
                title=f"Distribución de Predicciones por Año ({pred_col})"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 Tendencias (resumen por año)")
        trend = predictions_df.groupby("Year")[pred_col].agg(["mean", "median", "std", "min", "max"]).round(2)
        st.dataframe(trend, use_container_width=True)

    # Tab 5: Datos
    with tab5:
        st.header("📋 Datos Detallados")
        c1, c2 = st.columns(2)
        data_year = predictions_df.copy()

        with c1:
            min_g = float(data_year[pred_col].min())
            max_g = float(data_year[pred_col].max())
            growth_min = st.slider("Crecimiento mínimo (%)", min_g, max_g, min_g)
        with c2:
            growth_max = st.slider("Crecimiento máximo (%)", min_g, max_g, max_g)

        filtered = data_year[(data_year[pred_col] >= growth_min) & (data_year[pred_col] <= growth_max)]
        if selected_countries:
            filtered = filtered[filtered["Country"].isin(selected_countries)]

        st.subheader("📊 Datos Filtrados")
        show_cols = ["Country", "Year", pred_col, "pred_low", "pred_high", "iso3"]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered.sort_values(["Year", pred_col], ascending=[True, False])[show_cols], use_container_width=True)

        csv = filtered[show_cols].to_csv(index=False)
        st.download_button(
            label="📥 Descargar Datos CSV",
            data=csv,
            file_name=f"predicciones_crecimiento_{selected_year}.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown(f"**Escenario actual (mapa/rankings):** `{scenario_for_aggregates}`")
    st.markdown("**Fuente de datos**: predictions_2030.csv (y opcional histórico WB/FRED).")


if __name__ == "__main__":
    main()
