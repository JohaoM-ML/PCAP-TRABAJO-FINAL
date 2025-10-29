"""
Dashboard de Conectividad Global y Factores Externos
Visualización de la influencia del contexto internacional sobre economías locales
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Conectividad Global y Factores Externos",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Cargar y preprocesar los datos"""
    try:
        df = pd.read_csv('data/external/global_merged_all.csv')
        
        # Limpiar nombres de países
        df['Country'] = df['Country'].str.strip()
        
        # Convertir variables a numérico
        numeric_cols = ['GDP_usd', 'Population_wb', 'Oil_Price_Brent', 'Federal_Funds_Rate', 
                       'US_Inflation_Index', 'Gross_capital_formation_main', 'Exports_wb', 'Imports_wb']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calcular variables derivadas
        df['GDP_per_capita'] = df['GDP_usd'] / df['Population_wb']
        df['GDP_growth'] = df.groupby('Country')['GDP_usd'].pct_change() * 100
        
        # Inversión doméstica (% del PIB)
        df['Domestic_investment_pct_GDP'] = (df['Gross_capital_formation_main'] / df['GDP_usd']) * 100
        
        # Inflación nacional (simulada basada en cambios de precios)
        df['National_inflation'] = df.groupby('Country')['GDP_usd'].pct_change() * 100
        
        # Filtrar datos válidos
        df = df.dropna(subset=['GDP_usd', 'Population_wb'])
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()

def calculate_shock_transmission(df):
    """Calcular transmisión de shocks globales"""
    # Correlación entre variables globales y locales
    global_vars = ['Oil_Price_Brent', 'Federal_Funds_Rate', 'US_Inflation_Index']
    local_vars = ['GDP_growth', 'Domestic_investment_pct_GDP', 'National_inflation']
    
    correlations = {}
    for global_var in global_vars:
        if global_var in df.columns:
            correlations[global_var] = {}
            for local_var in local_vars:
                if local_var in df.columns:
                    corr = df[global_var].corr(df[local_var])
                    correlations[global_var][local_var] = corr
    
    return correlations

def identify_shock_periods(df):
    """Identificar períodos de shocks globales"""
    df_sorted = df.sort_values('Year')
    
    # Shock de petróleo (precio > percentil 90)
    oil_threshold = df['Oil_Price_Brent'].quantile(0.9)
    df['Oil_shock'] = df['Oil_Price_Brent'] > oil_threshold
    
    # Shock de tasas (FED rate > percentil 90)
    fed_threshold = df['Federal_Funds_Rate'].quantile(0.9)
    df['Fed_shock'] = df['Federal_Funds_Rate'] > fed_threshold
    
    # Shock de inflación (US inflation > percentil 90)
    inflation_threshold = df['US_Inflation_Index'].quantile(0.9)
    df['Inflation_shock'] = df['US_Inflation_Index'] > inflation_threshold
    
    return df

def main():
    st.title("🌐 Dashboard de Conectividad Global y Factores Externos")
    st.markdown("**Visualización de la influencia del contexto internacional sobre economías locales**")
    
    # Cargar datos
    df = load_data()
    
    if df.empty:
        st.error("No se pudieron cargar los datos")
        return
    
    # Sidebar con filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtro de países
    countries = sorted(df['Country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Seleccionar países",
        countries,
        default=countries[:10] if len(countries) > 10 else countries
    )
    
    # Filtro de años
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.sidebar.slider(
        "Rango de años",
        min_value=min_year,
        max_value=max_year,
        value=(2000, max_year)
    )
    
    # Aplicar filtros
    filtered_df = df[
        (df['Country'].isin(selected_countries)) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ].copy()
    
    if filtered_df.empty:
        st.warning("No hay datos para los filtros seleccionados")
        return
    
    # Identificar shocks
    filtered_df = identify_shock_periods(filtered_df)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Países analizados", len(filtered_df['Country'].unique()))
    
    with col2:
        avg_oil_price = filtered_df['Oil_Price_Brent'].mean()
        st.metric("Precio petróleo promedio", f"${avg_oil_price:.2f}")
    
    with col3:
        avg_fed_rate = filtered_df['Federal_Funds_Rate'].mean()
        st.metric("Tasa FED promedio", f"{avg_fed_rate:.2f}%")
    
    with col4:
        oil_shocks = filtered_df['Oil_shock'].sum()
        st.metric("Períodos shock petróleo", f"{oil_shocks}")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🛢️ Petróleo vs Crecimiento", 
        "🏦 FED vs Inversión", 
        "📈 Inflación Global vs Local", 
        "⚡ Transmisión de Shocks",
        "🔍 Sensibilidad Externa"
    ])
    
    with tab1:
        st.header("Precio del Petróleo vs Crecimiento del PIB")
        
        # Gráfico principal: Petróleo vs Crecimiento
        oil_growth_data = filtered_df[['Country', 'Year', 'Oil_Price_Brent', 'GDP_growth']].dropna()
        
        if not oil_growth_data.empty:
            # Crear variable de tamaño positiva
            oil_growth_data['size_positive'] = abs(oil_growth_data['GDP_growth'])
            
            fig = px.scatter(
                oil_growth_data,
                x='Oil_Price_Brent',
                y='GDP_growth',
                color='Country',
                size='size_positive',
                title="Precio del Petróleo vs Crecimiento del PIB",
                labels={'Oil_Price_Brent': 'Precio Petróleo Brent (USD)', 'GDP_growth': 'Crecimiento PIB (%)'},
                hover_data=['Year']
            )
            
            # Línea de tendencia
            z = np.polyfit(oil_growth_data['Oil_Price_Brent'], oil_growth_data['GDP_growth'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=oil_growth_data['Oil_Price_Brent'],
                y=p(oil_growth_data['Oil_Price_Brent']),
                mode='lines',
                name='Tendencia',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis temporal
        st.subheader("Evolución Temporal: Petróleo y Crecimiento")
        
        # Gráfico de líneas temporales
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Precio del Petróleo Brent', 'Crecimiento del PIB'),
            vertical_spacing=0.1
        )
        
        # Petróleo
        oil_avg = filtered_df.groupby('Year')['Oil_Price_Brent'].mean()
        fig.add_trace(
            go.Scatter(x=oil_avg.index, y=oil_avg.values, mode='lines', name='Precio Petróleo'),
            row=1, col=1
        )
        
        # Crecimiento PIB
        gdp_avg = filtered_df.groupby('Year')['GDP_growth'].mean()
        fig.add_trace(
            go.Scatter(x=gdp_avg.index, y=gdp_avg.values, mode='lines', name='Crecimiento PIB'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title="Evolución Temporal de Variables Globales")
        fig.update_xaxes(title_text="Año", row=2, col=1)
        fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Crecimiento (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlación
        if not oil_growth_data.empty:
            correlation = oil_growth_data['Oil_Price_Brent'].corr(oil_growth_data['GDP_growth'])
            st.metric("Correlación Petróleo-Crecimiento", f"{correlation:.3f}")
    
    with tab2:
        st.header("Tasa de Fondos Federales vs Inversión Doméstica")
        
        # Gráfico principal: FED vs Inversión
        fed_investment_data = filtered_df[['Country', 'Year', 'Federal_Funds_Rate', 'Domestic_investment_pct_GDP']].dropna()
        
        if not fed_investment_data.empty:
            # Crear variable de tamaño positiva
            fed_investment_data['size_positive'] = abs(fed_investment_data['Domestic_investment_pct_GDP'])
            
            fig = px.scatter(
                fed_investment_data,
                x='Federal_Funds_Rate',
                y='Domestic_investment_pct_GDP',
                color='Country',
                size='size_positive',
                title="Tasa FED vs Inversión Doméstica (% del PIB)",
                labels={'Federal_Funds_Rate': 'Tasa FED (%)', 'Domestic_investment_pct_GDP': 'Inversión Doméstica (% PIB)'},
                hover_data=['Year']
            )
            
            # Línea de tendencia
            z = np.polyfit(fed_investment_data['Federal_Funds_Rate'], fed_investment_data['Domestic_investment_pct_GDP'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=fed_investment_data['Federal_Funds_Rate'],
                y=p(fed_investment_data['Federal_Funds_Rate']),
                mode='lines',
                name='Tendencia',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis por período
        st.subheader("Impacto de Cambios en Tasa FED")
        
        # Calcular cambios en tasa FED
        fed_changes = filtered_df.groupby('Year')['Federal_Funds_Rate'].mean().diff()
        investment_changes = filtered_df.groupby('Year')['Domestic_investment_pct_GDP'].mean().diff()
        
        fig = px.scatter(
            x=fed_changes,
            y=investment_changes,
            title="Cambios en Tasa FED vs Cambios en Inversión",
            labels={'x': 'Cambio Tasa FED (%)', 'y': 'Cambio Inversión (% PIB)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlación
        if not fed_investment_data.empty:
            correlation = fed_investment_data['Federal_Funds_Rate'].corr(fed_investment_data['Domestic_investment_pct_GDP'])
            st.metric("Correlación FED-Inversión", f"{correlation:.3f}")
    
    with tab3:
        st.header("Inflación Global vs Inflación Nacional")
        
        # Gráfico principal: Inflación global vs local
        inflation_data = filtered_df[['Country', 'Year', 'US_Inflation_Index', 'National_inflation']].dropna()
        
        if not inflation_data.empty:
            # Crear variable de tamaño positiva
            inflation_data['size_positive'] = abs(inflation_data['National_inflation'])
            
            fig = px.scatter(
                inflation_data,
                x='US_Inflation_Index',
                y='National_inflation',
                color='Country',
                size='size_positive',
                title="Inflación Global (US) vs Inflación Nacional",
                labels={'US_Inflation_Index': 'Inflación US (%)', 'National_inflation': 'Inflación Nacional (%)'},
                hover_data=['Year']
            )
            
            # Línea de 45 grados
            max_inflation = max(inflation_data['US_Inflation_Index'].max(), inflation_data['National_inflation'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_inflation], y=[0, max_inflation],
                mode='lines', line=dict(dash='dash', color='red'),
                name='Línea de igualdad'
            ))
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Evolución temporal de inflación
        st.subheader("Evolución Temporal de Inflación")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Inflación Global (US)', 'Inflación Nacional Promedio'),
            vertical_spacing=0.1
        )
        
        # Inflación US
        us_inflation = filtered_df.groupby('Year')['US_Inflation_Index'].mean()
        fig.add_trace(
            go.Scatter(x=us_inflation.index, y=us_inflation.values, mode='lines', name='Inflación US'),
            row=1, col=1
        )
        
        # Inflación nacional promedio
        national_inflation = filtered_df.groupby('Year')['National_inflation'].mean()
        fig.add_trace(
            go.Scatter(x=national_inflation.index, y=national_inflation.values, mode='lines', name='Inflación Nacional'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title="Evolución de Inflación Global vs Nacional")
        fig.update_xaxes(title_text="Año", row=2, col=1)
        fig.update_yaxes(title_text="Inflación (%)", row=1, col=1)
        fig.update_yaxes(title_text="Inflación (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlación
        if not inflation_data.empty:
            correlation = inflation_data['US_Inflation_Index'].corr(inflation_data['National_inflation'])
            st.metric("Correlación Inflación Global-Nacional", f"{correlation:.3f}")
    
    with tab4:
        st.header("Transmisión de Shocks Globales")
        
        # Calcular correlaciones de transmisión
        correlations = calculate_shock_transmission(filtered_df)
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación: Variables Globales vs Locales")
        
        if correlations:
            # Crear matriz de correlación
            global_vars = list(correlations.keys())
            local_vars = ['GDP_growth', 'Domestic_investment_pct_GDP', 'National_inflation']
            
            corr_matrix = []
            for global_var in global_vars:
                row = []
                for local_var in local_vars:
                    if local_var in correlations[global_var]:
                        row.append(correlations[global_var][local_var])
                    else:
                        row.append(0)
                corr_matrix.append(row)
            
            fig = px.imshow(
                corr_matrix,
                x=local_vars,
                y=global_vars,
                title="Correlación entre Variables Globales y Locales",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de shocks
        st.subheader("Períodos de Shocks Globales")
        
        shock_summary = filtered_df.groupby('Year').agg({
            'Oil_shock': 'sum',
            'Fed_shock': 'sum',
            'Inflation_shock': 'sum'
        })
        
        fig = px.bar(
            shock_summary,
            title="Número de Países Afectados por Shocks por Año",
            labels={'value': 'Número de Países', 'index': 'Año'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Sensibilidad Externa de las Economías")
        
        # Calcular índice de sensibilidad
        sensitivity_data = filtered_df.groupby('Country').agg({
            'GDP_growth': lambda x: x.corr(filtered_df.loc[x.index, 'Oil_Price_Brent']),
            'Domestic_investment_pct_GDP': lambda x: x.corr(filtered_df.loc[x.index, 'Federal_Funds_Rate']),
            'National_inflation': lambda x: x.corr(filtered_df.loc[x.index, 'US_Inflation_Index'])
        }).dropna()
        
        sensitivity_data.columns = ['Sensibilidad_Petróleo', 'Sensibilidad_FED', 'Sensibilidad_Inflación']
        
        # Índice compuesto de sensibilidad
        sensitivity_data['Sensibilidad_Global'] = sensitivity_data.mean(axis=1)
        
        # Ranking de sensibilidad
        st.subheader("Ranking de Sensibilidad Externa")
        sensitivity_ranking = sensitivity_data.sort_values('Sensibilidad_Global', ascending=False)
        st.dataframe(sensitivity_ranking.round(3), use_container_width=True)
        
        # Gráfico de sensibilidad
        # Crear variable de tamaño positiva
        sensitivity_ranking['size_positive'] = abs(sensitivity_ranking['Sensibilidad_Global'])
        
        fig = px.scatter(
            sensitivity_ranking,
            x='Sensibilidad_Petróleo',
            y='Sensibilidad_FED',
            size='size_positive',
            hover_name=sensitivity_ranking.index,
            title="Sensibilidad Externa: Petróleo vs FED",
            labels={'Sensibilidad_Petróleo': 'Sensibilidad al Petróleo', 'Sensibilidad_FED': 'Sensibilidad a FED'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Exportar datos de sensibilidad
        csv = sensitivity_ranking.to_csv()
        st.download_button(
            label="Descargar datos de sensibilidad",
            data=csv,
            file_name=f"sensibilidad_externa_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard de Conectividad Global** | Datos: World Bank, FRED")

if __name__ == "__main__":
    main()
