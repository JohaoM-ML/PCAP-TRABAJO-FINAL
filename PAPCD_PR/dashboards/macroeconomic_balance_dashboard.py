"""
Dashboard de Equilibrio Macroeconómico y Demanda Agregada
Análisis de composición y estabilidad del gasto nacional
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
    page_title="Equilibrio Macroeconómico y Demanda Agregada",
    page_icon="⚖️",
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
        
        # Calcular variables macroeconómicas
        df['GDP_usd'] = pd.to_numeric(df['GDP_usd'], errors='coerce')
        df['Population_wb'] = pd.to_numeric(df['Population_wb'], errors='coerce')
        
        # Componentes de demanda agregada (% del PIB)
        df['Consumption_pct_GDP'] = (df['Household_consumption'] / df['GDP_usd']) * 100
        df['Investment_pct_GDP'] = (df['Gross_capital_formation_main'] / df['GDP_usd']) * 100
        df['Government_pct_GDP'] = (df['Government_consumption'] / df['GDP_usd']) * 100
        df['Exports_pct_GDP'] = (df['Exports_main'] / df['GDP_usd']) * 100
        df['Imports_pct_GDP'] = (df['Imports_main'] / df['GDP_usd']) * 100
        
        # Balance comercial
        df['Trade_balance'] = df['Exports_main'] - df['Imports_main']
        df['Trade_balance_pct_GDP'] = (df['Trade_balance'] / df['GDP_usd']) * 100
        
        # Formación bruta de capital fijo (% del PIB)
        df['Fixed_capital_formation_pct_GDP'] = (df['Gross_fixed_capital_formation'] / df['GDP_usd']) * 100
        
        # PIB per cápita
        df['GDP_per_capita'] = df['GDP_usd'] / df['Population_wb']
        
        # Filtrar datos válidos
        df = df.dropna(subset=['GDP_usd', 'Population_wb'])
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()

def identify_cycles(df):
    """Identificar ciclos de expansión y recesión basado en crecimiento del PIB"""
    df_sorted = df.sort_values(['Country', 'Year'])
    df_sorted['GDP_growth'] = df_sorted.groupby('Country')['GDP_usd'].pct_change() * 100
    
    # Clasificar ciclos
    df_sorted['Cycle'] = 'Neutral'
    df_sorted.loc[df_sorted['GDP_growth'] > 3, 'Cycle'] = 'Expansion'
    df_sorted.loc[df_sorted['GDP_growth'] < -1, 'Cycle'] = 'Recession'
    
    return df_sorted

def main():
    st.title("⚖️ Dashboard de Equilibrio Macroeconómico y Demanda Agregada")
    st.markdown("**Análisis de composición y estabilidad del gasto nacional**")
    
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
    
    # Identificar ciclos
    filtered_df = identify_cycles(filtered_df)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Países analizados", len(filtered_df['Country'].unique()))
    
    with col2:
        avg_consumption = filtered_df['Consumption_pct_GDP'].mean()
        st.metric("Consumo promedio (% PIB)", f"{avg_consumption:.1f}%")
    
    with col3:
        avg_investment = filtered_df['Investment_pct_GDP'].mean()
        st.metric("Inversión promedio (% PIB)", f"{avg_investment:.1f}%")
    
    with col4:
        avg_trade_balance = filtered_df['Trade_balance_pct_GDP'].mean()
        st.metric("Balance comercial promedio", f"{avg_trade_balance:.1f}%")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Composición Demanda", 
        "⚖️ Balance Comercial", 
        "🏗️ Formación Capital", 
        "📈 Ciclos Económicos",
        "🔍 Análisis Comparativo"
    ])
    
    with tab1:
        st.header("Composición de la Demanda Agregada")
        
        # Gráfico de componentes de demanda
        demand_components = ['Consumption_pct_GDP', 'Investment_pct_GDP', 
                           'Government_pct_GDP', 'Exports_pct_GDP', 'Imports_pct_GDP']
        component_names = ['Consumo', 'Inversión', 'Gobierno', 'Exportaciones', 'Importaciones']
        
        # Preparar datos
        plot_data = []
        for _, row in filtered_df.iterrows():
            for comp, name in zip(demand_components, component_names):
                if not pd.isna(row[comp]):
                    plot_data.append({
                        'País': row['Country'],
                        'Año': row['Year'],
                        'Componente': name,
                        'Porcentaje': row[comp]
                    })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Gráfico de barras apiladas
            fig = px.bar(
                plot_df,
                x='País',
                y='Porcentaje',
                color='Componente',
                title="Composición de la Demanda Agregada (% del PIB)",
                labels={'Porcentaje': 'Porcentaje del PIB (%)', 'País': 'País'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de composición promedio
        st.subheader("Composición Promedio por País")
        composition_table = filtered_df.groupby('Country')[demand_components].mean().round(2)
        composition_table.columns = component_names
        st.dataframe(composition_table, use_container_width=True)
    
    with tab2:
        st.header("Balance Comercial y Comercio Exterior")
        
        # Gráfico de balance comercial
        trade_data = filtered_df[['Country', 'Year', 'Trade_balance_pct_GDP', 'Exports_pct_GDP', 'Imports_pct_GDP']].dropna()
        
        if not trade_data.empty:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Balance Comercial (% del PIB)', 'Exportaciones vs Importaciones (% del PIB)'),
                vertical_spacing=0.1
            )
            
            # Balance comercial
            for country in trade_data['Country'].unique():
                country_data = trade_data[trade_data['Country'] == country]
                fig.add_trace(
                    go.Scatter(x=country_data['Year'], y=country_data['Trade_balance_pct_GDP'],
                              mode='lines+markers', name=f'{country} - Balance', showlegend=True),
                    row=1, col=1
                )
            
            # Exportaciones vs Importaciones
            for country in trade_data['Country'].unique():
                country_data = trade_data[trade_data['Country'] == country]
                fig.add_trace(
                    go.Scatter(x=country_data['Year'], y=country_data['Exports_pct_GDP'],
                              mode='lines+markers', name=f'{country} - Exportaciones', showlegend=False),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=country_data['Year'], y=country_data['Imports_pct_GDP'],
                              mode='lines+markers', name=f'{country} - Importaciones', showlegend=False),
                    row=2, col=1
                )
            
            fig.update_layout(height=800, title="Análisis del Comercio Exterior")
            fig.update_xaxes(title_text="Año", row=2, col=1)
            fig.update_yaxes(title_text="Balance Comercial (% PIB)", row=1, col=1)
            fig.update_yaxes(title_text="% del PIB", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de comercio
        col1, col2, col3 = st.columns(3)
        with col1:
            surplus_countries = len(filtered_df[filtered_df['Trade_balance_pct_GDP'] > 0]['Country'].unique())
            st.metric("Países con superávit", surplus_countries)
        
        with col2:
            deficit_countries = len(filtered_df[filtered_df['Trade_balance_pct_GDP'] < 0]['Country'].unique())
            st.metric("Países con déficit", deficit_countries)
        
        with col3:
            avg_balance = filtered_df['Trade_balance_pct_GDP'].mean()
            st.metric("Balance promedio", f"{avg_balance:.2f}%")
    
    with tab3:
        st.header("Formación Bruta de Capital Fijo")
        
        # Gráfico de formación de capital
        capital_data = filtered_df[['Country', 'Year', 'Fixed_capital_formation_pct_GDP', 'Investment_pct_GDP']].dropna()
        
        if not capital_data.empty:
            fig = px.line(
                capital_data,
                x='Year',
                y='Fixed_capital_formation_pct_GDP',
                color='Country',
                title="Formación Bruta de Capital Fijo (% del PIB)",
                labels={'Fixed_capital_formation_pct_GDP': 'Formación Capital (% PIB)', 'Year': 'Año'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparación con inversión total
        st.subheader("Formación de Capital vs Inversión Total")
        comparison_data = filtered_df.groupby('Country').agg({
            'Fixed_capital_formation_pct_GDP': 'mean',
            'Investment_pct_GDP': 'mean'
        }).dropna()
        
        if not comparison_data.empty:
            fig = px.scatter(
                comparison_data,
                x='Investment_pct_GDP',
                y='Fixed_capital_formation_pct_GDP',
                hover_name=comparison_data.index,
                title="Formación de Capital vs Inversión Total",
                labels={'Investment_pct_GDP': 'Inversión Total (% PIB)', 
                       'Fixed_capital_formation_pct_GDP': 'Formación Capital (% PIB)'}
            )
            
            # Línea de 45 grados
            max_val = max(comparison_data['Investment_pct_GDP'].max(), 
                         comparison_data['Fixed_capital_formation_pct_GDP'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines', line=dict(dash='dash', color='red'),
                name='Línea de igualdad'
            ))
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Ciclos de Expansión y Recesión")
        
        # Distribución de ciclos
        cycle_dist = filtered_df['Cycle'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras de ciclos
            fig = px.bar(
                x=cycle_dist.index,
                y=cycle_dist.values,
                title="Distribución de Ciclos Económicos",
                labels={'x': 'Tipo de Ciclo', 'y': 'Número de Observaciones'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gráfico de crecimiento del PIB
            growth_data = filtered_df[['Country', 'Year', 'GDP_growth', 'Cycle']].dropna()
            
            if not growth_data.empty:
                fig = px.scatter(
                    growth_data,
                    x='Year',
                    y='GDP_growth',
                    color='Cycle',
                    title="Crecimiento del PIB por Ciclo Económico",
                    labels={'GDP_growth': 'Crecimiento PIB (%)', 'Year': 'Año'},
                    color_discrete_map={'Expansion': 'green', 'Recession': 'red', 'Neutral': 'blue'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de ciclos por país
        st.subheader("Resumen de Ciclos por País")
        cycle_summary = filtered_df.groupby('Country')['Cycle'].value_counts().unstack(fill_value=0)
        if not cycle_summary.empty:
            st.dataframe(cycle_summary, use_container_width=True)
    
    with tab5:
        st.header("Análisis Comparativo de Estabilidad")
        
        # Matriz de correlación de componentes
        correlation_data = filtered_df[demand_components].corr()
        
        fig = px.imshow(
            correlation_data,
            title="Correlación entre Componentes de Demanda",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de estabilidad por país
        st.subheader("Índice de Estabilidad Macroeconómica")
        
        # Calcular índice de estabilidad (menor variabilidad = más estable)
        stability_data = filtered_df.groupby('Country').agg({
            'Consumption_pct_GDP': 'std',
            'Investment_pct_GDP': 'std',
            'Trade_balance_pct_GDP': 'std',
            'GDP_growth': 'std'
        }).dropna()
        
        # Normalizar y crear índice compuesto
        for col in stability_data.columns:
            stability_data[f'{col}_norm'] = (stability_data[col] - stability_data[col].min()) / (stability_data[col].max() - stability_data[col].min())
        
        stability_data['Stability_Index'] = 1 - (stability_data[['Consumption_pct_GDP_norm', 'Investment_pct_GDP_norm', 
                                                               'Trade_balance_pct_GDP_norm', 'GDP_growth_norm']].mean(axis=1))
        
        stability_ranking = stability_data.sort_values('Stability_Index', ascending=False)
        
        st.dataframe(stability_ranking[['Stability_Index']].round(3), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard de Equilibrio Macroeconómico** | Datos: World Bank, FRED")

if __name__ == "__main__":
    main()
