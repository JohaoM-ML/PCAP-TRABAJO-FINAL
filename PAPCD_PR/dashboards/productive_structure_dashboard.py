"""
Dashboard de Estructura Productiva y Diversificación Económica
Análisis de composición sectorial y concentración económica
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Estructura Productiva y Diversificación",
    page_icon="🏭",
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
        
        # Calcular PIB per cápita
        df['GDP_per_capita'] = df['GDP_usd'] / df['Population_wb']
        
        # Filtrar datos válidos
        df = df.dropna(subset=['Total_value_added', 'GDP_usd', 'Population_wb'])
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()

def calculate_hhi(row):
    """Calcular índice de Herfindahl-Hirschman"""
    sectors = [
        'Agriculture_value_added',
        'Manufacturing_value_added', 
        'Construction_value_added',
        'Mining_manufacturing_utilities',
        'Other_activities'
    ]
    
    total_va = row['Total_value_added']
    if total_va == 0 or pd.isna(total_va):
        return np.nan
    
    hhi = 0
    for sector in sectors:
        if sector in row and not pd.isna(row[sector]) and row[sector] > 0:
            share = row[sector] / total_va
            hhi += share ** 2
    
    return hhi

def calculate_sector_shares(df):
    """Calcular participación sectorial en el PIB"""
    sectors = {
        'Agricultura': 'Agriculture_value_added',
        'Manufactura': 'Manufacturing_value_added',
        'Construcción': 'Construction_value_added', 
        'Minería': 'Mining_manufacturing_utilities',
        'Otros': 'Other_activities'
    }
    
    for name, col in sectors.items():
        if col in df.columns:
            df[f'{name}_share'] = (df[col] / df['Total_value_added']) * 100
    
    return df

def main():
    st.title("🏭 Dashboard de Estructura Productiva y Diversificación")
    st.markdown("**Análisis de composición sectorial y concentración económica**")
    
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
    
    # Filtro de región (simulado)
    regions = ['Todos', 'América', 'Europa', 'Asia', 'África', 'Oceanía']
    selected_region = st.sidebar.selectbox("Región", regions)
    
    # Aplicar filtros
    filtered_df = df[
        (df['Country'].isin(selected_countries)) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ].copy()
    
    if filtered_df.empty:
        st.warning("No hay datos para los filtros seleccionados")
        return
    
    # Calcular métricas
    filtered_df = calculate_sector_shares(filtered_df)
    filtered_df['HHI'] = filtered_df.apply(calculate_hhi, axis=1)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Países analizados", len(filtered_df['Country'].unique()))
    
    with col2:
        st.metric("Años de datos", f"{year_range[0]}-{year_range[1]}")
    
    with col3:
        avg_hhi = filtered_df['HHI'].mean()
        st.metric("HHI Promedio", f"{avg_hhi:.3f}")
    
    with col4:
        avg_gdp_pc = filtered_df['GDP_per_capita'].mean()
        st.metric("PIB per cápita promedio", f"${avg_gdp_pc:,.0f}")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Participación Sectorial", 
        "📈 Evolución HHI", 
        "🎯 Diversificación vs PIB", 
        "🏆 Ranking Diversificación",
        "🔥 Análisis Temporal"
    ])
    
    with tab1:
        st.header("Participación Sectorial en el PIB")
        
        # Gráfico de barras apiladas
        sector_cols = ['Agricultura_share', 'Manufactura_share', 'Construcción_share', 'Minería_share', 'Otros_share']
        available_sectors = [col for col in sector_cols if col in filtered_df.columns]
        
        if available_sectors:
            # Preparar datos para el gráfico
            plot_data = []
            for _, row in filtered_df.iterrows():
                for sector in available_sectors:
                    if not pd.isna(row[sector]):
                        plot_data.append({
                            'País': row['Country'],
                            'Año': row['Year'],
                            'Sector': sector.replace('_share', ''),
                            'Participación': row[sector]
                        })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                fig = px.bar(
                    plot_df, 
                    x='País', 
                    y='Participación',
                    color='Sector',
                    title="Participación Sectorial por País",
                    labels={'Participación': 'Participación (%)', 'País': 'País'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de participación
        st.subheader("Tabla de Participación Sectorial")
        if available_sectors:
            sector_table = filtered_df.groupby('Country')[available_sectors].mean().round(2)
            st.dataframe(sector_table, use_container_width=True)
    
    with tab2:
        st.header("Evolución del Índice de Herfindahl-Hirschman")
        
        # Gráfico de evolución temporal
        hhi_data = filtered_df[['Country', 'Year', 'HHI']].dropna()
        
        if not hhi_data.empty:
            fig = px.line(
                hhi_data,
                x='Year',
                y='HHI',
                color='Country',
                title="Evolución del HHI por País",
                labels={'HHI': 'Índice HHI', 'Year': 'Año'}
            )
            
            # Líneas de referencia para interpretación
            fig.add_hline(y=0.25, line_dash="dash", line_color="green", 
                         annotation_text="Diversificado (<0.25)")
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                         annotation_text="Moderado (0.25-0.5)")
            fig.add_hline(y=0.75, line_dash="dash", line_color="red", 
                         annotation_text="Concentrado (>0.5)")
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretación del HHI
            st.info("""
            **Interpretación del HHI:**
            - **0-0.25**: Economía diversificada y resiliente
            - **0.25-0.5**: Diversificación moderada
            - **0.5-1.0**: Economía concentrada y vulnerable
            """)
    
    with tab3:
        st.header("Diversificación vs PIB per Cápita")
        
        # Scatter plot
        scatter_data = filtered_df[['Country', 'HHI', 'GDP_per_capita']].dropna()
        
        if not scatter_data.empty:
            fig = px.scatter(
                scatter_data,
                x='HHI',
                y='GDP_per_capita',
                color='Country',
                title="Relación entre Diversificación (HHI) y PIB per Cápita",
                labels={'HHI': 'Índice HHI', 'GDP_per_capita': 'PIB per Cápita (USD)'},
                hover_data=['Country']
            )
            
            # Línea de tendencia
            fig.add_trace(go.Scatter(
                x=scatter_data['HHI'],
                y=np.poly1d(np.polyfit(scatter_data['HHI'], scatter_data['GDP_per_capita'], 1))(scatter_data['HHI']),
                mode='lines',
                name='Tendencia',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlación
            correlation = scatter_data['HHI'].corr(scatter_data['GDP_per_capita'])
            st.metric("Correlación HHI-PIB per cápita", f"{correlation:.3f}")
    
    with tab4:
        st.header("Ranking de Diversificación")
        
        # Calcular ranking
        ranking_data = filtered_df.groupby('Country').agg({
            'HHI': ['mean', 'std'],
            'GDP_per_capita': 'mean'
        }).round(4)
        
        ranking_data.columns = ['HHI_Promedio', 'HHI_Std', 'PIB_per_capita']
        ranking_data = ranking_data.sort_values('HHI_Promedio')
        ranking_data['Ranking'] = range(1, len(ranking_data) + 1)
        
        # Mostrar ranking
        st.subheader("Ranking por Diversificación (menor HHI = más diversificado)")
        display_ranking = ranking_data[['Ranking', 'HHI_Promedio', 'HHI_Std', 'PIB_per_capita']].head(20)
        st.dataframe(display_ranking, use_container_width=True)
        
        # Exportar ranking
        csv = ranking_data.to_csv()
        st.download_button(
            label="Descargar ranking completo",
            data=csv,
            file_name=f"ranking_diversificacion_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab5:
        st.header("Análisis Temporal Avanzado")
        
        # Heatmap de HHI
        st.subheader("Heatmap de HHI por País y Año")
        
        heatmap_data = filtered_df.pivot_table(
            values='HHI', 
            index='Country', 
            columns='Year', 
            aggfunc='mean'
        )
        
        if not heatmap_data.empty:
            fig = px.imshow(
                heatmap_data,
                title="Heatmap de HHI por País y Año",
                labels={'x': 'Año', 'y': 'País', 'color': 'HHI'},
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Evolución sectorial
        st.subheader("Evolución de la Estructura Sectorial")
        
        if available_sectors:
            sector_evolution = filtered_df.groupby('Year')[available_sectors].mean()
            
            fig = px.area(
                sector_evolution,
                title="Evolución de la Participación Sectorial Promedio",
                labels={'value': 'Participación (%)', 'index': 'Año'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard de Estructura Productiva y Diversificación** | Datos: World Bank, FRED")

if __name__ == "__main__":
    main()
