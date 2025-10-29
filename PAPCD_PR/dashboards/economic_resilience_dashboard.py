"""
Dashboard de Resiliencia y Estabilidad Económica
Medición de la capacidad de los países para resistir y recuperarse de crisis económicas
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
    page_title="Resiliencia y Estabilidad Económica",
    page_icon="🧠",
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
        df['GDP_usd'] = pd.to_numeric(df['GDP_usd'], errors='coerce')
        df['Population_wb'] = pd.to_numeric(df['Population_wb'], errors='coerce')
        
        # Calcular crecimiento del PIB
        df_sorted = df.sort_values(['Country', 'Year'])
        df_sorted['GDP_growth'] = df_sorted.groupby('Country')['GDP_usd'].pct_change() * 100
        
        # PIB per cápita
        df_sorted['GDP_per_capita'] = df_sorted['GDP_usd'] / df_sorted['Population_wb']
        
        # Filtrar datos válidos
        df_sorted = df_sorted.dropna(subset=['GDP_usd', 'Population_wb', 'GDP_growth'])
        
        return df_sorted
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()

def calculate_volatility(df, window=5):
    """Calcular volatilidad del PIB (desviación estándar móvil)"""
    df_sorted = df.sort_values(['Country', 'Year'])
    df_sorted['GDP_volatility'] = df_sorted.groupby('Country')['GDP_growth'].rolling(
        window=window, min_periods=3
    ).std().reset_index(0, drop=True)
    
    return df_sorted

def identify_crises_and_recoveries(df):
    """Identificar caídas y recuperaciones del PIB por década"""
    df_sorted = df.sort_values(['Country', 'Year'])
    
    # Identificar caídas (crecimiento negativo significativo)
    df_sorted['Crisis'] = (df_sorted['GDP_growth'] < -2.0).astype(int)
    
    # Calcular década
    df_sorted['Decade'] = (df_sorted['Year'] // 10) * 10
    
    # Contar crisis por década
    crisis_by_decade = df_sorted.groupby(['Country', 'Decade']).agg({
        'Crisis': 'sum',
        'GDP_growth': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    crisis_by_decade.columns = ['Crisis_Count', 'Avg_Growth', 'Growth_Volatility', 'Min_Growth', 'Max_Growth']
    crisis_by_decade = crisis_by_decade.reset_index()
    
    return crisis_by_decade

def calculate_resilience_index(df):
    """Calcular índice de resiliencia macroeconómica"""
    # Métricas de resiliencia
    resilience_metrics = df.groupby('Country').agg({
        'GDP_growth': ['mean', 'std', 'min'],
        'GDP_volatility': 'mean',
        'GDP_per_capita': 'mean'
    }).round(3)
    
    resilience_metrics.columns = ['Avg_Growth', 'Growth_Volatility', 'Min_Growth', 'Avg_Volatility', 'GDP_per_capita']
    resilience_metrics = resilience_metrics.reset_index()
    
    # Normalizar métricas (0-1, donde 1 es mejor)
    for col in ['Avg_Growth', 'GDP_per_capita']:
        resilience_metrics[f'{col}_norm'] = (
            resilience_metrics[col] - resilience_metrics[col].min()
        ) / (resilience_metrics[col].max() - resilience_metrics[col].min())
    
    for col in ['Growth_Volatility', 'Avg_Volatility', 'Min_Growth']:
        resilience_metrics[f'{col}_norm'] = 1 - (
            resilience_metrics[col] - resilience_metrics[col].min()
        ) / (resilience_metrics[col].max() - resilience_metrics[col].min())
    
    # Calcular índice compuesto de resiliencia
    resilience_metrics['Resilience_Index'] = (
        resilience_metrics['Avg_Growth_norm'] * 0.3 +
        resilience_metrics['GDP_per_capita_norm'] * 0.2 +
        resilience_metrics['Growth_Volatility_norm'] * 0.2 +
        resilience_metrics['Avg_Volatility_norm'] * 0.2 +
        resilience_metrics['Min_Growth_norm'] * 0.1
    )
    
    return resilience_metrics.sort_values('Resilience_Index', ascending=False)

def main():
    st.title("🧠 Dashboard de Resiliencia y Estabilidad Económica")
    st.markdown("**Medición de la capacidad de los países para resistir y recuperarse de crisis económicas**")
    
    # Cargar datos
    df = load_data()
    
    if df.empty:
        st.error("No se pudieron cargar los datos")
        return
    
    # Calcular métricas de resiliencia
    df = calculate_volatility(df)
    crisis_data = identify_crises_and_recoveries(df)
    resilience_data = calculate_resilience_index(df)
    
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
    
    # Filtro de década (usar crisis_data que ya tiene la columna Decade)
    decades = sorted(crisis_data['Decade'].unique()) if not crisis_data.empty else []
    selected_decades = st.sidebar.multiselect(
        "Seleccionar décadas",
        decades,
        default=decades
    )
    
    # Aplicar filtros
    filtered_df = df[
        (df['Country'].isin(selected_countries)) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ].copy()
    
    filtered_crisis = crisis_data[
        (crisis_data['Country'].isin(selected_countries)) &
        (crisis_data['Decade'].isin(selected_decades))
    ]
    
    filtered_resilience = resilience_data[
        resilience_data['Country'].isin(selected_countries)
    ]
    
    if filtered_df.empty:
        st.warning("No hay datos para los filtros seleccionados")
        return
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Países analizados", len(filtered_df['Country'].unique()))
    
    with col2:
        avg_volatility = filtered_df['GDP_volatility'].mean()
        st.metric("Volatilidad promedio", f"{avg_volatility:.2f}%")
    
    with col3:
        total_crises = filtered_crisis['Crisis_Count'].sum()
        st.metric("Total crisis identificadas", f"{total_crises}")
    
    with col4:
        avg_resilience = filtered_resilience['Resilience_Index'].mean()
        st.metric("Índice resiliencia promedio", f"{avg_resilience:.3f}")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Volatilidad del PIB", 
        "📉 Crisis y Recuperaciones", 
        "🏆 Ranking Resiliencia", 
        "📈 Análisis Temporal",
        "🔍 Comparación de Países"
    ])
    
    with tab1:
        st.header("Volatilidad del PIB (Desviación Estándar Móvil)")
        
        # Gráfico de volatilidad por país
        volatility_data = filtered_df[['Country', 'Year', 'GDP_volatility', 'GDP_growth']].dropna()
        
        if not volatility_data.empty:
            fig = px.line(
                volatility_data,
                x='Year',
                y='GDP_volatility',
                color='Country',
                title="Evolución de la Volatilidad del PIB por País",
                labels={'GDP_volatility': 'Volatilidad (%)', 'Year': 'Año'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap de volatilidad
        st.subheader("Heatmap de Volatilidad por País y Año")
        
        volatility_pivot = filtered_df.pivot_table(
            values='GDP_volatility', 
            index='Country', 
            columns='Year', 
            aggfunc='mean'
        )
        
        if not volatility_pivot.empty:
            fig = px.imshow(
                volatility_pivot,
                title="Heatmap de Volatilidad del PIB",
                labels={'x': 'Año', 'y': 'País', 'color': 'Volatilidad (%)'},
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas de volatilidad
        st.subheader("Estadísticas de Volatilidad por País")
        volatility_stats = filtered_df.groupby('Country')['GDP_volatility'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)
        volatility_stats.columns = ['Promedio', 'Desv_Est', 'Mínimo', 'Máximo']
        st.dataframe(volatility_stats, use_container_width=True)
    
    with tab2:
        st.header("Caídas y Recuperaciones del PIB por Década")
        
        # Gráfico de crisis por década
        if not filtered_crisis.empty:
            fig = px.bar(
                filtered_crisis,
                x='Decade',
                y='Crisis_Count',
                color='Country',
                title="Número de Crisis por Década y País",
                labels={'Crisis_Count': 'Número de Crisis', 'Decade': 'Década'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de recuperación
        st.subheader("Análisis de Recuperación Post-Crisis")
        
        # Calcular métricas de recuperación
        recovery_analysis = filtered_crisis.groupby('Country').agg({
            'Crisis_Count': 'sum',
            'Avg_Growth': 'mean',
            'Min_Growth': 'min',
            'Max_Growth': 'max'
        }).round(3)
        
        recovery_analysis['Recovery_Strength'] = (
            recovery_analysis['Max_Growth'] - recovery_analysis['Min_Growth']
        )
        recovery_analysis = recovery_analysis.sort_values('Recovery_Strength', ascending=False)
        
        st.dataframe(recovery_analysis, use_container_width=True)
        
        # Gráfico de fuerza de recuperación
        fig = px.scatter(
            recovery_analysis,
            x='Crisis_Count',
            y='Recovery_Strength',
            size='Avg_Growth',
            hover_name=recovery_analysis.index,
            title="Fuerza de Recuperación vs Número de Crisis",
            labels={'Crisis_Count': 'Número de Crisis', 'Recovery_Strength': 'Fuerza de Recuperación'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Ranking de Resiliencia Macroeconómica")
        
        # Mostrar ranking completo
        st.subheader("Ranking Completo de Resiliencia")
        display_resilience = filtered_resilience[['Country', 'Resilience_Index', 'Avg_Growth', 
                                                  'GDP_per_capita', 'Growth_Volatility']].head(20)
        display_resilience.columns = ['País', 'Índice Resiliencia', 'Crecimiento Promedio', 
                                     'PIB per Cápita', 'Volatilidad Crecimiento']
        st.dataframe(display_resilience, use_container_width=True)
        
        # Gráfico de resiliencia
        fig = px.bar(
            display_resilience.head(15),
            x='Índice Resiliencia',
            y='País',
            orientation='h',
            title="Top 15 Países por Resiliencia Macroeconómica",
            labels={'Índice Resiliencia': 'Índice de Resiliencia'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de componentes de resiliencia
        st.subheader("Componentes del Índice de Resiliencia")
        
        # Crear gráfico de radar para los mejores países
        top_countries = filtered_resilience.head(5)
        
        fig = go.Figure()
        
        for country in top_countries['Country']:
            country_data = top_countries[top_countries['Country'] == country].iloc[0]
            
            fig.add_trace(go.Scatterpolar(
                r=[
                    country_data['Avg_Growth_norm'],
                    country_data['GDP_per_capita_norm'],
                    country_data['Growth_Volatility_norm'],
                    country_data['Avg_Volatility_norm'],
                    country_data['Min_Growth_norm']
                ],
                theta=['Crecimiento', 'PIB per Cápita', 'Estabilidad Crecimiento', 
                       'Baja Volatilidad', 'Resistencia a Caídas'],
                fill='toself',
                name=country
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Perfil de Resiliencia - Top 5 Países"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Análisis Temporal de Resiliencia")
        
        # Evolución del índice de resiliencia
        st.subheader("Evolución Temporal de la Resiliencia")
        
        # Calcular resiliencia por período
        period_resilience = filtered_df.groupby(['Country', 'Year']).agg({
            'GDP_growth': 'mean',
            'GDP_volatility': 'mean'
        }).reset_index()
        
        # Calcular índice simple por año
        period_resilience['Simple_Resilience'] = (
            period_resilience['GDP_growth'] / (1 + period_resilience['GDP_volatility'])
        )
        
        fig = px.line(
            period_resilience,
            x='Year',
            y='Simple_Resilience',
            color='Country',
            title="Evolución de la Resiliencia por País",
            labels={'Simple_Resilience': 'Índice de Resiliencia Simple', 'Year': 'Año'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de tendencias
        st.subheader("Análisis de Tendencias de Resiliencia")
        
        # Calcular tendencia por país
        trends = []
        for country in filtered_df['Country'].unique():
            country_data = period_resilience[period_resilience['Country'] == country]
            if len(country_data) > 5:
                try:
                    # Usar método más robusto para calcular tendencias
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        country_data['Year'], 
                        country_data['Simple_Resilience']
                    )
                    trends.append({
                        'Country': country,
                        'Trend': slope,
                        'R_squared': r_value**2
                    })
                except (ValueError, np.linalg.LinAlgError):
                    # Si falla el ajuste, usar método simple
                    try:
                        correlation = np.corrcoef(country_data['Year'], country_data['Simple_Resilience'])[0,1]
                        if not np.isnan(correlation):
                            trends.append({
                                'Country': country,
                                'Trend': correlation * 0.01,  # Escalar para que sea comparable
                                'R_squared': correlation**2
                            })
                    except:
                        continue
        
        trends_df = pd.DataFrame(trends).sort_values('Trend', ascending=False)
        
        if not trends_df.empty:
            fig = px.bar(
                trends_df.head(15),
                x='Country',
                y='Trend',
                title="Tendencias de Resiliencia (Países con Mejora)",
                labels={'Trend': 'Tendencia de Mejora', 'Country': 'País'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(trends_df.round(3), use_container_width=True)
    
    with tab5:
        st.header("Comparación de Países")
        
        # Scatter plot de resiliencia vs volatilidad
        fig = px.scatter(
            filtered_resilience,
            x='Growth_Volatility',
            y='Resilience_Index',
            size='GDP_per_capita',
            color='Avg_Growth',
            hover_name='Country',
            title="Resiliencia vs Volatilidad del Crecimiento",
            labels={'Growth_Volatility': 'Volatilidad del Crecimiento', 
                   'Resilience_Index': 'Índice de Resiliencia'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Matriz de correlación
        st.subheader("Correlaciones entre Métricas de Resiliencia")
        
        correlation_data = filtered_resilience[['Resilience_Index', 'Avg_Growth', 
                                              'GDP_per_capita', 'Growth_Volatility', 
                                              'Avg_Volatility']].corr()
        
        fig = px.imshow(
            correlation_data,
            title="Matriz de Correlación - Métricas de Resiliencia",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Exportar datos de resiliencia
        csv = filtered_resilience.to_csv()
        st.download_button(
            label="Descargar datos de resiliencia",
            data=csv,
            file_name=f"resiliencia_economica_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard de Resiliencia y Estabilidad Económica** | Datos: World Bank, FRED")

if __name__ == "__main__":
    main()
