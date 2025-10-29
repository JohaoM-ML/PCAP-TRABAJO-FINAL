import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Macroeconomic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the merged dataset"""
    df = pd.read_csv('data/external/global_merged_all.csv')
    
    # Clean and prepare data
    df['Country'] = df['Country'].str.strip()
    df['Year'] = df['Year'].astype(int)
    
    # Calculate GDP per capita
    df['GDP_per_capita'] = df['GDP_usd'] / df['Population_wb']
    df['GDP_per_capita_main'] = df['GDP_main'] / df['Population_main']
    
    # Calculate growth rates
    df = df.sort_values(['Country', 'Year'])
    df['GDP_growth_rate'] = df.groupby('Country')['GDP_usd'].pct_change() * 100
    df['GDP_per_capita_growth'] = df.groupby('Country')['GDP_per_capita'].pct_change() * 100
    
    # Create region mapping (simplified)
    region_mapping = {
        'United States': 'North America',
        'Canada': 'North America',
        'Mexico': 'North America',
        'Brazil': 'South America',
        'Argentina': 'South America',
        'Chile': 'South America',
        'Germany': 'Europe',
        'France': 'Europe',
        'United Kingdom': 'Europe',
        'Italy': 'Europe',
        'Spain': 'Europe',
        'China': 'Asia',
        'Japan': 'Asia',
        'India': 'Asia',
        'South Korea': 'Asia',
        'Australia': 'Oceania',
        'New Zealand': 'Oceania',
        'South Africa': 'Africa',
        'Nigeria': 'Africa',
        'Egypt': 'Africa'
    }
    
    # Add region column
    df['Region'] = df['Country'].map(region_mapping).fillna('Other')
    
    return df

def create_gdp_overview(df, selected_countries, start_year, end_year):
    """Create GDP overview charts"""
    filtered_df = df[
        (df['Country'].isin(selected_countries)) & 
        (df['Year'] >= start_year) & 
        (df['Year'] <= end_year)
    ].copy()
    
    if filtered_df.empty:
        st.warning("No data available for the selected criteria.")
        return
    
    # GDP Levels
    fig_gdp = px.line(
        filtered_df, 
        x='Year', 
        y='GDP_usd', 
        color='Country',
        title='GDP Levels (USD)',
        labels={'GDP_usd': 'GDP (USD)', 'Year': 'Year'}
    )
    fig_gdp.update_layout(height=400)
    
    # GDP Growth Rates
    fig_growth = px.line(
        filtered_df, 
        x='Year', 
        y='GDP_growth_rate', 
        color='Country',
        title='GDP Growth Rates (%)',
        labels={'GDP_growth_rate': 'GDP Growth Rate (%)', 'Year': 'Year'}
    )
    fig_growth.update_layout(height=400)
    
    # GDP per Capita
    fig_per_capita = px.line(
        filtered_df, 
        x='Year', 
        y='GDP_per_capita', 
        color='Country',
        title='GDP per Capita (USD)',
        labels={'GDP_per_capita': 'GDP per Capita (USD)', 'Year': 'Year'}
    )
    fig_per_capita.update_layout(height=400)
    
    return fig_gdp, fig_growth, fig_per_capita

def create_sector_analysis(df, selected_countries, selected_year):
    """Create sectoral composition analysis"""
    filtered_df = df[
        (df['Country'].isin(selected_countries)) & 
        (df['Year'] == selected_year)
    ].copy()
    
    if filtered_df.empty:
        st.warning("No data available for the selected criteria.")
        return None
    
    # Calculate sector shares
    sector_columns = [
        'Agriculture_value_added',
        'Manufacturing_value_added', 
        'Construction_value_added',
        'Transport_communication',
        'Wholesale_retail_trade',
        'Other_activities'
    ]
    
    sector_data = []
    for country in selected_countries:
        country_data = filtered_df[filtered_df['Country'] == country]
        if not country_data.empty:
            total_value = country_data['Total_value_added'].iloc[0]
            if pd.notna(total_value) and total_value > 0:
                row = {'Country': country}
                for sector in sector_columns:
                    sector_value = country_data[sector].iloc[0]
                    if pd.notna(sector_value):
                        row[sector.replace('_value_added', '').replace('_', ' ').title()] = (sector_value / total_value) * 100
                    else:
                        row[sector.replace('_value_added', '').replace('_', ' ').title()] = 0
                sector_data.append(row)
    
    if not sector_data:
        return None
    
    sector_df = pd.DataFrame(sector_data)
    
    # Create stacked bar chart
    fig_sector = px.bar(
        sector_df,
        x='Country',
        y=[col for col in sector_df.columns if col != 'Country'],
        title=f'Value Added Structure by Sector ({selected_year})',
        labels={'value': 'Percentage of Total Value Added', 'variable': 'Sector'}
    )
    fig_sector.update_layout(height=500, barmode='stack')
    
    return fig_sector

def create_demographic_analysis(df, selected_countries, start_year, end_year):
    """Create demographic trends analysis"""
    filtered_df = df[
        (df['Country'].isin(selected_countries)) & 
        (df['Year'] >= start_year) & 
        (df['Year'] <= end_year)
    ].copy()
    
    if filtered_df.empty:
        st.warning("No data available for the selected criteria.")
        return None, None
    
    # Population trends
    fig_population = px.line(
        filtered_df, 
        x='Year', 
        y='Population_wb', 
        color='Country',
        title='Population Evolution',
        labels={'Population_wb': 'Population', 'Year': 'Year'}
    )
    fig_population.update_layout(height=400)
    
    # Market size (GDP)
    fig_market = px.line(
        filtered_df, 
        x='Year', 
        y='GDP_usd', 
        color='Country',
        title='Market Size (GDP)',
        labels={'GDP_usd': 'GDP (USD)', 'Year': 'Year'}
    )
    fig_market.update_layout(height=400)
    
    return fig_population, fig_market

def create_growth_ranking(df, start_year, end_year, top_n=20):
    """Create growth ranking by regions"""
    # Calculate average growth rate for each country
    growth_data = []
    
    for country in df['Country'].unique():
        country_data = df[
            (df['Country'] == country) & 
            (df['Year'] >= start_year) & 
            (df['Year'] <= end_year)
        ].copy()
        
        if len(country_data) > 1:
            # Calculate average growth rate
            avg_growth = country_data['GDP_growth_rate'].mean()
            region = country_data['Region'].iloc[0]
            
            if pd.notna(avg_growth):
                growth_data.append({
                    'Country': country,
                    'Region': region,
                    'Average_Growth_Rate': avg_growth,
                    'Final_GDP': country_data['GDP_usd'].iloc[-1] if pd.notna(country_data['GDP_usd'].iloc[-1]) else 0
                })
    
    growth_df = pd.DataFrame(growth_data)
    
    if growth_df.empty:
        return None
    
    # Get top performers
    top_growth = growth_df.nlargest(top_n, 'Average_Growth_Rate')
    
    # Create ranking chart
    fig_ranking = px.bar(
        top_growth,
        x='Average_Growth_Rate',
        y='Country',
        color='Region',
        title=f'Top {top_n} Countries by Average GDP Growth Rate ({start_year}-{end_year})',
        labels={'Average_Growth_Rate': 'Average Growth Rate (%)', 'Country': 'Country'},
        orientation='h'
    )
    fig_ranking.update_layout(height=600)
    
    return fig_ranking, top_growth

def main():
    # Load data
    df = load_data()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Macroeconomic Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**General Macroeconomic Overview for Global Analysis**")
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters & Controls")
    
    # Country selection
    available_countries = sorted(df['Country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        available_countries,
        default=['United States', 'China', 'Germany', 'Japan', 'India'] if all(c in available_countries for c in ['United States', 'China', 'Germany', 'Japan', 'India']) else available_countries[:5]
    )
    
    # Year range selection
    min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
    start_year, end_year = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Analysis year for sector analysis
    analysis_year = st.sidebar.selectbox(
        "Select Year for Sector Analysis",
        options=sorted(df['Year'].unique(), reverse=True),
        index=0
    )
    
    # Main content
    if not selected_countries:
        st.warning("Please select at least one country from the sidebar.")
        return
    
    # Key Metrics
    st.markdown('<div class="section-header">📈 Key Economic Indicators</div>', unsafe_allow_html=True)
    
    latest_data = df[
        (df['Country'].isin(selected_countries)) & 
        (df['Year'] == end_year)
    ]
    
    if not latest_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_gdp = latest_data['GDP_usd'].mean()
            st.metric("Average GDP (USD)", f"${avg_gdp:,.0f}" if pd.notna(avg_gdp) else "N/A")
        
        with col2:
            avg_gdp_per_capita = latest_data['GDP_per_capita'].mean()
            st.metric("Average GDP per Capita", f"${avg_gdp_per_capita:,.0f}" if pd.notna(avg_gdp_per_capita) else "N/A")
        
        with col3:
            total_population = latest_data['Population_wb'].sum()
            st.metric("Total Population", f"{total_population:,.0f}" if pd.notna(total_population) else "N/A")
        
        with col4:
            avg_growth = latest_data['GDP_growth_rate'].mean()
            st.metric("Average Growth Rate", f"{avg_growth:.2f}%" if pd.notna(avg_growth) else "N/A")
    
    # GDP Analysis
    st.markdown('<div class="section-header">💰 GDP Analysis</div>', unsafe_allow_html=True)
    
    fig_gdp, fig_growth, fig_per_capita = create_gdp_overview(df, selected_countries, start_year, end_year)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_gdp, use_container_width=True)
    with col2:
        st.plotly_chart(fig_growth, use_container_width=True)
    
    st.plotly_chart(fig_per_capita, use_container_width=True)
    
    # Sector Analysis
    st.markdown('<div class="section-header">🏭 Economic Structure Analysis</div>', unsafe_allow_html=True)
    
    fig_sector = create_sector_analysis(df, selected_countries, analysis_year)
    if fig_sector:
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.warning("No sectoral data available for the selected criteria.")
    
    # Demographic Analysis
    st.markdown('<div class="section-header">👥 Demographic Trends & Market Size</div>', unsafe_allow_html=True)
    
    fig_population, fig_market = create_demographic_analysis(df, selected_countries, start_year, end_year)
    if fig_population and fig_market:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_population, use_container_width=True)
        with col2:
            st.plotly_chart(fig_market, use_container_width=True)
    
    # Growth Ranking
    st.markdown('<div class="section-header">🏆 Growth Performance Ranking</div>', unsafe_allow_html=True)
    
    fig_ranking, ranking_data = create_growth_ranking(df, start_year, end_year)
    if fig_ranking is not None:
        st.plotly_chart(fig_ranking, use_container_width=True)
        
        # Display ranking table
        st.subheader("Growth Ranking Table")
        st.dataframe(
            ranking_data[['Country', 'Region', 'Average_Growth_Rate', 'Final_GDP']].round(2),
            use_container_width=True
        )
        
        # Download button
        csv = ranking_data.to_csv(index=False)
        st.download_button(
            label="Download Growth Ranking Data",
            data=csv,
            file_name=f"growth_ranking_{start_year}_{end_year}.csv",
            mime="text/csv"
        )
    
    # Export functionality
    st.markdown('<div class="section-header">📥 Export Data</div>', unsafe_allow_html=True)
    
    # Filter data for export
    export_data = df[
        (df['Country'].isin(selected_countries)) & 
        (df['Year'] >= start_year) & 
        (df['Year'] <= end_year)
    ]
    
    if not export_data.empty:
        csv_export = export_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Dataset",
            data=csv_export,
            file_name=f"macroeconomic_data_{start_year}_{end_year}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Sources:** World Bank, FRED, and merged economic datasets")
    st.markdown("**Last Updated:** Based on available data through 2021")

if __name__ == "__main__":
    main()

