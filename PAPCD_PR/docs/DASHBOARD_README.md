# 📊 Macroeconomic Dashboard

An interactive web-based dashboard for comprehensive macroeconomic analysis using merged data from World Bank, FRED, and other economic sources.

## 🚀 Quick Start

### Option 1: Using the Launcher Script
```bash
python run_dashboard.py
```

### Option 2: Direct Streamlit Command
```bash
streamlit run macroeconomic_dashboard.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## 📋 Features

### 🔧 Interactive Controls
- **Country Selection**: Choose multiple countries for comparison
- **Time Period Filters**: Select custom year ranges (1970-2021)
- **Analysis Year**: Choose specific year for sectoral analysis

### 📈 Economic Analysis Components

#### 1. **GDP Analysis**
- **GDP Levels**: Absolute GDP values over time
- **GDP Growth Rates**: Annual growth percentage trends
- **GDP per Capita**: Per capita economic performance

#### 2. **Economic Structure Analysis**
- **Value Added by Sector**: ISIC A-P sectoral composition
  - Agriculture, hunting, forestry, fishing
  - Manufacturing
  - Construction
  - Transport, storage and communication
  - Wholesale, retail trade, restaurants and hotels
  - Other Activities

#### 3. **Demographic Trends & Market Size**
- **Population Evolution**: Historical population trends
- **Market Size**: GDP-based market size analysis
- **Demographic Indicators**: Population growth patterns

#### 4. **Growth Performance Ranking**
- **Regional Comparison**: Countries ranked by growth performance
- **Top Performers**: Best performing economies by region
- **Growth Rate Analysis**: Average growth rates over selected periods

### 📊 Visualization Types
- **Line Charts**: Time series trends
- **Bar Charts**: Comparative analysis
- **Stacked Bar Charts**: Sectoral composition
- **Horizontal Bar Charts**: Ranking visualizations
- **Interactive Tables**: Detailed data views

### 📥 Export Functionality
- **Growth Ranking Export**: Download top performers data
- **Filtered Dataset Export**: Export custom filtered data
- **CSV Format**: All exports in CSV format for further analysis

## 🗂️ Data Sources

The dashboard uses the merged dataset (`global_merged_all.csv`) which combines:

1. **Main Economic Data**: GDP, GNI, sectoral value added, population
2. **World Bank Data**: GDP growth, exports, imports, capital formation
3. **FRED Data**: Oil prices, Federal funds rate, US inflation index

## 📊 Key Metrics Displayed

### Economic Indicators
- Total GDP (USD)
- GDP per Capita
- GDP Growth Rates
- Population
- Market Size

### Sectoral Analysis
- Agriculture share
- Manufacturing share
- Construction share
- Services share
- Other activities share

### Regional Analysis
- North America
- South America
- Europe
- Asia
- Oceania
- Africa
- Other regions

## 🎯 Use Cases

### For Researchers
- Comparative economic analysis
- Historical trend analysis
- Sectoral composition studies
- Growth performance evaluation

### For Decision Makers
- Market size assessment
- Economic performance benchmarking
- Regional economic comparisons
- Investment opportunity analysis

### For Students
- Economic data visualization
- Interactive learning tool
- Historical economic analysis
- Comparative studies

## 🔧 Technical Requirements

### Dependencies
- Python 3.8+
- Streamlit
- Plotly
- Pandas
- NumPy

### Installation
```bash
pip install -r requirements.txt
```

## 📱 Dashboard Interface

### Sidebar Controls
- Country multi-select dropdown
- Year range slider
- Analysis year selector

### Main Dashboard Sections
1. **Key Economic Indicators**: Summary metrics
2. **GDP Analysis**: Three comprehensive charts
3. **Economic Structure Analysis**: Sectoral breakdown
4. **Demographic Trends**: Population and market analysis
5. **Growth Performance Ranking**: Comparative rankings
6. **Export Data**: Download functionality

## 🎨 Customization

The dashboard is built with modular components that can be easily customized:

- **Color schemes**: Modify CSS in the dashboard file
- **Chart types**: Change Plotly chart configurations
- **Metrics**: Add or remove economic indicators
- **Filters**: Extend filtering capabilities

## 📈 Data Coverage

- **Countries**: 220 countries
- **Time Period**: 1970-2021
- **Observations**: 10,512 data points
- **Variables**: 36 economic indicators

## 🚨 Notes

- Some data may have missing values for certain countries/years
- Growth rates are calculated as year-over-year percentage changes
- Regional classifications are simplified for demonstration
- Oil price data is global and applies to all countries

## 🔄 Updates

The dashboard automatically loads the latest merged dataset. To update with new data:

1. Run the data merge script: `python merge_all_datasets.py`
2. Restart the dashboard
3. New data will be automatically loaded

## 📞 Support

For technical issues or feature requests, please refer to the project documentation or contact the development team.

---

**Happy Analyzing! 📊✨**

