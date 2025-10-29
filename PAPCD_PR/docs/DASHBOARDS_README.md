# 📊 Dashboards de Análisis Macroeconómico

Esta carpeta contiene todos los dashboards interactivos para el análisis macroeconómico.

## 🎯 Dashboards Disponibles

### 1. 📈 Dashboard Macroeconómico Principal
- **Archivo:** `macroeconomic_dashboard.py`
- **Puerto:** 8501
- **Descripción:** Análisis general de GDP, estructura económica, demografía y ranking de crecimiento
- **Lanzamiento:** `python run_dashboard.py` o `launch_dashboard.bat`

### 2. 🏭 Estructura Productiva y Diversificación
- **Archivo:** `productive_structure_dashboard.py`
- **Puerto:** 8502
- **Descripción:** Análisis de composición sectorial y concentración económica (HHI)
- **Lanzamiento:** `streamlit run productive_structure_dashboard.py --server.port 8502`

### 3. ⚖️ Equilibrio Macroeconómico y Demanda Agregada
- **Archivo:** `macroeconomic_balance_dashboard.py`
- **Puerto:** 8503
- **Descripción:** Composición del gasto nacional, balance comercial, ciclos económicos
- **Lanzamiento:** `python run_balance_dashboard.py` o `launch_balance_dashboard.bat`

### 4. 🌐 Conectividad Global y Factores Externos
- **Archivo:** `global_connectivity_dashboard.py`
- **Puerto:** 8504
- **Descripción:** Influencia del contexto internacional sobre economías locales
- **Lanzamiento:** `python run_connectivity_dashboard.py` o `launch_connectivity_dashboard.bat`

### 5. 🧠 Resiliencia y Estabilidad Económica
- **Archivo:** `economic_resilience_dashboard.py`
- **Puerto:** 8505
- **Descripción:** Capacidad de resistencia y recuperación de crisis económicas
- **Lanzamiento:** `python run_resilience_dashboard.py` o `launch_resilience_dashboard.bat`

## 🚀 Formas de Ejecutar

### Opción 1: Scripts Python
```bash
cd dashboards
python run_dashboard.py                    # Puerto 8501
streamlit run productive_structure_dashboard.py --server.port 8502
python run_balance_dashboard.py            # Puerto 8503
python run_connectivity_dashboard.py        # Puerto 8504
```

### Opción 2: Archivos Batch (Windows)
```bash
cd dashboards
launch_dashboard.bat                        # Puerto 8501
launch_balance_dashboard.bat                # Puerto 8503
launch_connectivity_dashboard.bat           # Puerto 8504
```

### Opción 3: Streamlit Directo
```bash
cd dashboards
streamlit run macroeconomic_dashboard.py --server.port 8501
streamlit run productive_structure_dashboard.py --server.port 8502
streamlit run macroeconomic_balance_dashboard.py --server.port 8503
streamlit run global_connectivity_dashboard.py --server.port 8504
```

## 📊 Características de los Dashboards

### Funcionalidades Comunes:
- ✅ **Filtros interactivos** por país y año
- ✅ **Gráficos interactivos** con Plotly
- ✅ **Métricas en tiempo real**
- ✅ **Exportación de datos**
- ✅ **Diseño responsive**

### Datos Utilizados:
- **Dataset:** `global_merged_all.csv`
- **Período:** 1970-2024
- **Países:** 214 países
- **Fuentes:** World Bank, FRED, datos nacionales

## 🔧 Requisitos

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- NumPy

## 📝 Notas

- Los dashboards pueden ejecutarse simultáneamente en diferentes puertos
- Cada dashboard es independiente y puede ejecutarse por separado
- Los datos se cargan desde `../data/external/global_merged_all.csv`
