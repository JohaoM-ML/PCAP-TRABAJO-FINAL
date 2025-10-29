# 📊 Proyecto de Análisis Macroeconómico y Predicciones 2030

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema completo de análisis macroeconómico que combina múltiples fuentes de datos (World Bank, FRED) para crear modelos de machine learning que predicen el crecimiento del PIB de 133 países hasta 2030. El sistema incluye dashboards interactivos, análisis de regresión avanzado y visualizaciones dinámicas.

## 🏗️ Arquitectura del Proyecto

```
PAPCD_PR/
├── 📁 data/                               # Datos del proyecto
│   ├── 📁 external/                       # Datos externos
│   │   ├── global_merged_all.csv          # Dataset principal consolidado
│   │   ├── worldbank_data_clean.csv       # Datos del Banco Mundial
│   │   └── fred_data.csv                  # Datos de FRED
│   └── 📁 reference/                      # Datos de referencia
│       └── worldbank_countries.csv        # Referencia de países
├── 📁 src/                                # Código fuente principal
│   ├── __init__.py
│   ├── merge_datasets.py                  # Script de consolidación de datos
│   ├── macroeconomic_regression.py        # Modelos de regresión básicos
│   ├── macroeconomic_regression_corrected.py # Modelos corregidos (endogeneidad)
│   ├── enhanced_macroeconomic_ml.py       # Modelos ML avanzados
│   ├── api_worldbank.py                   # API del Banco Mundial
│   └── api_fred.py                        # API de FRED
├── 📁 scripts/                            # Scripts de análisis
│   ├── generate_predictions_2030.py       # Generador de predicciones
│   ├── generate_complete_report.py        # Generador de reporte completo
│   └── run_enhanced_ml_analysis_v3.py     # Análisis ML mejorado
├── 📁 dashboards/                         # Dashboards interactivos
│   ├── predictions_2030_dashboard.py      # Dashboard de predicciones
│   ├── macroeconomic_balance_dashboard.py # Dashboard de equilibrio macro
│   ├── productive_structure_dashboard.py  # Dashboard de estructura productiva
│   ├── global_connectivity_dashboard.py   # Dashboard de conectividad global
│   ├── economic_resilience_dashboard.py   # Dashboard de resiliencia económica
│   ├── run_predictions_dashboard.py       # Lanzador de dashboard principal
│   └── launch_all_dashboards.py           # Lanzador de todos los dashboards
├── 📁 results/                            # Resultados del análisis
│   ├── predictions_2030.csv               # Predicciones finales (1,197 registros)
│   └── 📁 enhanced_results_v3/            # Resultados del modelo mejorado
│       ├── model_performance_v3.csv
│       ├── high_correlations.csv
│       └── autocorrelation_analysis.csv
├── 📁 docs/                               # Documentación
│   ├── DASHBOARD_README.md                # Documentación de dashboards
│   ├── REGRESSION_MODULE_README.md        # Documentación de modelos
│   ├── ENDOGENEITY_ANALYSIS.md            # Análisis de endogeneidad
│   └── ENHANCED_ML_ANALYSIS_SUMMARY.md    # Resumen de ML mejorado
├── 📁 examples/                           # Ejemplos de uso
│   ├── compare_regression_approaches.py   # Comparación de enfoques
│   └── example_regression_analysis.py     # Ejemplo de análisis
├── 📁 tests/                              # Tests y validaciones
│   └── test_temporal_robustness.py        # Test de robustez temporal
├── 📄 config.py                           # Configuración central del proyecto
├── 📄 launch_analysis.py                  # Lanzador de análisis completo
├── 📄 launch_dashboards.py                # Lanzador de dashboards
├── 📄 launch_all.py                       # Lanzador completo del proyecto
├── 📄 requirements.txt                    # Dependencias del proyecto
└── 📄 README.md                           # Este archivo
```

## 🚀 Características Principales

### 📊 **Análisis de Datos**
- **Consolidación de múltiples fuentes:** World Bank, FRED, datos nacionales
- **10,512 observaciones** de 36 variables macroeconómicas
- **133 países** cubiertos (1970-2021)
- **Limpieza y estandarización** automática de datos

### 🤖 **Modelos de Machine Learning**
- **Ridge Regression** (R² = 0.899, RMSE = 1.61) - Modelo principal
- **Random Forest** (R² = 0.700)
- **XGBoost, Gradient Boosting, Lasso, ElasticNet**
- **Validación temporal** con TimeSeriesSplit
- **Análisis de endogeneidad** y corrección de sesgos

### 📈 **Predicciones hasta 2030**
- **1,197 predicciones** (133 países × 9 años)
- **Variabilidad temporal realista** con tendencias diferenciadas
- **Convergencia hacia crecimiento global** del 3%
- **Rango económico apropiado:** -13% a +22%

### 🎛️ **Dashboards Interactivos**
- **Dashboard de Predicciones 2030** - Visualización principal
- **Dashboard de Equilibrio Macroeconómico** - Análisis de demanda agregada
- **Dashboard de Estructura Productiva** - Diversificación económica
- **Dashboard de Conectividad Global** - Factores externos
- **Dashboard de Resiliencia Económica** - Estabilidad y crisis

## 🛠️ Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- pip (gestor de paquetes de Python)

### Instalación

#### **Opción A: Instalación Automática (Recomendada)**
```bash
# Clonar el repositorio
git clone <repository-url>
cd PAPCD_PR

# Windows
install.bat

# Linux/Mac
chmod +x install.sh
./install.sh

# O usando Python
python install_dependencies.py
```

#### **Opción B: Instalación Manual**
```bash
# Clonar el repositorio
git clone <repository-url>
cd PAPCD_PR

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias principales
pip install -r requirements.txt

# Instalar dependencias opcionales (opcional)
pip install geopandas folium great-expectations numba
```

#### **Verificar Instalación**
```bash
python -c "import pandas, numpy, plotly, streamlit, sklearn; print('Instalación exitosa')"
```

### Dependencias Principales
```
# Análisis de datos
pandas>=2.2.2
numpy>=1.26.4

# Visualización y dashboards
plotly>=5.24.1
streamlit>=1.38.0
matplotlib>=3.9.2
seaborn>=0.13.2

# Machine Learning
scikit-learn>=1.5.2
xgboost>=2.1.1
shap>=0.46.0

# APIs y datos externos
requests>=2.32.3
fredapi>=0.5.2
wbdata>=0.3.0
wbgapi>=1.0.12

# Análisis estadístico
statsmodels>=0.14.2
scipy>=1.9.0
```

## 🎮 Uso del Sistema

### 1. 🚀 **Lanzamiento Completo (Recomendado)**
```bash
# Ejecutar todo el proyecto: análisis + dashboards
python launch_all.py
```

### 2. 📊 **Solo Análisis**
```bash
# Ejecutar análisis completo (ML + predicciones + reporte)
python launch_analysis.py
```

### 3. 🎛️ **Solo Dashboards**
```bash
# Lanzar todos los dashboards
python launch_dashboards.py
```

### 4. 🔧 **Scripts Individuales**
```bash
# Generar predicciones hasta 2030
python scripts/generate_predictions_2030.py

# Generar reporte completo
python scripts/generate_complete_report.py

# Análisis ML mejorado (versión 3)
python scripts/run_enhanced_ml_analysis_v3.py

# Test de robustez temporal
python tests/test_temporal_robustness.py
```

### 5. 📈 **Análisis de Regresión**
```bash
# Modelos básicos
python src/macroeconomic_regression.py

# Modelos corregidos (endogeneidad)
python src/macroeconomic_regression_corrected.py
```

## 📊 Resultados Principales

### 🏆 **Mejor Modelo: Ridge Regression**
- **R² = 0.899** (excelente rendimiento)
- **RMSE = 1.61** (error muy bajo)
- **Sin sobreajuste** (gap negativo)
- **18 features finales** después de limpieza

### 📈 **Predicciones Globales (2022-2030)**
- **Crecimiento promedio:** 2.18%
- **Crecimiento mediano:** 2.29%
- **Rango:** -13.01% a +22.35%
- **Tendencia:** Convergencia hacia 3% global

### 🏆 **Top 5 Países por Crecimiento Promedio**
1. **Timor-Leste:** 13.50%
2. **Irlanda:** 6.41%
3. **Tajikistan:** 5.09%
4. **Ethiopia:** 4.83%
5. **Uzbekistan:** 4.35%

### 📉 **Países con Menor Crecimiento**
1. **Lebanon:** -4.84%
2. **Fiji:** -3.51%
3. **Palau:** -1.99%
4. **Zimbabwe:** -0.88%
5. **Lesotho:** -0.49%

## 🔧 Características Técnicas

### 📊 **Feature Engineering Avanzado**
- **Variables absolutas:** Inversión, exportaciones, importaciones
- **Variables estructurales:** Diversificación (HHI), términos de intercambio
- **Variables externas:** Petróleo, tasas de interés, inflación
- **Variables lagged:** Hasta 2 años de retraso
- **Diferencias temporales:** Captura de dinámicas
- **Promedios móviles:** Suavizado de tendencias

### 🎯 **Validación y Robustez**
- **TimeSeriesSplit:** Validación temporal apropiada
- **Análisis de correlaciones:** Eliminación de variables redundantes
- **VIF (Variance Inflation Factor):** Detección de multicolinealidad
- **Análisis de autocorrelación:** Por país y variable
- **Test de robustez temporal:** Comparación entre períodos

### 📈 **Mejoras Implementadas**
- **Corrección de endogeneidad:** Variables absolutas vs ratios
- **Regularización fuerte:** Alpha Ridge = 50.0
- **Limpieza de datos:** 4,767 filas finales de calidad
- **Predicciones dinámicas:** Variabilidad temporal realista

## 📁 Estructura de Datos

### 📊 **Dataset Principal (global_merged_all.csv)**
- **10,512 filas** × **36 columnas**
- **Período:** 1970-2021
- **Países:** 133 países únicos
- **Variables:** GDP, población, comercio, inversión, inflación, etc.

### 📈 **Predicciones (predictions_2030.csv)**
- **1,197 filas** × **6 columnas**
- **Período:** 2022-2030
- **Países:** 133 países únicos
- **Variables:** País, Año, Crecimiento Predicho, Región, Nivel de Ingreso

## 🎛️ Dashboards Disponibles

### 1. 📈 **Dashboard de Predicciones 2030**
- **URL:** http://localhost:8506
- **Características:**
  - Series temporales interactivas
  - Mapas geográficos
  - Rankings por año
  - Análisis comparativo
  - Exportación de datos

### 2. 📊 **Dashboard de Equilibrio Macroeconómico**
- **URL:** http://localhost:8507
- **Análisis:** Consumo, inversión, gasto público, comercio

### 3. 🏭 **Dashboard de Estructura Productiva**
- **URL:** http://localhost:8508
- **Análisis:** Diversificación, HHI, sectores económicos

### 4. 🌍 **Dashboard de Conectividad Global**
- **URL:** http://localhost:8509
- **Análisis:** Petróleo, tasas de interés, inflación global

### 5. 💪 **Dashboard de Resiliencia Económica**
- **URL:** http://localhost:8510
- **Análisis:** Volatilidad, crisis, recuperación

## 📊 Métricas de Rendimiento

### 🎯 **Modelo Ridge (Mejor Rendimiento)**
| Métrica | Valor |
|---------|-------|
| **R² Training** | 0.874 |
| **R² Validation** | 0.899 |
| **RMSE** | 1.61 |
| **MAE** | 1.00 |
| **Overfitting** | ❌ No |

### 📈 **Comparación de Modelos**
| Modelo | R² Validation | RMSE | Estado |
|--------|---------------|------|--------|
| **Ridge** | **0.899** | 1.61 | 🥇 Excelente |
| **RandomForest** | 0.700 | 2.78 | 🥈 Muy Bueno |
| **GradientBoosting** | 0.285 | 4.30 | 🥉 Bueno |
| **ElasticNet** | 0.193 | 4.56 | ✅ Aceptable |
| **Lasso** | -0.006 | 5.09 | ⚠️ Pobre |

## 🔍 Análisis de Calidad de Datos

### 📊 **Limpieza de Datos**
- **Filas iniciales:** 10,512
- **Filas finales:** 4,767 (54% retenidas)
- **Variables eliminadas:** 44 por alta correlación
- **Variables finales:** 18 features optimizadas

### 🔗 **Análisis de Correlaciones**
- **69 pares** altamente correlacionados (≥95%)
- **Variables más importantes:**
  1. Gross_capital_formation_percent_GDP (13.2%)
  2. Population_main (5.7%)
  3. GNI_per_capita_main (4.4%)

### 📈 **Análisis de Autocorrelación**
- **180 países** analizados
- **87 países (48.3%)** con autocorrelación significativa
- **Países con mayor autocorrelación:** Azerbaijan (0.815), Afghanistan (0.639)

## 🚀 Comandos Rápidos

### 📊 **Generar Todo el Análisis**
```bash
# Opción 1: Lanzamiento completo (recomendado)
python launch_all.py

# Opción 2: Solo análisis
python launch_analysis.py

# Opción 3: Solo dashboards
python launch_dashboards.py
```

### 🎛️ **Lanzar Todos los Dashboards**
```bash
python launch_dashboards.py
```

### 📈 **Generar Reporte Completo**
```bash
python scripts/generate_complete_report.py
```

## 📚 Documentación Adicional

### 📄 **Archivos de Documentación**
- `DASHBOARD_README.md` - Documentación de dashboards
- `REGRESSION_MODULE_README.md` - Documentación de modelos
- `ENDOGENEITY_ANALYSIS.md` - Análisis de endogeneidad
- `ENHANCED_ML_ANALYSIS_SUMMARY.md` - Resumen de ML mejorado

### 🔧 **Scripts de Utilidad**
- `tests/test_temporal_robustness.py` - Test de robustez
- `examples/compare_regression_approaches.py` - Comparación de enfoques
- `examples/example_regression_analysis.py` - Ejemplo de uso

## 🎯 Casos de Uso

### 🏛️ **Para Políticos y Planificadores**
- Proyecciones de crecimiento económico
- Análisis de tendencias regionales
- Identificación de países de alto rendimiento
- Planificación de políticas económicas

### 🏢 **Para Analistas Económicos**
- Modelos de predicción avanzados
- Análisis de factores determinantes
- Comparación entre países
- Análisis de volatilidad y riesgo

### 🎓 **Para Investigadores**
- Dataset consolidado de múltiples fuentes
- Metodología de machine learning
- Análisis de endogeneidad
- Validación temporal robusta

### 📊 **Para Visualización de Datos**
- Dashboards interactivos
- Mapas geográficos
- Series temporales
- Análisis comparativo

## 🔮 Próximos Pasos

### 🚀 **Mejoras Planificadas**
1. **Ensemble de modelos** (Ridge + RandomForest)
2. **Análisis por regiones** (diferentes patrones)
3. **Variables de política** (tasas de interés, políticas fiscales)
4. **Análisis de residuos** por país
5. **Predicciones de intervalos** de confianza

### 📈 **Extensiones Posibles**
1. **Predicciones sectoriales** (agricultura, manufactura, servicios)
2. **Análisis de crisis** y recuperación
3. **Modelos de volatilidad** (GARCH)
4. **Análisis de causalidad** (Granger causality)
5. **Predicciones de inflación** y desempleo

## 🤝 Contribuciones

### 📝 **Cómo Contribuir**
1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### 🐛 **Reportar Issues**
- Usar el sistema de issues de GitHub
- Incluir descripción detallada del problema
- Proporcionar logs y datos de ejemplo
- Especificar versión de Python y dependencias

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- **Desarrollador Principal:** [Tu Nombre]
- **Análisis Económico:** [Tu Nombre]
- **Visualización de Datos:** [Tu Nombre]

## 📞 Contacto

- **Email:** [tu-email@ejemplo.com]
- **GitHub:** [tu-usuario-github]
- **LinkedIn:** [tu-perfil-linkedin]

## 🙏 Agradecimientos

- **World Bank** por los datos macroeconómicos
- **FRED (Federal Reserve Economic Data)** por los datos de tasas e inflación
- **Comunidad de Python** por las librerías de análisis de datos
- **Streamlit** por la plataforma de dashboards
- **Plotly** por las visualizaciones interactivas

---

## 📊 Resumen Ejecutivo

Este proyecto representa un sistema completo de análisis macroeconómico que combina:

- **📊 Datos consolidados** de múltiples fuentes internacionales
- **🤖 Modelos de ML avanzados** con validación temporal robusta
- **📈 Predicciones realistas** hasta 2030 para 133 países
- **🎛️ Dashboards interactivos** para visualización y análisis
- **🔧 Metodología rigurosa** con corrección de endogeneidad

**El modelo Ridge final logra un R² del 89.9% con predicciones económicamente creíbles y variabilidad temporal realista, proporcionando una herramienta valiosa para la planificación económica y el análisis de tendencias globales.**

---

*Última actualización: Diciembre 2024*
