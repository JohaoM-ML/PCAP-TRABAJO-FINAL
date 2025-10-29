# 🚀 Guía de Instalación - Análisis Macroeconómico y Predicciones 2030

## 📋 **RESUMEN DE INSTALACIÓN**

### **✅ Requirements.txt Actualizado**
El archivo `requirements.txt` ha sido completamente actualizado con todas las dependencias necesarias para el proyecto completo.

---

## 🎯 **COMANDOS DE INSTALACIÓN**

### **1. 🚀 INSTALACIÓN AUTOMÁTICA (RECOMENDADA)**

#### **Windows:**
```bash
# Ejecutar el script de instalación automática
install.bat
```

#### **Linux/Mac:**
```bash
# Hacer ejecutable y ejecutar
chmod +x install.sh
./install.sh
```

#### **Multiplataforma (Python):**
```bash
# Usar el script de instalación en Python
python install_dependencies.py
```

---

### **2. 🔧 INSTALACIÓN MANUAL**

#### **Paso 1: Actualizar pip**
```bash
python -m pip install --upgrade pip
```

#### **Paso 2: Instalar dependencias principales**
```bash
pip install -r requirements.txt
```

#### **Paso 3: Instalar dependencias opcionales**
```bash
pip install geopandas folium great-expectations numba
```

#### **Paso 4: Verificar instalación**
```bash
python -c "import pandas, numpy, plotly, streamlit, sklearn; print('Instalación exitosa')"
```

---

## 📦 **DEPENDENCIAS INCLUIDAS**

### **🔧 Análisis de Datos**
- `pandas>=2.2.2` - Manipulación de datos
- `numpy>=1.26.4` - Computación numérica

### **📊 Visualización y Dashboards**
- `plotly>=5.24.1` - Gráficos interactivos
- `streamlit>=1.38.0` - Dashboards web
- `matplotlib>=3.9.2` - Gráficos estáticos
- `seaborn>=0.13.2` - Visualización estadística

### **🤖 Machine Learning**
- `scikit-learn>=1.5.2` - Algoritmos de ML
- `xgboost>=2.1.1` - Gradient boosting
- `shap>=0.46.0` - Interpretabilidad de modelos

### **📈 Análisis Estadístico**
- `statsmodels>=0.14.2` - Modelos estadísticos
- `scipy>=1.9.0` - Computación científica

### **🌐 APIs y Datos Externos**
- `requests>=2.32.3` - Peticiones HTTP
- `fredapi>=0.5.2` - API de FRED
- `wbdata>=0.3.0` - API del Banco Mundial
- `wbgapi>=1.0.12` - API moderna del Banco Mundial

### **🗺️ Geocoding y Mapas**
- `geopandas>=0.14.0` - Análisis geoespacial
- `folium>=0.15.0` - Mapas interactivos

### **⚡ Optimización**
- `numba>=0.58.0` - Aceleración numérica

### **🔍 Validación de Datos**
- `great-expectations>=0.18.0` - Validación de calidad

### **📝 Utilidades**
- `tqdm>=4.66.5` - Barras de progreso
- `openpyxl>=3.1.5` - Archivos Excel
- `pyyaml>=6.0.1` - Configuración YAML
- `loguru>=0.7.0` - Logging avanzado

---

## 🎯 **COMANDO RÁPIDO DE INSTALACIÓN**

### **Para Windows:**
```bash
install.bat
```

### **Para Linux/Mac:**
```bash
chmod +x install.sh && ./install.sh
```

### **Para cualquier sistema:**
```bash
python install_dependencies.py
```

---

## ✅ **VERIFICACIÓN POST-INSTALACIÓN**

### **1. Verificar Dependencias Críticas**
```bash
python -c "
import pandas as pd
import numpy as np
import plotly
import streamlit
import sklearn
import requests
import matplotlib
import seaborn
print('✅ Todas las dependencias críticas instaladas correctamente')
"
```

### **2. Verificar Dependencias Opcionales**
```bash
python -c "
try:
    import geopandas
    print('✅ GeoPandas instalado')
except ImportError:
    print('⚠️  GeoPandas no instalado (opcional)')

try:
    import folium
    print('✅ Folium instalado')
except ImportError:
    print('⚠️  Folium no instalado (opcional)')

try:
    import xgboost
    print('✅ XGBoost instalado')
except ImportError:
    print('⚠️  XGBoost no instalado (opcional)')
"
```

### **3. Probar el Proyecto**
```bash
# Lanzar análisis completo
python launch_all.py
```

---

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### **Error: "No module named 'pandas'"**
```bash
# Reinstalar pandas
pip install --upgrade pandas
```

### **Error: "No module named 'streamlit'"**
```bash
# Reinstalar streamlit
pip install --upgrade streamlit
```

### **Error: "No module named 'sklearn'"**
```bash
# Reinstalar scikit-learn
pip install --upgrade scikit-learn
```

### **Error de permisos en Windows**
```bash
# Ejecutar como administrador o usar --user
pip install --user -r requirements.txt
```

### **Error de permisos en Linux/Mac**
```bash
# Usar sudo si es necesario
sudo pip install -r requirements.txt
```

---

## 🎉 **DESPUÉS DE LA INSTALACIÓN**

### **Comandos Disponibles:**
```bash
# Lanzar todo el proyecto
python launch_all.py

# Solo análisis
python launch_analysis.py

# Solo dashboards
python launch_dashboards.py
```

### **Dashboards Disponibles:**
- **Predicciones 2030:** http://localhost:8506
- **Equilibrio Macro:** http://localhost:8507
- **Estructura Productiva:** http://localhost:8508
- **Conectividad Global:** http://localhost:8509
- **Resiliencia Económica:** http://localhost:8510

---

## 📞 **SOPORTE**

Si encuentras problemas durante la instalación:

1. **Verifica la versión de Python:** `python --version` (debe ser 3.8+)
2. **Actualiza pip:** `python -m pip install --upgrade pip`
3. **Reinstala dependencias:** `pip install --force-reinstall -r requirements.txt`
4. **Usa entornos virtuales:** `python -m venv venv && venv\Scripts\activate`

---

**¡Instalación completada! El proyecto está listo para usar.** 🎉

