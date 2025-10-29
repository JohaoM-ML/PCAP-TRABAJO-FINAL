#!/bin/bash

echo "========================================"
echo "INSTALADOR DE DEPENDENCIAS"
echo "Análisis Macroeconómico y Predicciones 2030"
echo "========================================"
echo

# Verificar Python
echo "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 no está instalado"
    echo "Por favor, instala Python 3.8 o superior"
    exit 1
fi

python3 --version

# Actualizar pip
echo
echo "Actualizando pip..."
python3 -m pip install --upgrade pip

# Instalar dependencias principales
echo
echo "Instalando dependencias principales..."
python3 -m pip install -r requirements.txt

# Instalar dependencias opcionales
echo
echo "Instalando dependencias opcionales..."
python3 -m pip install geopandas folium great-expectations numba

# Verificar instalación
echo
echo "Verificando instalación..."
python3 -c "import pandas, numpy, plotly, streamlit, sklearn; print('Todas las dependencias críticas instaladas correctamente')"

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo "INSTALACION COMPLETADA EXITOSAMENTE!"
    echo "========================================"
    echo
    echo "Comandos disponibles:"
    echo "  python3 launch_all.py          # Lanzar todo el proyecto"
    echo "  python3 launch_analysis.py     # Solo análisis"
    echo "  python3 launch_dashboards.py   # Solo dashboards"
    echo
    echo "Dashboards disponibles:"
    echo "  - Predicciones 2030: http://localhost:8506"
    echo "  - Equilibrio Macro: http://localhost:8507"
    echo "  - Estructura Productiva: http://localhost:8508"
    echo "  - Conectividad Global: http://localhost:8509"
    echo "  - Resiliencia Económica: http://localhost:8510"
else
    echo
    echo "ERROR: Algunas dependencias no se instalaron correctamente"
    echo "Revisa los mensajes de error arriba"
    exit 1
fi

echo

