@echo off
echo ========================================
echo INSTALADOR DE DEPENDENCIAS
echo Análisis Macroeconómico y Predicciones 2030
echo ========================================
echo.

echo Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python no está instalado o no está en el PATH
    echo Por favor, instala Python 3.8 o superior
    pause
    exit /b 1
)

echo.
echo Actualizando pip...
python -m pip install --upgrade pip

echo.
echo Instalando dependencias principales...
python -m pip install -r requirements.txt

echo.
echo Instalando dependencias opcionales...
python -m pip install geopandas folium great-expectations numba

echo.
echo Verificando instalación...
python -c "import pandas, numpy, plotly, streamlit, sklearn; print('Todas las dependencias críticas instaladas correctamente')"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo INSTALACION COMPLETADA EXITOSAMENTE!
    echo ========================================
    echo.
    echo Comandos disponibles:
    echo   python launch_all.py          # Lanzar todo el proyecto
    echo   python launch_analysis.py     # Solo análisis
    echo   python launch_dashboards.py   # Solo dashboards
    echo.
    echo Dashboards disponibles:
    echo   - Predicciones 2030: http://localhost:8506
    echo   - Equilibrio Macro: http://localhost:8507
    echo   - Estructura Productiva: http://localhost:8508
    echo   - Conectividad Global: http://localhost:8509
    echo   - Resiliencia Económica: http://localhost:8510
) else (
    echo.
    echo ERROR: Algunas dependencias no se instalaron correctamente
    echo Revisa los mensajes de error arriba
)

echo.
pause

