"""
Script de Instalación de Dependencias
Instala todas las dependencias necesarias para el proyecto de análisis macroeconómico
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Ejecuta un comando y maneja errores."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - COMPLETADO")
            if result.stdout:
                print("📊 Salida:")
                print(result.stdout[-300:])  # Últimas 300 caracteres
        else:
            print(f"❌ {description} - ERROR")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando {description}: {str(e)}")
        return False
    
    return True

def check_python_version():
    """Verifica la versión de Python."""
    print("🐍 VERIFICANDO VERSIÓN DE PYTHON")
    print("=" * 50)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        print("Por favor, actualiza Python antes de continuar")
        return False
    
    print("✅ Versión de Python compatible")
    return True

def upgrade_pip():
    """Actualiza pip a la última versión."""
    return run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Actualizando pip"
    )

def install_requirements():
    """Instala las dependencias desde requirements.txt."""
    if not os.path.exists("requirements.txt"):
        print("❌ ERROR: No se encuentra requirements.txt")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Instalando dependencias desde requirements.txt"
    )

def install_optional_dependencies():
    """Instala dependencias opcionales que pueden fallar."""
    optional_packages = [
        "geopandas>=0.14.0",
        "folium>=0.15.0", 
        "great-expectations>=0.18.0",
        "numba>=0.58.0"
    ]
    
    print(f"\n📦 INSTALANDO DEPENDENCIAS OPCIONALES")
    print("=" * 50)
    
    for package in optional_packages:
        print(f"Instalando {package}...")
        success = run_command(
            f"{sys.executable} -m pip install {package}",
            f"Instalando {package}"
        )
        if not success:
            print(f"⚠️  Advertencia: No se pudo instalar {package}")
            print("El proyecto funcionará sin esta dependencia")

def verify_installation():
    """Verifica que las dependencias principales estén instaladas."""
    print(f"\n🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 50)
    
    critical_packages = [
        "pandas", "numpy", "plotly", "streamlit", 
        "scikit-learn", "requests", "matplotlib"
    ]
    
    missing_packages = []
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NO INSTALADO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Paquetes faltantes: {missing_packages}")
        return False
    
    print(f"\n🎉 ¡TODAS LAS DEPENDENCIAS CRÍTICAS INSTALADAS!")
    return True

def main():
    """Función principal de instalación."""
    print("🎯 INSTALADOR DE DEPENDENCIAS - ANÁLISIS MACROECONÓMICO")
    print("=" * 70)
    print("Este script instalará todas las dependencias necesarias")
    print("para el proyecto de análisis macroeconómico y predicciones 2030")
    print("=" * 70)
    
    # Verificar Python
    if not check_python_version():
        return
    
    # Actualizar pip
    if not upgrade_pip():
        print("⚠️  Advertencia: No se pudo actualizar pip")
    
    # Instalar dependencias principales
    if not install_requirements():
        print("❌ Error instalando dependencias principales")
        return
    
    # Instalar dependencias opcionales
    install_optional_dependencies()
    
    # Verificar instalación
    if verify_installation():
        print(f"\n{'='*70}")
        print("🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
        print("=" * 70)
        print("\n📋 Próximos pasos:")
        print("  1. python launch_all.py          # Lanzar todo el proyecto")
        print("  2. python launch_analysis.py     # Solo análisis")
        print("  3. python launch_dashboards.py   # Solo dashboards")
        print("\n🎛️ Dashboards disponibles:")
        print("  - Predicciones 2030: http://localhost:8506")
        print("  - Equilibrio Macro: http://localhost:8507")
        print("  - Estructura Productiva: http://localhost:8508")
        print("  - Conectividad Global: http://localhost:8509")
        print("  - Resiliencia Económica: http://localhost:8510")
    else:
        print(f"\n{'='*70}")
        print("⚠️  INSTALACIÓN COMPLETADA CON ADVERTENCIAS")
        print("Algunas dependencias opcionales no se pudieron instalar")
        print("El proyecto debería funcionar con las dependencias críticas")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

