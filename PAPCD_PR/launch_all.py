"""
Script Maestro - Lanzamiento Completo del Proyecto
Ejecuta todo el pipeline: análisis + dashboards
"""

import os
import sys
import subprocess
import time

def main():
    """Función principal que ejecuta todo el proyecto."""
    print("🎯 PROYECTO DE ANÁLISIS MACROECONÓMICO - LANZAMIENTO COMPLETO")
    print("=" * 70)
    print("Este script ejecutará:")
    print("  1. 📊 Análisis ML y predicciones")
    print("  2. 🎛️ Lanzamiento de todos los dashboards")
    print("=" * 70)
    
    # Verificar directorio
    if not os.path.exists("data/external/global_merged_all.csv"):
        print("❌ Error: No se encuentra el dataset principal")
        print("Asegúrate de estar en el directorio raíz del proyecto")
        return
    
    # Paso 1: Ejecutar análisis
    print("\n🚀 PASO 1: EJECUTANDO ANÁLISIS COMPLETO")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, "launch_analysis.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Análisis completado exitosamente")
        else:
            print("❌ Error en el análisis:")
            print(result.stderr)
            return
            
    except Exception as e:
        print(f"❌ Error ejecutando análisis: {str(e)}")
        return
    
    # Pausa entre pasos
    print("\n⏳ Esperando 5 segundos antes de lanzar dashboards...")
    time.sleep(5)
    
    # Paso 2: Lanzar dashboards
    print("\n🎛️ PASO 2: LANZANDO DASHBOARDS")
    print("-" * 50)
    
    try:
        # Lanzar dashboards en segundo plano
        dashboard_process = subprocess.Popen([sys.executable, "launch_dashboards.py"])
        
        print("✅ Dashboards lanzándose...")
        print("\n📋 URLs disponibles:")
        print("  🌐 Predicciones 2030: http://localhost:8506")
        print("  📊 Equilibrio Macro: http://localhost:8507")
        print("  🏭 Estructura Productiva: http://localhost:8508")
        print("  🌍 Conectividad Global: http://localhost:8509")
        print("  💪 Resiliencia Económica: http://localhost:8510")
        
        print(f"\n🎉 ¡PROYECTO COMPLETAMENTE LANZADO!")
        print("=" * 70)
        print("💡 Para detener todo: Presiona Ctrl+C")
        print("⏳ El sistema seguirá ejecutándose...")
        
        # Mantener ejecutándose
        try:
            dashboard_process.wait()
        except KeyboardInterrupt:
            print(f"\n🛑 Deteniendo proyecto...")
            dashboard_process.terminate()
            print("✅ Proyecto detenido")
            
    except Exception as e:
        print(f"❌ Error lanzando dashboards: {str(e)}")

if __name__ == "__main__":
    main()

