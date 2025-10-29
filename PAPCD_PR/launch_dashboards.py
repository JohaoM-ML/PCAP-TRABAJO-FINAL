"""
Script de Lanzamiento de Dashboards
Lanza todos los dashboards del proyecto de análisis macroeconómico
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def launch_dashboard(dashboard_path, port, description):
    """Lanza un dashboard en un puerto específico."""
    print(f"🚀 Lanzando {description} en puerto {port}...")
    
    try:
        # Cambiar al directorio de dashboards
        os.chdir("dashboards")
        
        # Lanzar dashboard en segundo plano
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port", str(port),
            "--server.headless", "true"
        ])
        
        print(f"✅ {description} iniciado en http://localhost:{port}")
        return process
        
    except Exception as e:
        print(f"❌ Error lanzando {description}: {str(e)}")
        return None

def main():
    """Función principal para lanzar todos los dashboards."""
    print("🎛️ LANZADOR DE DASHBOARDS - ANÁLISIS MACROECONÓMICO")
    print("=" * 60)
    print("Este script lanzará todos los dashboards disponibles")
    print("=" * 60)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("dashboards"):
        print("❌ Error: No se encuentra la carpeta 'dashboards'")
        print("Asegúrate de estar en el directorio raíz del proyecto")
        return
    
    # Verificar que existen las predicciones
    if not os.path.exists("results/predictions_2030.csv"):
        print("⚠️  Advertencia: No se encuentran las predicciones")
        print("Ejecuta primero: python launch_analysis.py")
        print()
    
    # Dashboards disponibles
    dashboards = [
        ("predictions_2030_dashboard.py", 8506, "Dashboard de Predicciones 2030"),
        ("macroeconomic_balance_dashboard.py", 8507, "Dashboard de Equilibrio Macroeconómico"),
        ("productive_structure_dashboard.py", 8508, "Dashboard de Estructura Productiva"),
        ("global_connectivity_dashboard.py", 8509, "Dashboard de Conectividad Global"),
        ("economic_resilience_dashboard.py", 8510, "Dashboard de Resiliencia Económica")
    ]
    
    processes = []
    launched_count = 0
    
    print("🌐 URLs de los dashboards:")
    print("-" * 40)
    
    for dashboard_file, port, description in dashboards:
        dashboard_path = f"dashboards/{dashboard_file}"
        
        if os.path.exists(dashboard_path):
            process = launch_dashboard(dashboard_file, port, description)
            if process:
                processes.append(process)
                launched_count += 1
                print(f"  📊 {description}: http://localhost:{port}")
            time.sleep(2)  # Pausa entre lanzamientos
        else:
            print(f"⚠️  Dashboard no encontrado: {dashboard_file}")
    
    # Resumen
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE DASHBOARDS")
    print(f"{'='*60}")
    print(f"✅ Dashboards lanzados: {launched_count}/{len(dashboards)}")
    
    if launched_count > 0:
        print("\n🎉 ¡DASHBOARDS LANZADOS EXITOSAMENTE!")
        print("\n📋 Dashboards disponibles:")
        for dashboard_file, port, description in dashboards:
            if os.path.exists(f"dashboards/{dashboard_file}"):
                print(f"  🌐 {description}: http://localhost:{port}")
        
        print(f"\n💡 Para detener todos los dashboards:")
        print("  Presiona Ctrl+C en esta ventana")
        print(f"\n⏳ Los dashboards seguirán ejecutándose en segundo plano...")
        
        try:
            # Mantener el script ejecutándose
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print(f"\n🛑 Deteniendo dashboards...")
            for process in processes:
                try:
                    process.terminate()
                except:
                    pass
            print("✅ Dashboards detenidos")
    else:
        print("❌ No se pudo lanzar ningún dashboard")
        print("Verifica que los archivos de dashboard existan")

if __name__ == "__main__":
    main()

