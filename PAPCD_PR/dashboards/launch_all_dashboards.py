"""
Script maestro para ejecutar todos los dashboards simultáneamente
"""

import subprocess
import sys
import os
import time
import threading

def run_dashboard(script_name, port, description):
    """Ejecutar un dashboard en un hilo separado"""
    try:
        print(f"🚀 Iniciando {description}...")
        print(f"📊 Puerto: {port}")
        print(f"🌐 URL: http://localhost:{port}")
        print("-" * 60)
        
        # Ejecutar streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            script_name,
            "--server.port", str(port)
        ])
        
    except Exception as e:
        print(f"❌ Error ejecutando {description}: {str(e)}")

def main():
    """Ejecutar todos los dashboards"""
    print("=" * 80)
    print("🚀 LANZADOR DE TODOS LOS DASHBOARDS")
    print("=" * 80)
    print("📊 Iniciando todos los dashboards de análisis macroeconómico...")
    print()
    
    # Definir dashboards
    dashboards = [
        {
            "script": "dashboards/macroeconomic_dashboard.py",
            "port": 8501,
            "description": "Dashboard Macroeconómico Principal"
        },
        {
            "script": "dashboards/productive_structure_dashboard.py", 
            "port": 8502,
            "description": "Estructura Productiva y Diversificación"
        },
        {
            "script": "dashboards/macroeconomic_balance_dashboard.py",
            "port": 8503,
            "description": "Equilibrio Macroeconómico y Demanda Agregada"
        },
        {
            "script": "dashboards/global_connectivity_dashboard.py",
            "port": 8504,
            "description": "Conectividad Global y Factores Externos"
        },
        {
            "script": "dashboards/economic_resilience_dashboard.py",
            "port": 8505,
            "description": "Resiliencia y Estabilidad Económica"
        }
    ]
    
    # Crear hilos para cada dashboard
    threads = []
    
    for dashboard in dashboards:
        thread = threading.Thread(
            target=run_dashboard,
            args=(dashboard["script"], dashboard["port"], dashboard["description"])
        )
        thread.daemon = True
        threads.append(thread)
        thread.start()
        time.sleep(2)  # Esperar 2 segundos entre lanzamientos
    
    print("\n✅ Todos los dashboards están iniciando...")
    print("\n📋 URLs disponibles:")
    for dashboard in dashboards:
        print(f"  • {dashboard['description']}: http://localhost:{dashboard['port']}")
    
    print("\n⏹️ Presiona Ctrl+C para detener todos los dashboards")
    
    try:
        # Mantener el script ejecutándose
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo todos los dashboards...")
        print("✅ Dashboards detenidos")

if __name__ == "__main__":
    main()
