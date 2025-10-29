"""
Script para ejecutar el Dashboard de Conectividad Global
"""

import subprocess
import sys
import os

def run_connectivity_dashboard():
    """Ejecutar el dashboard de conectividad global"""
    try:
        print("🚀 Iniciando Dashboard de Conectividad Global...")
        print("🌐 Análisis de influencia del contexto internacional")
        print("🌐 Abriendo en: http://localhost:8504")
        print("-" * 60)
        
        # Ejecutar streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "global_connectivity_dashboard.py",
            "--server.port", "8504"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard detenido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando dashboard: {str(e)}")

if __name__ == "__main__":
    run_connectivity_dashboard()
