"""
Script para ejecutar el Dashboard de Resiliencia y Estabilidad Económica
"""

import subprocess
import sys
import os

def run_resilience_dashboard():
    """Ejecutar el dashboard de resiliencia económica"""
    try:
        print("🚀 Iniciando Dashboard de Resiliencia y Estabilidad Económica...")
        print("🧠 Análisis de capacidad de resistencia y recuperación de crisis")
        print("🌐 Abriendo en: http://localhost:8505")
        print("-" * 60)
        
        # Ejecutar streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "economic_resilience_dashboard.py",
            "--server.port", "8505"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard detenido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando dashboard: {str(e)}")

if __name__ == "__main__":
    run_resilience_dashboard()


