"""
Script para ejecutar el Dashboard de Equilibrio Macroeconómico
"""

import subprocess
import sys
import os

def run_balance_dashboard():
    """Ejecutar el dashboard de equilibrio macroeconómico"""
    try:
        print("🚀 Iniciando Dashboard de Equilibrio Macroeconómico...")
        print("📊 Análisis de composición y estabilidad del gasto nacional")
        print("🌐 Abriendo en: http://localhost:8503")
        print("-" * 60)
        
        # Ejecutar streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "macroeconomic_balance_dashboard.py",
            "--server.port", "8503"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard detenido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando dashboard: {str(e)}")

if __name__ == "__main__":
    run_balance_dashboard()
