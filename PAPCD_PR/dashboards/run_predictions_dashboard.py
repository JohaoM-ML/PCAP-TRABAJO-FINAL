"""
Script para lanzar el Dashboard de Predicciones 2030
"""

import subprocess
import sys
import os

def main():
    print("🚀 Iniciando Dashboard de Predicciones 2030...")
    print("📈 Predicciones de crecimiento económico hasta 2030")
    print("🌐 Abriendo en: http://localhost:8506")
    print("-" * 60)
    
    try:
        # Cambiar al directorio de dashboards
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Ejecutar Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "predictions_2030_dashboard.py",
            "--server.port", "8506",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard cerrado por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando dashboard: {str(e)}")

if __name__ == "__main__":
    main()

