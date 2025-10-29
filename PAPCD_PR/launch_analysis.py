"""
Script Principal de Lanzamiento - Análisis Macroeconómico
Ejecuta todo el pipeline de análisis desde datos hasta predicciones
"""

import os
import sys
import subprocess
from pathlib import Path

def run_script(script_path, description):
    """Ejecuta un script y maneja errores."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {description} - COMPLETADO")
            if result.stdout:
                print("📊 Salida:")
                print(result.stdout[-500:])  # Últimas 500 caracteres
        else:
            print(f"❌ {description} - ERROR")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando {description}: {str(e)}")
        return False
    
    return True

def main():
    """Función principal del pipeline de análisis."""
    print("🎯 PIPELINE DE ANÁLISIS MACROECONÓMICO")
    print("=" * 60)
    print("Este script ejecutará todo el análisis desde datos hasta predicciones")
    print("Incluye: ML mejorado, predicciones 2030, y reporte completo")
    print("=" * 60)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("data/external/global_merged_all.csv"):
        print("❌ Error: No se encuentra el dataset principal")
        print("Asegúrate de estar en el directorio raíz del proyecto")
        return
    
    # Pipeline de análisis
    steps = [
        ("scripts/run_enhanced_ml_analysis_v3.py", "Análisis ML Mejorado (V3)"),
        ("scripts/generate_predictions_2030.py", "Generación de Predicciones 2030"),
        ("scripts/generate_complete_report.py", "Generación de Reporte Completo")
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for script_path, description in steps:
        if os.path.exists(script_path):
            if run_script(script_path, description):
                success_count += 1
        else:
            print(f"⚠️  Script no encontrado: {script_path}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DEL PIPELINE")
    print(f"{'='*60}")
    print(f"✅ Pasos completados: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print("\n📁 Archivos generados:")
        print("  - results/enhanced_results_v3/ (análisis ML)")
        print("  - results/predictions_2030.csv (predicciones)")
        print("  - Reporte completo en consola")
        print("\n🎛️ Para lanzar dashboards:")
        print("  python launch_dashboards.py")
    else:
        print("⚠️  Algunos pasos fallaron. Revisa los errores arriba.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

