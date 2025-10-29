"""
Script de Limpieza del Proyecto
Elimina archivos duplicados y organiza la estructura final
"""

import os
import shutil
from pathlib import Path

def cleanup_duplicates():
    """Elimina archivos duplicados y organiza la estructura."""
    print("LIMPIEZA DEL PROYECTO")
    print("=" * 50)
    
    # Archivos duplicados a eliminar del directorio raíz
    files_to_remove = [
        "data.csv",
        "merge_all_datasets.py", 
        "regression_analysis_report.png",
        "regression_approaches_comparison.png",
        "run_enhanced_ml_analysis.py",
        "test_enhanced_ml_simple.py",
        "ver nulos.py",
        ".env"
    ]
    
    # Carpetas duplicadas a eliminar
    folders_to_remove = [
        "enhanced_results",
        "results_corrected", 
        "test_results",
        "__pycache__"
    ]
    
    removed_files = 0
    removed_folders = 0
    
    # Eliminar archivos duplicados
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Eliminado: {file}")
                removed_files += 1
            except Exception as e:
                print(f"❌ Error eliminando {file}: {str(e)}")
    
    # Eliminar carpetas duplicadas
    for folder in folders_to_remove:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"Eliminada carpeta: {folder}")
                removed_folders += 1
            except Exception as e:
                print(f"❌ Error eliminando {folder}: {str(e)}")
    
    # Limpiar __pycache__ en subcarpetas
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"Eliminado __pycache__: {cache_path}")
                except Exception as e:
                    print(f"❌ Error eliminando {cache_path}: {str(e)}")
    
    print(f"\nRESUMEN DE LIMPIEZA:")
    print(f"  Archivos eliminados: {removed_files}")
    print(f"  Carpetas eliminadas: {removed_folders}")
    print("Limpieza completada")

def verify_structure():
    """Verifica que la estructura del proyecto esté correcta."""
    print(f"\nVERIFICACION DE ESTRUCTURA")
    print("=" * 50)
    
    required_dirs = [
        "data/external",
        "data/reference", 
        "src",
        "scripts",
        "dashboards",
        "results",
        "docs",
        "examples",
        "tests"
    ]
    
    required_files = [
        "config.py",
        "launch_all.py",
        "launch_analysis.py", 
        "launch_dashboards.py",
        "README.md",
        "requirements.txt"
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Verificar directorios
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"Directorio: {dir_path}")
    
    # Verificar archivos
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"Archivo: {file_path}")
    
    if missing_dirs:
        print(f"\n⚠️  Directorios faltantes: {missing_dirs}")
    
    if missing_files:
        print(f"\n⚠️  Archivos faltantes: {missing_files}")
    
    if not missing_dirs and not missing_files:
        print(f"\nESTRUCTURA DEL PROYECTO CORRECTA!")
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def main():
    """Función principal de limpieza."""
    print("LIMPIEZA Y ORGANIZACION DEL PROYECTO")
    print("=" * 60)
    
    # Limpiar duplicados
    cleanup_duplicates()
    
    # Verificar estructura
    structure_ok = verify_structure()
    
    print(f"\n{'='*60}")
    if structure_ok:
        print("PROYECTO COMPLETAMENTE ORGANIZADO!")
        print("\nComandos disponibles:")
        print("  python launch_all.py          # Lanzar todo el proyecto")
        print("  python launch_analysis.py     # Solo análisis")
        print("  python launch_dashboards.py   # Solo dashboards")
    else:
        print("⚠️  Revisa los elementos faltantes antes de continuar")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
