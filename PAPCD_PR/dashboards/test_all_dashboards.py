"""
Script de prueba para ejecutar todos los dashboards individualmente
"""

import subprocess
import sys
import time

def test_dashboard(script_name, port, description):
    """Probar un dashboard individual"""
    try:
        print(f"🧪 Probando {description}...")
        print(f"📊 Puerto: {port}")
        print(f"🌐 URL: http://localhost:{port}")
        print("-" * 60)
        
        # Ejecutar streamlit con timeout
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            script_name,
            "--server.port", str(port),
            "--server.headless", "true"
        ])
        
        # Esperar un poco para que se inicie
        time.sleep(5)
        
        # Verificar si el proceso sigue corriendo
        if process.poll() is None:
            print(f"✅ {description} iniciado correctamente")
            process.terminate()
            return True
        else:
            print(f"❌ {description} falló al iniciar")
            return False
            
    except Exception as e:
        print(f"❌ Error probando {description}: {str(e)}")
        return False

def main():
    """Probar todos los dashboards"""
    print("=" * 80)
    print("🧪 PRUEBA DE TODOS LOS DASHBOARDS")
    print("=" * 80)
    print("📊 Probando cada dashboard individualmente...")
    print()
    
    # Definir dashboards
    dashboards = [
        {
            "script": "dashboards/macroeconomic_dashboard.py",
            "port": 8601,
            "description": "Dashboard Macroeconómico Principal"
        },
        {
            "script": "dashboards/productive_structure_dashboard.py", 
            "port": 8602,
            "description": "Estructura Productiva y Diversificación"
        },
        {
            "script": "dashboards/macroeconomic_balance_dashboard.py",
            "port": 8603,
            "description": "Equilibrio Macroeconómico y Demanda Agregada"
        },
        {
            "script": "dashboards/global_connectivity_dashboard.py",
            "port": 8604,
            "description": "Conectividad Global y Factores Externos"
        }
    ]
    
    results = []
    
    for dashboard in dashboards:
        result = test_dashboard(
            dashboard["script"], 
            dashboard["port"], 
            dashboard["description"]
        )
        results.append((dashboard["description"], result))
        time.sleep(2)  # Esperar entre pruebas
    
    # Mostrar resumen
    print("\n" + "=" * 80)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 80)
    
    for description, success in results:
        status = "✅ FUNCIONA" if success else "❌ FALLA"
        print(f"  {status} - {description}")
    
    # Contar éxitos
    successes = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n🎯 Resultado: {successes}/{total} dashboards funcionan correctamente")
    
    if successes == total:
        print("🎉 ¡Todos los dashboards están funcionando!")
    else:
        print("⚠️ Algunos dashboards necesitan corrección")

if __name__ == "__main__":
    main()


