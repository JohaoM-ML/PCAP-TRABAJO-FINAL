"""
Generador de Reporte Completo de Predicciones 2030
Muestra TODOS los países con sus predicciones detalladas
"""

import pandas as pd
import numpy as np

def generate_complete_report():
    """Genera reporte completo de todas las predicciones."""
    print("=== PREDICCIONES COMPLETAS DE TODOS LOS PAISES ===")
    print("Modelo: Ridge Regression | R² = 0.899 | RMSE = 1.61")
    print("Período: 2022-2030 | Total: 133 países")
    print("=" * 80)
    
    # Cargar datos
    df = pd.read_csv('../results/predictions_2030.csv')
    countries = sorted(df['Country'].unique())
    
    print(f"\nTotal de países: {len(countries)}")
    print(f"Total de predicciones: {len(df):,}")
    print()
    
    # Estadísticas generales
    print("ESTADÍSTICAS GLOBALES:")
    print(f"  Crecimiento promedio: {df['GDP_growth_predicted'].mean():.2f}%")
    print(f"  Crecimiento mediano: {df['GDP_growth_predicted'].median():.2f}%")
    print(f"  Rango: {df['GDP_growth_predicted'].min():.2f}% - {df['GDP_growth_predicted'].max():.2f}%")
    print(f"  Desviación estándar: {df['GDP_growth_predicted'].std():.2f}%")
    print()
    
    # Análisis por año
    print("ANÁLISIS POR AÑO:")
    year_stats = df.groupby('Year')['GDP_growth_predicted'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    for year, stats in year_stats.iterrows():
        print(f"  {year}: Promedio={stats['mean']:.2f}% | Mediano={stats['median']:.2f}% | Rango={stats['min']:.2f}% a {stats['max']:.2f}%")
    print()
    
    # Predicciones por país
    print("PREDICCIONES DETALLADAS POR PAÍS:")
    print("=" * 80)
    
    for i, country in enumerate(countries, 1):
        country_data = df[df['Country'] == country].sort_values('Year')
        
        print(f"\n{i:3d}. {country.upper()}")
        print("    " + "-" * 50)
        
        # Mostrar predicciones por año
        for _, row in country_data.iterrows():
            year = int(row['Year'])
            growth = row['GDP_growth_predicted']
            print(f"    {year}: {growth:6.2f}%")
        
        # Estadísticas del país
        avg_growth = country_data['GDP_growth_predicted'].mean()
        trend = "CRECIENTE" if country_data['GDP_growth_predicted'].iloc[-1] > country_data['GDP_growth_predicted'].iloc[0] else "DECRECIENTE"
        volatility = country_data['GDP_growth_predicted'].std()
        
        print(f"    Promedio: {avg_growth:.2f}% | Tendencia: {trend} | Volatilidad: {volatility:.2f}%")
    
    print("\n" + "=" * 80)
    print("REPORTE COMPLETADO")
    print("=" * 80)

if __name__ == "__main__":
    generate_complete_report()
