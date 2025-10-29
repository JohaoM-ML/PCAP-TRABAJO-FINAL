"""
Test de robustez temporal para evaluar overfitting y generalización
 del modelo de machine learning macroeconómico.
"""

import sys
import os
sys.path.append('src')

from enhanced_macroeconomic_ml import EnhancedMacroeconomicML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def test_temporal_robustness():
    """Test de robustez temporal en diferentes períodos."""
    
    print("=" * 80)
    print("TEST DE ROBUSTEZ TEMPORAL - EVALUACIÓN DE OVERFITTING")
    print("=" * 80)
    
    # Definir diferentes períodos para comparar (acotado)
    time_periods = [
        {"name": "Período Reciente", "start": 2000, "end": 2021, "description": "Datos recientes (entrenamiento principal)"},
        {"name": "Período Anterior", "start": 1980, "end": 1999, "description": "Datos históricos (validación temporal)"}
    ]
    
    results = []
    
    for period in time_periods:
        print(f"\n{'='*60}")
        print(f"ANALIZANDO: {period['name']} ({period['start']}-{period['end']})")
        print(f"Descripción: {period['description']}")
        print(f"{'='*60}")
        
        try:
            # Inicializar modelo para este período
            ml_model = EnhancedMacroeconomicML()
            
            # Excluir GradientBoosting del conjunto de modelos
            if 'GradientBoosting' in ml_model.models:
                del ml_model.models['GradientBoosting']
            
            # Ejecutar análisis
            summary_df, vif_scores = ml_model.run_full_analysis(
                start_year=period['start'],
                end_year=period['end'],
                test_size=0.2,
                scaler_type='standard',
                lag_years=[1],
                include_structural=True,
                use_time_series_cv=True,
                cv_folds=3
            )
            
            # Extraer métricas del mejor modelo (RandomForest si existe, si no Ridge)
            preferred_model = 'RandomForest' if 'RandomForest' in ml_model.results else 'Ridge'
            if preferred_model in ml_model.results:
                res = ml_model.results[preferred_model]
                train_r2 = res['train_metrics']['r2']
                val_r2 = res['val_metrics']['r2']
                cv_r2_mean = res['cv_scores'].mean()
                cv_r2_std = res['cv_scores'].std()
                val_rmse = res['val_metrics']['rmse']
                
                # Calcular overfitting
                overfitting = train_r2 - val_r2
                overfitting_pct = (overfitting / train_r2) * 100 if train_r2 > 0 else 0
                
                # Calcular estabilidad CV
                cv_stability = cv_r2_std
                
                period_result = {
                    'Periodo': period['name'],
                    'Años': f"{period['start']}-{period['end']}",
                    'Muestras': len(ml_model.processed_data),
                    'Modelo': preferred_model,
                    'Train_R2': train_r2,
                    'Val_R2': val_r2,
                    'CV_R2_Mean': cv_r2_mean,
                    'CV_R2_Std': cv_r2_std,
                    'Val_RMSE': val_rmse,
                    'Overfitting': overfitting,
                    'Overfitting_Pct': overfitting_pct,
                    'CV_Stability': cv_stability,
                    'Generalization_Gap': train_r2 - cv_r2_mean
                }
                
                results.append(period_result)
                
                print(f"\nResultados para {period['name']} ({preferred_model}):")
                print(f"  Muestras: {len(ml_model.processed_data):,}")
                print(f"  Train R²: {train_r2:.4f}")
                print(f"  Val R²: {val_r2:.4f}")
                print(f"  CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
                print(f"  Overfitting: {overfitting:.4f} ({overfitting_pct:.1f}%)")
                print(f"  Estabilidad CV: {cv_stability:.4f}")
            else:
                print(f"  ❌ No se obtuvo resultado para el modelo preferido en {period['name']}")
                
        except Exception as e:
            print(f"  ❌ Error en {period['name']}: {str(e)}")
            continue
    
    # Crear resumen comparativo
    if results:
        print(f"\n{'='*80}")
        print("RESUMEN COMPARATIVO DE ROBUSTEZ TEMPORAL")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(results)
        
        # Mostrar tabla comparativa
        print("\nTabla Comparativa:")
        print(results_df[['Periodo', 'Años', 'Modelo', 'Muestras', 'Train_R2', 'Val_R2', 'CV_R2_Mean', 
                         'Overfitting_Pct', 'CV_Stability']].round(4))
        
        # Conclusión rápida
        if set(results_df['Periodo']) == {"Período Reciente", "Período Anterior"}:
            recent_r2 = results_df.loc[results_df['Periodo'] == 'Período Reciente', 'Val_R2'].iloc[0]
            hist_r2 = results_df.loc[results_df['Periodo'] == 'Período Anterior', 'Val_R2'].iloc[0]
            gap = recent_r2 - hist_r2
            print("\nComparación Reciente vs. Histórico:")
            print(f"  R² Reciente (2000-2021): {recent_r2:.4f}")
            print(f"  R² Histórico (1980-1999): {hist_r2:.4f}")
            print(f"  Gap Temporal: {gap:.4f}")
        
        return results_df
    else:
        print("❌ No se pudieron obtener resultados para ningún período")
        return None


def create_temporal_visualization(results_df):
    """Crear visualización de los resultados temporales."""
    # (Se mantiene igual)
    pass


def main():
    """Función principal para ejecutar el test de robustez temporal."""
    
    print("🧪 INICIANDO TEST DE ROBUSTEZ TEMPORAL")
    results_df = test_temporal_robustness()
    if results_df is not None:
        print("\n✅ TEST COMPLETADO")
    else:
        print("\n❌ TEST FALLÓ")
    return results_df

if __name__ == "__main__":
    results = main()
