"""
Configuración Central del Proyecto
Define rutas, parámetros y configuraciones globales
"""

import os
from pathlib import Path

# =========================
# RUTAS DEL PROYECTO
# =========================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.absolute()

# Rutas de datos
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

# Rutas de resultados
RESULTS_DIR = PROJECT_ROOT / "results"
ENHANCED_RESULTS_DIR = RESULTS_DIR / "enhanced_results_v3"

# Rutas de scripts
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DASHBOARDS_DIR = PROJECT_ROOT / "dashboards"
DOCS_DIR = PROJECT_ROOT / "docs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
TESTS_DIR = PROJECT_ROOT / "tests"

# =========================
# ARCHIVOS PRINCIPALES
# =========================

# Datos
MAIN_DATASET = EXTERNAL_DATA_DIR / "global_merged_all.csv"
WORLDBANK_DATA = EXTERNAL_DATA_DIR / "worldbank_data_clean.csv"
FRED_DATA = EXTERNAL_DATA_DIR / "fred_data.csv"
COUNTRIES_REFERENCE = REFERENCE_DATA_DIR / "worldbank_countries.csv"

# Resultados
PREDICTIONS_2030 = RESULTS_DIR / "predictions_2030.csv"
MODEL_PERFORMANCE = ENHANCED_RESULTS_DIR / "model_performance_v3.csv"
HIGH_CORRELATIONS = ENHANCED_RESULTS_DIR / "high_correlations.csv"
AUTOCORRELATION_ANALYSIS = ENHANCED_RESULTS_DIR / "autocorrelation_analysis.csv"

# =========================
# CONFIGURACIÓN DE MODELOS
# =========================

# Parámetros del modelo Ridge (mejor modelo)
RIDGE_ALPHA = 50.0
RIDGE_MAX_ITER = 2000

# Parámetros de Random Forest
RF_N_ESTIMATORS = 30
RF_MAX_DEPTH = 6
RF_MIN_SAMPLES_SPLIT = 10

# Parámetros de validación
TIME_SERIES_SPLITS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# CONFIGURACIÓN DE DASHBOARDS
# =========================

# Puertos de dashboards
DASHBOARD_PORTS = {
    "predictions": 8506,
    "macroeconomic_balance": 8507,
    "productive_structure": 8508,
    "global_connectivity": 8509,
    "economic_resilience": 8510
}

# URLs de dashboards
DASHBOARD_URLS = {
    name: f"http://localhost:{port}" 
    for name, port in DASHBOARD_PORTS.items()
}

# =========================
# CONFIGURACIÓN DE ANÁLISIS
# =========================

# Período de análisis
START_YEAR = 1970
END_YEAR = 2021
PREDICTION_START_YEAR = 2022
PREDICTION_END_YEAR = 2030

# Parámetros de limpieza de datos
MAX_CORRELATION_THRESHOLD = 0.95
VIF_THRESHOLD = 10.0
MIN_OBSERVATIONS_PER_COUNTRY = 5

# Parámetros de predicción
CONVERGENCE_FACTOR = 0.1
GLOBAL_MEAN_GROWTH = 3.0
MIN_GROWTH_LIMIT = -15.0
MAX_GROWTH_LIMIT = 25.0

# =========================
# CONFIGURACIÓN DE VISUALIZACIÓN
# =========================

# Colores para gráficos
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#d62728",
    "info": "#9467bd",
    "light": "#8c564b",
    "dark": "#e377c2"
}

# Configuración de Plotly
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"]
}

# =========================
# FUNCIONES DE UTILIDAD
# =========================

def get_data_path(filename: str) -> Path:
    """Obtiene la ruta completa de un archivo de datos."""
    return EXTERNAL_DATA_DIR / filename

def get_results_path(filename: str) -> Path:
    """Obtiene la ruta completa de un archivo de resultados."""
    return RESULTS_DIR / filename

def get_script_path(filename: str) -> Path:
    """Obtiene la ruta completa de un script."""
    return SCRIPTS_DIR / filename

def ensure_directories():
    """Asegura que todos los directorios necesarios existan."""
    directories = [
        DATA_DIR, EXTERNAL_DATA_DIR, REFERENCE_DATA_DIR,
        RESULTS_DIR, ENHANCED_RESULTS_DIR,
        SCRIPTS_DIR, DASHBOARDS_DIR, DOCS_DIR, 
        EXAMPLES_DIR, TESTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_dashboard_url(dashboard_name: str) -> str:
    """Obtiene la URL de un dashboard específico."""
    return DASHBOARD_URLS.get(dashboard_name, "http://localhost:8506")

# =========================
# INICIALIZACIÓN
# =========================

# Asegurar que los directorios existan
ensure_directories()

# =========================
# INFORMACIÓN DEL PROYECTO
# =========================

PROJECT_INFO = {
    "name": "Análisis Macroeconómico y Predicciones 2030",
    "version": "3.0",
    "description": "Sistema completo de análisis macroeconómico con ML y dashboards",
    "author": "Tu Nombre",
    "email": "tu-email@ejemplo.com",
    "github": "tu-usuario-github",
    "license": "MIT"
}

# =========================
# CONFIGURACIÓN DE LOGGING
# =========================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"]
}

# =========================
# CONFIGURACIÓN DE PERFORMANCE
# =========================

PERFORMANCE_CONFIG = {
    "max_workers": 4,
    "chunk_size": 1000,
    "memory_limit": "2GB",
    "timeout": 300
}

