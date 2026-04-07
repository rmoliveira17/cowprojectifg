"""
Configurações da aplicação Streamlit para predição de classe bovina
"""
import os
from pathlib import Path

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
FEATURES_DIR = PROJECT_ROOT / "src" / "features"

# Caminhos dos modelos
YOLO_MODEL_PATH = MODELS_DIR / "yolo" / "best.pt"
RF_MODEL_PATH = MODELS_DIR / "random_forest" / "best_rf_model.pkl"
METRICS_PATH = MODELS_DIR / "random_forest" / "metrics.json"

# Caminhos dos dados
CLASSES_CSV = DATA_DIR / "datasets" / "classifications" / "classes.csv"
FEATURES_CSV = DATA_DIR / "datasets" / "classifications" / "geometric_features.csv"
GEOMETRIC_REPORT_JSON = DATA_DIR / "datasets" / "classifications" / "geometric_features_report.json"

# Configurações de detecção YOLO
YOLO_CONFIG = {
    "imgsz": [640, 960, 1280],
    "conf_threshold": [0.25, 0.15, 0.08, 0.05],
    "device": "cpu",
}

# Features esperadas
DISTANCE_FEATURES = [
    "dist_withers_back",
    "dist_withers_hook_up",
    "dist_withers_hook_down",
    "dist_back_hip",
    "dist_back_hook_up",
    "dist_back_hook_down",
    "dist_hook_up_hook_down",
    "dist_hook_up_hip",
    "dist_hook_down_hip",
    "dist_hip_tail_head",
    "dist_hook_up_tail_head",
    "dist_hook_down_tail_head",
    "dist_hook_up_pin_up",
    "dist_hook_down_pin_down",
    "dist_tail_head_pin_up",
    "dist_tail_head_pin_down",
    "dist_pin_up_pin_down",
]

ANGLE_FEATURES = [
    "angle_withers_back_hook_up",
    "angle_withers_back_hook_down",
    "angle_withers_hook_up_hook_down",
    "angle_back_hook_up_hook_down",
    "angle_back_hook_up_hip",
    "angle_back_hook_down_hip",
    "angle_hook_up_hook_down_hip",
    "angle_hook_up_hook_down_tail_head",
    "angle_hook_up_tail_head_pin_up",
    "angle_hook_down_tail_head_pin_down",
    "angle_tail_head_pin_up_pin_down",
]

ALL_FEATURES = DISTANCE_FEATURES + ANGLE_FEATURES

# Keypoints corporais da vaca
KEYPOINT_NAMES = {
    0: "Cernelha (Withers)",
    1: "Costas (Back)",
    2: "Gancho Superior (Hook Up)",
    3: "Gancho Inferior (Hook Down)",
    4: "Anca (Hip)",
    5: "Base da Cauda (Tail Head)",
    6: "Pino Superior (Pin Up)",
    7: "Pino Inferior (Pin Down)",
}

# Cores e estilos
COLORS = {
    "primary": "#2ECC71",  # Verde
    "secondary": "#3498DB",  # Azul
    "accent": "#E74C3C",  # Vermelho
    "background": "#1E1E1E",  # Cinza escuro
    "text": "#ECF0F1",  # Cinza claro
    "success": "#27AE60",  # Verde escuro
    "warning": "#F39C12",  # Laranja
}

# Configurações de UI
UI_CONFIG = {
    "page_title": "🐄 Classificação Bovina - AI Predictor",
    "page_icon": "🐄",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Cache de modelos
CACHE_TTL = 3600  # 1 hora em segundos
