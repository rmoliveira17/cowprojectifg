"""
Funções auxiliares para a aplicação Streamlit
"""
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from ui.config import (
    YOLO_MODEL_PATH,
    RF_MODEL_PATH,
    METRICS_PATH,
    CLASSES_CSV,
    ALL_FEATURES,
    KEYPOINT_NAMES,
    YOLO_CONFIG,
)


def load_yolo_model(model_path: Path):
    """Carrega modelo YOLO com cache"""
    try:
        if not model_path.exists():
            st.error(f"❌ Modelo YOLO não encontrado em: {model_path}")
            st.stop()
        return YOLO(str(model_path))
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo YOLO: {str(e)}")
        st.stop()


def load_rf_model(model_path: Path):
    """Carrega modelo Random Forest com cache"""
    try:
        import joblib
        
        if not model_path.exists():
            st.error(f"❌ Modelo Random Forest não encontrado em: {model_path}")
            st.stop()
        return joblib.load(str(model_path))
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo Random Forest: {str(e)}")
        st.stop()


def load_classes(classes_csv: Path) -> list:
    """Carrega lista de classes (IDs dos bovinos)"""
    try:
        if not classes_csv.exists():
            st.error(f"❌ Arquivo de classes não encontrado em: {classes_csv}")
            st.stop()
        
        df = pd.read_csv(classes_csv)
        # Assume que a primeira coluna contém os IDs das classes
        return sorted(df.iloc[:, 0].astype(str).tolist())
    except Exception as e:
        st.error(f"❌ Erro ao carregar classes: {str(e)}")
        st.stop()


def detect_keypoints(image: np.ndarray, yolo_model) -> Optional[np.ndarray]:
    """
    Detecta keypoints corporais da vaca usando YOLO
    
    Returns:
        np.ndarray: Array de shape (8, 2) com coordenadas x,y dos keypoints
        None: Se não conseguir detectar os keypoints
    """
    try:
        # Tenta diferentes configurações de tamanho e confiança
        for imgsz in YOLO_CONFIG["imgsz"]:
            for conf in YOLO_CONFIG["conf_threshold"]:
                results = yolo_model.predict(
                    source=image,
                    task="pose",
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                    device=YOLO_CONFIG["device"],
                )
                
                if results and results[0].keypoints is not None:
                    keypoints = results[0].keypoints.xy.cpu().numpy()
                    if len(keypoints) > 0 and len(keypoints[0]) == 8:
                        return keypoints[0]
        
        return None
    except Exception as e:
        st.error(f"❌ Erro ao detectar keypoints: {str(e)}")
        return None


def build_geometric_features(keypoints: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Constrói features geométricas a partir dos keypoints
    
    Args:
        keypoints: Array de shape (8, 2) com coordenadas dos keypoints
    
    Returns:
        Dict com as 28 features geométricas
    """
    try:
        # Importar a função build_feature_dict do projeto
        from features.build_features import build_feature_dict
        
        features = build_feature_dict(keypoints)
        return features
    except Exception as e:
        st.error(f"❌ Erro ao construir features: {str(e)}")
        return None


def predict_class(rf_model, features: Dict[str, float], classes: list) -> Tuple[str, float, Dict[str, float]]:
    """
    Faz predição de classe usando Random Forest
    
    Returns:
        Tuple:
            - class_id: ID da classe predita (str)
            - confidence: Confiança da predição (float 0-1)
            - proba_dict: Dict com probabilidades para todas as classes
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")
        
        # Preparar features no formato esperado pelo modelo
        features_df = pd.DataFrame([features])
        
        # Compatibilidade com dois formatos:
        # 1) modelo puro sklearn
        # 2) artefato dict com {"model": ..., "feature_cols": [...]}.
        model = rf_model
        selected_cols = [f for f in ALL_FEATURES if f in features_df.columns]
        if isinstance(rf_model, dict):
            model = rf_model.get("model")
            selected_cols = rf_model.get("feature_cols", selected_cols)
        
        features_df = features_df[[f for f in selected_cols if f in features_df.columns]]
        
        # Predição
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # Criar dicionário de probabilidades
        proba_dict = {str(class_id): float(prob) for class_id, prob in zip(classes, probabilities)}
        
        # Confiança é a probabilidade máxima
        confidence = float(np.max(probabilities))
        
        return str(prediction), confidence, proba_dict
    except Exception as e:
        st.error(f"❌ Erro ao fazer predição: {str(e)}")
        return None, 0.0, {}


def draw_keypoints_on_image(image: np.ndarray, keypoints: np.ndarray) -> Image.Image:
    """
    Desenha keypoints na imagem original
    
    Args:
        image: Imagem em formato OpenCV (BGR)
        keypoints: Array de shape (8, 2) com coordenadas dos keypoints
    
    Returns:
        PIL Image com keypoints desenhados
    """
    try:
        # Cópia para não modificar original
        img_copy = image.copy()
        
        # Desenhar círculos nos keypoints
        radius = 8
        thickness = 2
        
        for idx, (x, y) in enumerate(keypoints):
            x, y = int(x), int(y)
            # Cor diferente para cada keypoint
            color = (0, 255, 0) if idx % 2 == 0 else (255, 0, 0)
            cv2.circle(img_copy, (x, y), radius, color, thickness)
            
            # Adicionar número do keypoint
            cv2.putText(
                img_copy,
                str(idx),
                (x + radius + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        
        # Converter BGR para RGB e PIL
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    except Exception as e:
        st.error(f"❌ Erro ao desenhar keypoints: {str(e)}")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def load_model_metrics(metrics_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Carrega métricas do modelo treinado
    
    Returns:
        Dict com métricas (acurácia, precision, recall, etc.)
    """
    try:
        # Tenta carregar do arquivo JSON se existir
        resolved_metrics_path = metrics_path or METRICS_PATH
        if resolved_metrics_path.exists():
            with open(resolved_metrics_path, "r") as f:
                return json.load(f)
        
        # Caso contrário, retorna dict vazio (será preenchido após treinamento)
        return {
            "accuracy": 0.0,
            "classes_metrics": {},
            "total_classes": 30,
            "total_features": 28,
        }
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar métricas: {str(e)}")
        return {
            "accuracy": 0.0,
            "classes_metrics": {},
            "total_classes": 30,
            "total_features": 28,
        }


def get_top_predictions(proba_dict: Dict[str, float], top_k: int = 3) -> list:
    """
    Retorna as top-k predições ordenadas por confiança
    
    Args:
        proba_dict: Dict com probabilidades por classe
        top_k: Número de top predições a retornar
    
    Returns:
        Lista de tuplas (class_id, probability) ordenadas por probabilidade
    """
    sorted_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_proba[:top_k]


def format_percentage(value: float) -> str:
    """Formata valor como porcentagem"""
    return f"{value * 100:.2f}%"
