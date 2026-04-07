import numpy as np
from src.utils.geometry_func import (
    calculate_distance, 
    calculate_angle, 
    KEYPOINT_MAP, 
    POINT_CONNECTIONS, 
    ANGLE_TRIPLETS
)

def build_feature_dict(points: np.ndarray) -> dict:
    features = {}
    
    # Inverter KEYPOINT_MAP para dicionário de name -> index
    name_to_idx = {name: idx for idx, name in KEYPOINT_MAP.items()}
    
    # Calcula as distâncias entre conexões predefinidas
    for p1_name, p2_name in POINT_CONNECTIONS:
        if p1_name in name_to_idx and p2_name in name_to_idx:
            idx1 = name_to_idx[p1_name]
            idx2 = name_to_idx[p2_name]
            if len(points) > max(idx1, idx2):
                p1 = points[idx1]
                p2 = points[idx2]
                dist = calculate_distance(p1, p2)
                features[f"dist_{p1_name}_{p2_name}".replace(" ", "_")] = dist

    # Calcula os ângulos entre tripletos predefinidos
    for p1_name, p2_name, p3_name in ANGLE_TRIPLETS:
        if p1_name in name_to_idx and p2_name in name_to_idx and p3_name in name_to_idx:
            idx1 = name_to_idx[p1_name]
            idx2 = name_to_idx[p2_name]
            idx3 = name_to_idx[p3_name]
            if len(points) > max(idx1, idx2, idx3):
                p1 = points[idx1]
                p2 = points[idx2] # Vértice do ângulo
                p3 = points[idx3]
                angle = calculate_angle(p1, p2, p3)
                features[f"angle_{p1_name}_{p2_name}_{p3_name}".replace(" ", "_")] = angle
                
    return features

def build_xgb_feature_dict(points: np.ndarray) -> dict:
    """Suporte retroativo às referências legadas para extração de ML."""
    return build_feature_dict(points)
