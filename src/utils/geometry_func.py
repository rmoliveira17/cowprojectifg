import math
import numpy as np

KEYPOINT_MAP = {
    0: "withers",
    1: "back",
    2: "hook up",
    3: "hook down",
    4: "hip",
    5: "tail head",
    6: "pin up",
    7: "pin down",
}

POINT_CONNECTIONS = [
    ("withers", "back"),
    ("withers", "hook up"),
    ("withers", "hook down"),
    ("back", "hip"),
    ("back", "hook up"),
    ("back", "hook down"),
    ("hook up", "hook down"),
    ("hook up", "hip"),
    ("hook down", "hip"),
    ("hip", "tail head"),
    ("hook up", "tail head"),
    ("hook down", "tail head"),
    ("hook up", "pin up"),
    ("hook down", "pin down"),
    ("tail head", "pin up"),
    ("tail head", "pin down"),
    ("pin up", "pin down"),
]

ANGLE_TRIPLETS = [
    ("withers", "back", "hook up"),
    ("withers", "back", "hook down"),
    ("withers", "hook up", "hook down"),
    ("back", "hook up", "hook down"),
    ("back", "hook up", "hip"),
    ("back", "hook down", "hip"),
    ("hook up", "hook down", "hip"),
    ("hook up", "hook down", "tail head"),
    ("hook up", "tail head", "pin up"),
    ("hook down", "tail head", "pin down"),
    ("tail head", "pin up", "pin down"),
]

def slug(name: str) -> str:
    return name.replace(" ", "_")

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-12:
        return float("nan")

    cos_theta = float(np.dot(v1, v2) / denom)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(math.acos(cos_theta)))

def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return float(abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) / 2.0)
