import os
import shutil

src_dir = "/home/rafael/Projects/python/IFG-Computer_vision/cowprojectifg/src"

folders = ["data", "features", "models", "utils"]
for folder in folders:
    os.makedirs(os.path.join(src_dir, folder), exist_ok=True)

moves = {
    "geometry.py": "utils/geometry.py",
    "geometry_const.py": "utils/geometry_const.py",
    "prepare_classification_dataset.py": "data/prepare_classification_dataset.py",
    "extract_geometric_features.py": "features/extract_geometric_features.py",
    "generate_geometric_features_from_dataset.py": "features/generate_geometric_features_from_dataset.py",
    "predict_keypoints_from_image.py": "models/predict_keypoints_from_image.py",
}

# Realizar movimentos
for filename, new_rel_path in moves.items():
    source_path = os.path.join(src_dir, filename)
    target_path = os.path.join(src_dir, new_rel_path)
    if os.path.exists(source_path):
        print(f"Movendo {filename} -> {new_rel_path}")
        shutil.move(source_path, target_path)

# Remover XGBoost classificador legado
xgb_path = os.path.join(src_dir, "train_xgboost_classifier.py")
if os.path.exists(xgb_path):
    print(f"Removendo {xgb_path}")
    os.remove(xgb_path)

# Atualizar as importações dentro dos arquivos modificados
def replace_in_file(filepath, old_str, new_str):
    if not os.path.exists(filepath): return
    with open(filepath, "r") as f: content = f.read()
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(filepath, "w") as f: f.write(content)
        print(f"Imports atualizados em {filepath}")

f_extract = os.path.join(src_dir, "features", "extract_geometric_features.py")
replace_in_file(f_extract, "from src.config.geometry import KEYPOINT_MAP", "from src.utils.geometry_const import KEYPOINT_MAP")
# Note: the keypoint_features module was missing, so we'll just fix its import to relative 'utils' domain where things currently belong
replace_in_file(f_extract, "from src.utils.keypoint_features import build_xgb_feature_dict", "from src.features.build_features import build_xgb_feature_dict")

f_generate = os.path.join(src_dir, "features", "generate_geometric_features_from_dataset.py")
replace_in_file(f_generate, "from src.classification.inference_pipeline import KEYPOINT_MAP, build_feature_dict", "from src.utils.geometry_const import KEYPOINT_MAP\nfrom src.features.build_features import build_feature_dict")

f_predict = os.path.join(src_dir, "models", "predict_keypoints_from_image.py")
replace_in_file(f_predict, "from src.classification.inference_pipeline import KEYPOINT_MAP", "from src.utils.geometry_const import KEYPOINT_MAP")

print("Refatoração Arquitetural Fase 3 concluída com sucesso!")
