import json
from pathlib import Path
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# Carregar dados e modelo
csv_path = Path("src/data/datasets/classifications/geometric_features.csv")
model_path = Path("src/models/random_forest/best_rf_model.pkl")
output_path = Path("src/models/random_forest/metrics.json")

df = pd.read_csv(csv_path)
artifact_or_model = joblib.load(model_path)

# Preparar dados
feature_cols = [c for c in df.columns if c.startswith('dist_') or c.startswith('angle_')]
X = df[feature_cols]
y = df['class_name']

model = artifact_or_model
selected_feature_cols = feature_cols
if isinstance(artifact_or_model, dict):
    model = artifact_or_model.get("model")
    selected_feature_cols = artifact_or_model.get("feature_cols", feature_cols)
    X = X[selected_feature_cols]

# Fazer predições
y_pred = model.predict(X)

# Gerar relatório
report = classification_report(y.astype(str).values, y_pred.astype(str), output_dict=True)

# Extrair informações principais
metrics = {
    "accuracy": report["accuracy"],
    "total_classes": y.nunique(),
    "total_features": len(selected_feature_cols),
    "total_samples": len(df),
    "selected_features": selected_feature_cols,
    "classes_metrics": {}
}

# Adicionar métricas por classe
for class_id in sorted(y.unique()):
    class_id_str = str(class_id)
    if class_id_str in report:
        metrics["classes_metrics"][class_id_str] = {
            "precision": report[class_id_str]["precision"],
            "recall": report[class_id_str]["recall"],
            "f1-score": report[class_id_str]["f1-score"],
            "support": int(report[class_id_str]["support"])
        }

# Salvar arquivo JSON
with open(output_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métricas salvas em {output_path}")
print(f"   - Acurácia: {metrics['accuracy']:.4f}")
print(f"   - Classes: {metrics['total_classes']}")
print(f"   - Features: {metrics['total_features']}")
