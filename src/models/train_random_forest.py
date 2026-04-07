import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser("Aplica Random Forest nas Features Geométricas Tabulares")
    parser.add_argument("--features-csv", default="src/data/datasets/classifications/geometric_features.csv", help="Dataset CSV")
    parser.add_argument("--models-dir", default="src/models/random_forest", help="Local para salvar o modelo em .pkl e gráficos")
    parser.add_argument(
        "--topk-candidates",
        default="12,16,20,24,28",
        help="Lista de quantidades de features para testar (ex: 12,16,20,24,28)",
    )
    return parser.parse_args()


def parse_topk_candidates(raw_value: str, max_features: int) -> list[int]:
    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if 1 <= value <= max_features:
            values.append(value)

    if max_features not in values:
        values.append(max_features)

    return sorted(set(values))


def evaluate_candidate(X: pd.DataFrame, y: pd.Series, selected_cols: list[str], rf_params: dict) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc = []
    oof_preds = pd.Series(index=y.index, dtype="object")

    for train_idx, val_idx in skf.split(X[selected_cols], y):
        X_train, y_train = X.iloc[train_idx][selected_cols], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx][selected_cols], y.iloc[val_idx]

        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        oof_preds.iloc[val_idx] = preds
        fold_acc.append(accuracy_score(y_val, preds))

    y_true = y.astype(str).values
    y_pred = oof_preds.astype(str).values
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    return {
        "mean_acc": float(sum(fold_acc) / len(fold_acc)),
        "precision_macro": float(report_dict["macro avg"]["precision"]),
        "recall_macro": float(report_dict["macro avg"]["recall"]),
        "f1_macro": float(report_dict["macro avg"]["f1-score"]),
        "oof_preds": y_pred,
        "report_dict": report_dict,
    }


def main():
    args = parse_args()
    csv_path = Path(args.features_csv)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}\nVocê rodou a geração de features antes?")

    df = pd.read_csv(csv_path)

    feature_cols = [col for col in df.columns if col.startswith("dist_") or col.startswith("angle_")]
    if not feature_cols:
        raise RuntimeError("Nenhuma feature encontrada no CSV (esperado: dist_* e angle_*).")

    X = df[feature_cols]
    y = df["class_name"].astype(str)

    print(
        f"Iniciando Treinamento com {len(feature_cols)} atributos sobre {len(y)} fotos distribuídas em {y.nunique()} identificadores distintos."
    )

    topk_candidates = parse_topk_candidates(args.topk_candidates, max_features=len(feature_cols))

    ranking_model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        max_features="sqrt",
        n_jobs=-1,
    )
    ranking_model.fit(X, y)
    ranked_features = pd.Series(ranking_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    param_candidates = [
        {
            "n_estimators": 300,
            "random_state": 42,
            "max_depth": 16,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "random_state": 42,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "max_features": 0.8,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "random_state": 42,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "max_features": "sqrt",
            "n_jobs": -1,
        },
    ]

    print("\n[Busca de configuração focada em precisão macro]")
    best_candidate = None

    for topk in topk_candidates:
        selected_cols = ranked_features.head(topk).index.tolist()

        for config_idx, params in enumerate(param_candidates, start=1):
            metrics = evaluate_candidate(X, y, selected_cols, params)

            print(
                f"- topk={topk:02d} | cfg={config_idx} | "
                f"precision_macro={metrics['precision_macro']:.4f} | "
                f"accuracy={metrics['mean_acc']:.4f} | "
                f"f1_macro={metrics['f1_macro']:.4f}"
            )

            candidate = {
                "topk": topk,
                "selected_cols": selected_cols,
                "params": params,
                "metrics": metrics,
            }

            if best_candidate is None:
                best_candidate = candidate
                continue

            best_prec = best_candidate["metrics"]["precision_macro"]
            cur_prec = candidate["metrics"]["precision_macro"]

            if cur_prec > best_prec:
                best_candidate = candidate
            elif cur_prec == best_prec and candidate["metrics"]["mean_acc"] > best_candidate["metrics"]["mean_acc"]:
                best_candidate = candidate

    selected_cols = best_candidate["selected_cols"]
    best_params = best_candidate["params"]
    best_metrics = best_candidate["metrics"]

    print("\n[Melhor configuração encontrada]")
    print(f"- Top features: {best_candidate['topk']}")
    print(f"- precision_macro: {best_metrics['precision_macro']:.4f}")
    print(f"- accuracy: {best_metrics['mean_acc']:.4f}")
    print(f"- f1_macro: {best_metrics['f1_macro']:.4f}")

    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X[selected_cols], y)

    artifact = {
        "model": final_model,
        "feature_cols": selected_cols,
        "class_labels": sorted(y.unique().tolist()),
        "training_summary": {
            "rows": int(len(y)),
            "total_feature_candidates": int(len(feature_cols)),
            "selected_feature_count": int(len(selected_cols)),
            "selected_feature_names": selected_cols,
            "best_params": best_params,
            "cv_precision_macro": best_metrics["precision_macro"],
            "cv_accuracy": best_metrics["mean_acc"],
            "cv_f1_macro": best_metrics["f1_macro"],
        },
    }

    model_path = models_dir / "best_rf_model.pkl"
    joblib.dump(artifact, model_path)
    print(f"Melhor modelo do treinamento salvo em {model_path}")

    print("\n[Avaliação do Modelo via Out-Of-Fold Class Report]:")
    y_true = y.astype(str).values
    y_pred = best_metrics["oof_preds"]
    report_text = classification_report(y_true, y_pred)
    print(report_text)

    metrics_json = {
        "accuracy": best_metrics["mean_acc"],
        "precision_macro": best_metrics["precision_macro"],
        "recall_macro": best_metrics["recall_macro"],
        "f1_macro": best_metrics["f1_macro"],
        "total_classes": int(y.nunique()),
        "total_features": int(len(selected_cols)),
        "total_samples": int(len(y)),
        "selected_features": selected_cols,
        "classes_metrics": {
            class_id: {
                "precision": float(best_metrics["report_dict"][class_id]["precision"]),
                "recall": float(best_metrics["report_dict"][class_id]["recall"]),
                "f1-score": float(best_metrics["report_dict"][class_id]["f1-score"]),
                "support": int(best_metrics["report_dict"][class_id]["support"]),
            }
            for class_id in sorted(y.unique().tolist())
            if class_id in best_metrics["report_dict"]
        },
    }

    metrics_path = models_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Métricas salvas em {metrics_path}")

    feat_imp = pd.Series(final_model.feature_importances_, index=selected_cols).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title("Top 20 Features que o RF deu maior importância")
    plt.tight_layout()
    plt.savefig(models_dir / "feature_importance.png")
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xlabel("Previsão Predita")
    plt.ylabel("Animal Verdadeiro")
    plt.title("Matriz de Confusão Mapeada - Random Forest")
    plt.savefig(models_dir / "confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()
