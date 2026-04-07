import argparse
from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser("Aplica Random Forest nas Features Geométricas Tabulares")
    parser.add_argument("--features-csv", default="data/datasets/classifications/geometric_features.csv", help="Dataset CSV")
    parser.add_argument("--models-dir", default="src/models/random_forest", help="Local para salvar o modelo em .pkl e gráficos")
    return parser.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.features_csv)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}\nVocê rodou a geração de features antes?")
        
    df = pd.read_csv(csv_path)
    
    # Identificar colunas target e covariáveis (features)
    feature_cols = [c for c in df.columns if c.startswith('dist_') or c.startswith('angle_')]
    X = df[feature_cols]
    y = df['class_name']
    
    print(f"Iniciando Treinamento com {len(feature_cols)} atributos sobre {len(y)} fotos distribuídas em {y.nunique()} identificadores distintos.")
    
    # Separador Stratificado para 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    best_model = None
    best_acc = 0
    oof_preds = pd.Series(index=y.index, dtype='object')
    
    # Avaliando capacidade de generalização via K-Fold CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # O RF é excelente por trabalhar não linaridades sem necessitar de escalonamento!
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_val)
        oof_preds.iloc[val_idx] = preds
        
        acc = accuracy_score(y_val, preds)
        fold_metrics.append(acc)
        print(f"Fold {fold+1} Acurácia Base: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = clf
            
    print(f"Média Global de Acurácia: {sum(fold_metrics)/len(fold_metrics):.4f}")
    
    # Dump the Best Generalizer
    model_path = models_dir / "best_rf_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Melhor modelo do treinamento salvo em {model_path}")
    
    # Avaliação OOF de Classification e Plot
    print("\n[Avaliação do Modelo via Out-Of-Fold Class Report]:")
    report = classification_report(y, oof_preds)
    print(report)
    
    # Feature Importance
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10,8))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title('Top 20 Features que o RF deu maior importância')
    plt.tight_layout()
    plt.savefig(models_dir / 'feature_importance.png')
    plt.close()
    
    # Confusão (Ajuda a ver em qual cruzamento o modelo confunde as vacas)
    cm = confusion_matrix(y, oof_preds)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel('Previsão Predita')
    plt.ylabel('Animal Verdadeiro')
    plt.title('Matriz de Confusão Mapeada - Random Forest')
    plt.savefig(models_dir / 'confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main()
