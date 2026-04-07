import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    csv_path = Path("data/datasets/classifications/geometric_features.csv")
    out_dir = Path("data/eda_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Dataset {csv_path} não encontrado. Execute a extração de features primeiro.")
        return
        
    df = pd.read_csv(csv_path)
    print("Características do dataset geomérico base:")
    print(df.info())
    
    # 1. Gráfico de Disbribuição das classes / Animais
    if 'class_name' in df.columns:
        plt.figure(figsize=(12, 8))
        counts = df['class_name'].value_counts()
        sns.barplot(y=counts.index, x=counts.values, orient='h')
        plt.title('Distribuição de Amostras por Animal (Classe)')
        plt.tight_layout()
        plt.savefig(out_dir / 'class_distribution.png')
        plt.close()

    # Identificar colunas numéricas de geometria
    numeric_cols = [c for c in df.columns if c.startswith('dist_') or c.startswith('angle_')]
    
    if numeric_cols:
        # 2. Mapa de Calor (Correlações)
        plt.figure(figsize=(16, 14))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
        plt.title('Matriz de Correlação das Features Geométricas')
        plt.tight_layout()
        plt.savefig(out_dir / 'correlation_matrix.png')
        plt.close()
        
        # 3. Boxplot para avaliar usabilidade preditiva de features selecionadas
        # Iremos selecionar as top 5 com maior variação como exemplo de EDA
        variances = df[numeric_cols].var().sort_values(ascending=False)
        top_features = variances.head(5).index.tolist()
        
        for feat in top_features:
            if 'class_name' in df.columns:
                plt.figure(figsize=(14, 8))
                sns.boxplot(data=df, x=feat, y='class_name')
                plt.title(f'Distribuição Analítica de {feat} por Vaca')
                plt.tight_layout()
                plt.savefig(out_dir / f'boxplot_{feat}.png')
                plt.close()
                
    print(f"Análise Descritiva Finalizada! Resultados numéricos e diagramas em {out_dir}")

if __name__ == "__main__":
    main()
