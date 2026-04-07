# Projeto: Identificação de Bovinos via Visão Computacional (CowProject)

Desenvolvimento de uma pipeline baseada em Inteligência Artificial para a identificação autônoma de indivíduos bovinos através do rastreio de poses anatômicas e análises geométricas vetoriais.

📚 **[Clique aqui para acessar o Relatório Teórico Descritivo sobre as Features Biomêtricas](relatorio_analise_features.md)**

## Estrutura do Projeto (Fase 3 & 4)
* `src/data/`: Script(s) para manuseio e separação dos diretórios do Dataset base de classificação das vacas.
* `src/features/`: Engrenagem e algoritmos de extração geométrica, distâncias, e matrizes analíticas de exploração de dados (EDA).
* `src/models/`: Onde treinam-se e se definem os cérebros do sistema (Yolo para Keypoints, Random Forest para identificação).
* `src/utils/`: Funções utilitárias e matrizes de definições corporais (`geometry_func.py`).

---

## 🚀 Como executar o Setup do Projeto

Optamos por utilizar o gerenciador de dependências moderno e ultrarrápido [uv](https://github.com/astral-sh/uv). Siga o passo a passo curto para iniciar seu ambiente e baixar todos os pacotes:

1. Instale a ferramenta globalmente pelo terminal através do `pip install uv`.
2. Baixe os pesos ou repositório contendo tudo.
3. Crie e ative o ambiente virtal, instalando tudo contido no nosso `pyproject.toml` (que converteu nosso antigo requirements) da seguinte maneira:

```bash
# Na pasta da raiz do cowprojectifg:
uv sync

# Ative o ambiente .venv gerado:
# Linux/Mac
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

---

## 🧠 Pipeline de Treino e Execução
Uma vez dentro do ambiente configurado, você precisa executar as frentes do projeto na ordem correta para retro-alimentar as previsões. Para rodar qualquer uma destas fases, tenha certeza de estar com os datasets nos diretórios corretos e rode os códigos usando o ambiente ativado:

### Passo 1: Preparação
Organize seus datasets rodando a rotina de preparo. (Esse script deve construir as divisões de *train*/*validation*/test em folds dentro de `data/`).
```bash
python src/data/prepare_classification_dataset.py --input-root sua-pasta/ --output-root data/datasets/classifications/
```

### Passo 2: O Modelo de Visão (Keypoints YOLO Pose)
Antes de construir geometria tabular, é preciso ter a visão detectando o corpo do animal.
```bash
python src/train_yolo/03-train_yolo_kfold.py
```

### Passo 3: Geração de Atributos e EDA
Use a "visão" pra colher os metadados de biometria criando automaticamente um poderoso CSV. Depois, teste se eles condizem usando nosso relatório gráfico de Exploração Descritiva de Dados.
```bash
# Roda YOLO nas vacas, calcula ângulos e salva num .CSV
python src/features/generate_geometric_features_from_dataset.py

# Analisa o CSV salvo e cospe boxplots/histogramas preditivos
python src/features/eda_features.py
```

### Passo 4: O Modelo Cognitivo de Identificação (Random Forest)
Em posse do `.csv` das linhas geométricas, treine o classificador final de modo a criar o limiar de reconhecimento pro animal e gerar a Matriz de Confusão para você validar.
```bash
python src/models/train_random_forest.py
```

---

## Autores
Este projeto é conduzido pelas seguintes diretrizes sob a instituição IFG:
- **Renato Milhoem** (@rmoliveira17)
- **Rafael** (Orientação/Coordenação da Repositório)
- *(Sinta-se livre para completar a ficha de créditos do time através de Pull Requests!)*
