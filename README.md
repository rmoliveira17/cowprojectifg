# Projeto: Identificação de Bovinos via Visão Computacional (CowProject)

Desenvolvimento de uma pipeline baseada em Inteligência Artificial para a identificação autônoma de indivíduos bovinos através do rastreio de poses anatômicas e análises geométricas vetoriais.

📚 **[Clique aqui para acessar o Relatório Teórico Descritivo sobre as Features Biomêtricas](relatorio_analise_features.md)**

## Estrutura do Projeto

- `src/data/`: Script(s) para manuseio e separação dos diretórios do Dataset base de classificação das vacas.
- `src/features/`: Engrenagem e algoritmos de extração geométrica, distâncias, e matrizes analíticas de exploração de dados (EDA).
- `src/models/`: Onde treinam-se e se definem os cérebros do sistema (Yolo para Keypoints, Random Forest para identificação).
- `src/utils/`: Funções utilitárias e matrizes de definições corporais (`geometry_func.py`).

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

Organize seus datasets rodando a rotina de preparo. (Esse script deve construir as divisões de _train_/_validation_/test em folds dentro de `data/`).

```bash
python src/train_yolo/prepare_dataset.py --input-root sua-pasta/ --output-root data/datasets/classifications/
```

### Passo 1.1: Converter Labels

```bash
python src/train_yolo/convert_labels_to_yolo_pose.py --dataset-root data/datasets/keypoints
```

### Passo 2: O Modelo de Visão (Keypoints YOLO Pose)

Antes de construir geometria tabular, é preciso ter a visão detectando o corpo do animal.

```bash
python src/train_yolo/train_yolo_kfold.py
```

### Passo 3: Geração de Atributos e EDA

Use a "visão" pra colher os metadados de biometria criando automaticamente um poderoso CSV. Depois, teste se eles condizem usando nosso relatório gráfico de Exploração Descritiva de Dados.

#### Qual script usar para extrair features?

Você tem **duas opções** para gerar features geométricas a partir das imagens:

1. `src/features/generate_geometric_features_from_dataset.py`
   - **Propósito:** gerar um CSV consolidado de features geométricas e um relatório JSON com estatísticas de sucesso/falha por split (`train`, `val`, `test`).
   - **Quando usar:** fluxo padrão de geração de dataset tabular para classificação, com foco em robustez e resumo do processo.
   - **Como usar:**

   ```bash
   python src/features/generate_geometric_features_from_dataset.py \
   	 --dataset-root src/data/datasets/classifications \
   	 --model-path src/models/yolo/best.pt \
   	 --output-csv src/data/datasets/classifications/geometric_features.csv \
   	 --output-report src/data/datasets/classifications/geometric_features_report.json
   ```

2. `src/features/extract_geometric_features.py`
   - **Propósito:** além do CSV, salvar payloads de inferência por imagem (`boxes`, `keypoints_xy`, `keypoints_conf`) em arquivos JSON dentro de `labels`.
   - **Quando usar:** auditoria/depuração das predições do YOLO Pose, rastreabilidade por imagem e inspeção detalhada dos resultados.
   - **Como usar:**
   ```bash
   python src/features/extract_geometric_features.py \
   	 --dataset-root src/data/datasets/classifications \
   	 --model-path src/models/yolo/best.pt \
   	 --output-csv src/data/datasets/classifications/geometric_features.csv \
   	 --conf 0.25 \
   	 --imgsz 960
   ```

> **Resumo prático:**
>
> - Use `generate_geometric_features_from_dataset.py` para o pipeline principal de treinamento/classificação.
> - Use `extract_geometric_features.py` quando precisar de inspeção detalhada das predições do modelo.

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

- **Renato Milhomem** (@rmoliveira17)
- **Felipe Crispim** (Desenvolvimento)
