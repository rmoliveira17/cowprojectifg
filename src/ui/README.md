# 🐄 Classificação Bovina com IA - Interface Streamlit

Interface moderna e elegante para predição de classe bovina usando Deep Learning (YOLO) e Machine Learning (Random Forest).

## 🚀 Início Rápido

### Executar a Aplicação

```bash
# Usando o script de inicialização
./run_app.sh

# Ou diretamente com uv
uv run streamlit run src/ui/app.py
```

A aplicação estará disponível em: **http://localhost:8501**

## 📋 Requisitos

- Python >= 3.10
- `uv` (gerenciador de pacotes)
- Ambiente virtual já configurado

As dependências estão listadas em `pyproject.toml` e incluem:

- Streamlit >= 1.41.0
- PyTorch 2.10.0
- Ultralytics (YOLO)
- Scikit-learn
- OpenCV
- E mais...

## 📁 Estrutura do Projeto UI

```
src/ui/
├── app.py                  # Aplicação principal Streamlit
├── config.py              # Configurações e constantes
├── helpers.py             # Funções auxiliares
└── __init__.py
```

### config.py

Arquivo de configuração centralizado que contém:

- Caminhos dos modelos (YOLO, Random Forest)
- Caminhos dos arquivos de dados (classes, features)
- Configurações de detecção YOLO
- Lista de features geométricas
- Keypoints corporais da vaca
- Esquema de cores
- Configurações da UI

### helpers.py

Funções auxiliares reutilizáveis:

- `load_yolo_model()` - Carrega modelo YOLO com cache
- `load_rf_model()` - Carrega modelo Random Forest com cache
- `detect_keypoints()` - Detecta keypoints usando YOLO
- `build_geometric_features()` - Extrai 28 features geométricas
- `predict_class()` - Faz predição com Random Forest
- `draw_keypoints_on_image()` - Desenha keypoints na imagem
- `load_model_metrics()` - Carrega métricas do modelo
- `get_top_predictions()` - Retorna top-k predições
- E mais...

### app.py

Aplicação principal Streamlit com 3 páginas:

1. **🎯 Predição**
   - Upload de imagem
   - Detecção de keypoints com YOLO
   - Extração de features
   - Predição com Random Forest
   - Visualização de resultados
   - Top 3 predições alternativas
   - Visualização de probabilidades
   - Detalhes dos keypoints
   - Features extraídas

2. **📊 Métricas do Modelo**
   - Resumo geral (acurácia, total de classes/features)
   - Performance por classe (precision, recall, F1-score)
   - Visualizações (matriz de confusão, feature importance)
   - Gráficos interativos com Plotly

3. **ℹ️ Sobre**
   - Documentação do projeto
   - Explicação dos 8 keypoints corporais
   - Descrição das 28 features geométricas
   - Informações do dataset
   - Stack técnico utilizado

## 🔄 Fluxo de Predição

```
User             App                  Models
 │                │                      │
 ├─ Upload Imagem ──────────────────────>│
 │                │                      │
 │                ├─ Detecta keypoints ──>YOLO
 │                │                      │
 │                ├─ Extrai features ────>Geométricas
 │                │                      │
 │                ├─ Faz predição ───────>Random Forest
 │                │                      │
 │<─ Mostra resultados────────────────────│
 │                │                      │
```

## 📊 Features Geométricas

O modelo usa **28 features** extraídas dos 8 keypoints corporais:

### 8 Keypoints Corporais

0. Cernelha (Withers)
1. Costas (Back)
2. Gancho Superior (Hook Up)
3. Gancho Inferior (Hook Down)
4. Anca (Hip)
5. Base da Cauda (Tail Head)
6. Pino Superior (Pin Up)
7. Pino Inferior (Pin Down)

### 17 Distâncias

Distâncias espaciais entre pares de keypoints (em pixels)

### 11 Ângulos

Ângulos formados por triângulos de 3 keypoints (em graus)

## 🎨 Design da UI

A interface utiliza:

- **Tema Escuro**: Layout escuro com acentos verdes (#2ECC71)
- **Multi-página**: Navegação via `streamlit-option-menu`
- **Cards**: Elementos visuais destacados com gradientes
- **Gráficos Interativos**: Plotly para visualizações
- **Responsive**: Responsivo para diferentes resoluções
- **Indicadores de Carregamento**: Spinners durante processamento

## ⚙️ Configuração

### Variáveis de Ambiente

Nenhuma variável de ambiente obrigatória. Todas as configurações estão em `src/ui/config.py`.

### Modelos Necessários

O projeto espera os seguintes modelos treinados:

- `src/models/yolo/best.pt` - Modelo YOLO para detecção de keypoints
- `src/models/random_forest/best_rf_model.pkl` - Modelo Random Forest

Se os modelos não existirem, a aplicação exibirá mensagens de erro informativas.

## 📈 Performance

### Cache de Modelos

Os modelos YOLO e Random Forest são carregados uma única vez e reutilizados graças ao decorador `@st.cache_resource` do Streamlit, melhorando significativamente a performance.

### Tempo de Processamento

- Detecção de keypoints: ~1-2 segundos (YOLO)
- Extração de features: <100ms
- Predição: <10ms (Random Forest)

## 🔧 Troubleshooting

### ImportError: No module named 'streamlit'

```bash
uv sync
uv run streamlit run src/ui/app.py
```

### Modelos Não Encontrados

Certifique-se que os modelos estão em:

- `src/models/yolo/best.pt`
- `src/models/random_forest/best_rf_model.pkl`

Treins os modelos usando:

```bash
python src/models/train_random_forest.py
```

### Erro ao Detectar Keypoints

Verifique:

1. A imagem é de uma vaca em vista lateral clara?
2. O modelo YOLO está presente em `src/models/yolo/best.pt`?
3. A resolução da imagem é adequada (> 640x480 recomendado)?

## 📚 Documentação Adicional

- [README Principal](../../README.md) - Documentação geral do projeto
- [Relatório de Features](../../relatorio_analise_features.md) - Análise detalhada das features
- [src/features/](../features/) - Código de extração de features
- [src/models/](../models/) - Código de treinamento de modelos

## 🐄 Sobre o Projeto

Sistema de classificação de bovinos usando visão computacional com:

- **YOLO v8 Pose**: Detecção de keypoints corporais
- **Random Forest**: Classificação em 30 classes (IDs únicos de bovinos)
- **Streamlit**: Interface web moderna
- **30 classes**: IDs únicos de bovinos (1106, 1122, 1221, etc)
- **1.5k+ imagens**: Dataset de treinamento

## 📄 Licença

Consulte o arquivo LICENSE do projeto principal.

---

**Versão 1.0** | 2026 | 🐄 Classificação Bovina com IA
