"""
Aplicação Streamlit para Classificação de Bovinos com Random Forest
Interface moderna e elegante para predição de classe bovina a partir de imagens
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.config import (
    UI_CONFIG,
    COLORS,
    YOLO_MODEL_PATH,
    RF_MODEL_PATH,
    CLASSES_CSV,
    KEYPOINT_NAMES,
    ALL_FEATURES,
    DISTANCE_FEATURES,
    ANGLE_FEATURES,
    CACHE_TTL,
)
from ui.helpers import (
    load_yolo_model,
    load_rf_model,
    load_classes,
    detect_keypoints,
    build_geometric_features,
    predict_class,
    draw_keypoints_on_image,
    load_model_metrics,
    get_top_predictions,
    format_percentage,
)

# ============================================================================
# CONFIGURAÇÃO DE PÁGINA
# ============================================================================

st.set_page_config(**UI_CONFIG)

# CSS customizado
custom_css = """
<style>
    :root {
        --primary-color: #2ECC71;
        --secondary-color: #3498DB;
        --accent-color: #E74C3C;
        --background-color: #1E1E1E;
        --text-color: #ECF0F1;
    }
    
    body {
        background-color: #0F0F0F;
        color: #ECF0F1;
    }
    
    .metric-card {
        background-color: #2A2A2A;
        border-left: 4px solid #2ECC71;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        margin: 20px 0;
    }
    
    .confidence-high {
        color: #2ECC71;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #F39C12;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #E74C3C;
        font-weight: bold;
    }
    
    .keypoint-name {
        font-size: 0.85em;
        color: #95A5A6;
        font-family: monospace;
    }
    
    h1, h2, h3 {
        color: #2ECC71;
    }
    
    .stButton > button {
        background-color: #2ECC71;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        transition: background-color 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #27AE60;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# CACHE E ESTADO GLOBAL
# ============================================================================

@st.cache_resource(ttl=CACHE_TTL)
def get_yolo_model():
    """Carrega modelo YOLO uma vez e caches"""
    return load_yolo_model(YOLO_MODEL_PATH)


@st.cache_resource(ttl=CACHE_TTL)
def get_rf_model():
    """Carrega modelo Random Forest uma vez e caches"""
    return load_rf_model(RF_MODEL_PATH)


@st.cache_resource(ttl=CACHE_TTL)
def get_classes_list():
    """Carrega lista de classes uma vez"""
    return load_classes(CLASSES_CSV)


# Inicializar estado da sessão
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================================
# PÁGINA: PREDIÇÃO
# ============================================================================

def page_prediction():
    """Página de predição individual"""
    st.header("🎯 Predição de Classe Bovina")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📤 Upload da Imagem")
        uploaded_file = st.file_uploader(
            "Selecione uma imagem da vaca",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Imagem com vista lateral clara da vaca"
        )
        
        if uploaded_file is not None:
            # Exibir imagem original
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem carregada", use_column_width=True)
            
            # Botão para processar
            if st.button("🚀 Processar Imagem e Fazer Predição", use_container_width=True):
                with st.spinner("⏳ Detectando keypoints..."):
                    # Converter PIL para OpenCV
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Carregar modelo YOLO
                    yolo_model = get_yolo_model()
                    
                    # Detectar keypoints
                    keypoints = detect_keypoints(image_cv, yolo_model)
                    
                    if keypoints is None:
                        st.error("❌ Não foi possível detectar os keypoints da vaca. Tente outra imagem.")
                    else:
                        st.success("✅ Keypoints detectados com sucesso!")
                        
                        # Armazenar na sessão para usar em outra coluna
                        st.session_state.last_keypoints = keypoints
                        st.session_state.last_image = image_cv
    
    # Coluna direita para resultados
    with col2:
        if "last_keypoints" in st.session_state and st.session_state.last_keypoints is not None:
            st.subheader("🔍 Imagem com Keypoints")
            
            # Desenhar keypoints na imagem
            image_with_kpts = draw_keypoints_on_image(
                st.session_state.last_image,
                st.session_state.last_keypoints
            )
            st.image(image_with_kpts, caption="Keypoints detectados", use_column_width=True)
            
            # Informações dos keypoints
            with st.expander("📍 Detalhes dos Keypoints Detectados"):
                kpts_df = pd.DataFrame(
                    st.session_state.last_keypoints,
                    columns=["X", "Y"],
                    index=range(8)
                )
                kpts_df["Nome"] = kpts_df.index.map(KEYPOINT_NAMES)
                st.dataframe(
                    kpts_df[["Nome", "X", "Y"]],
                    use_container_width=True,
                    hide_index=False
                )
            
            # ================================================================
            # FAZER PREDIÇÃO
            # ================================================================
            
            st.subheader("🤖 Resultado da Predição")
            
            with st.spinner("⏳ Extraindo features geométricas..."):
                features = build_geometric_features(st.session_state.last_keypoints)
                
                if features is None:
                    st.error("❌ Erro ao construir features geométricas")
                else:
                    st.success("✅ Features extraídas com sucesso!")
                    
                    with st.spinner("⏳ Fazendo predição com modelo Random Forest..."):
                        # Carregar modelos
                        rf_model = get_rf_model()
                        classes = get_classes_list()
                        
                        # Fazer predição
                        predicted_class, confidence, proba_dict = predict_class(
                            rf_model, features, classes
                        )
                        
                        if predicted_class is None:
                            st.error("❌ Erro ao fazer predição")
                        else:
                            # =====================================================
                            # EXIBIR RESULTADO
                            # =====================================================
                            
                            # Card principal com resultado
                            st.markdown(
                                f"""
                                <div class="prediction-card">
                                    <h1>ID do Bovino: {predicted_class}</h1>
                                    <p style="font-size: 1.2em; margin-top: 10px;">
                                        Confiança: <span class="confidence-high">{format_percentage(confidence)}</span>
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Métricas secundárias
                            metrics_cols = st.columns(3)
                            with metrics_cols[0]:
                                st.metric("Confiança", format_percentage(confidence), delta=None)
                            with metrics_cols[1]:
                                st.metric("Total de Classes", len(classes))
                            with metrics_cols[2]:
                                st.metric("Total de Features", len(ALL_FEATURES))
                            
                            # Top 3 predições alternativas
                            st.subheader("🏆 Top 3 Predições")
                            top_predictions = get_top_predictions(proba_dict, top_k=3)
                            
                            top_cols = st.columns(3)
                            for idx, (class_id, prob) in enumerate(top_predictions):
                                with top_cols[idx % 3]:
                                    medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "   "
                                    st.info(
                                        f"{medal} **{class_id}**\n\n"
                                        f"Confiança: {format_percentage(prob)}"
                                    )
                            
                            # Visualização de probabilidades (todas as classes)
                            with st.expander("📊 Ver Probabilidades de Todas as Classes"):
                                proba_sorted = dict(sorted(
                                    proba_dict.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                ))
                                
                                # Criar DataFrame para visualizar
                                proba_df = pd.DataFrame({
                                    "ID do Bovino": list(proba_sorted.keys()),
                                    "Confiança (%)": [v * 100 for v in proba_sorted.values()]
                                })
                                
                                # Gráfico
                                fig = px.bar(
                                    proba_df.head(15),
                                    x="ID do Bovino",
                                    y="Confiança (%)",
                                    title="Top 15 Classes por Probabilidade",
                                    color="Confiança (%)",
                                    color_continuous_scale="Greens",
                                )
                                fig.update_layout(
                                    template="plotly_dark",
                                    hovermode="x unified",
                                    height=400,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Tabela completa
                                st.dataframe(proba_df, use_container_width=True, hide_index=True)
                            
                            # Features utilizadas for predição
                            with st.expander("📐 Features Geométricas Extraídas"):
                                features_df = pd.DataFrame({
                                    "Feature": list(features.keys()),
                                    "Valor": list(features.values())
                                })
                                
                                # Separar por tipo
                                dist_df = features_df[features_df["Feature"].str.startswith("dist_")]
                                angle_df = features_df[features_df["Feature"].str.startswith("angle_")]
                                
                                st.markdown("**Distâncias entre Keypoints** (px)")
                                st.dataframe(dist_df, use_container_width=True, hide_index=True)
                                
                                st.markdown("**Ângulos entre Keypoints** (graus)")
                                st.dataframe(angle_df, use_container_width=True, hide_index=True)
                            
                            # Salvar no histórico
                            st.session_state.prediction_history.append({
                                "predicted_class": predicted_class,
                                "confidence": confidence,
                                "features": features,
                            })
        else:
            st.info("👈 Carregue uma imagem e clique em 'Processar Imagem' para começar!")


# ============================================================================
# PÁGINA: MÉTRICAS DO MODELO
# ============================================================================

def page_metrics():
    """Página de exibição de métricas do modelo treinado"""
    st.header("📊 Métricas do Modelo Random Forest")
    
    st.info(
        "Esta página exibe as métricas de desempenho do modelo Random Forest "
        "treinado com K-Fold Stratificado (5 folds)."
    )
    
    try:
        metrics = load_model_metrics()
        
        # Tabs para diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["📈 Resumo", "🎯 Performance por Classe", "📊 Visualizações"])
        
        with tab1:
            st.subheader("Resumo Geral do Modelo")
            
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Acurácia Geral", f"{metrics.get('accuracy', 0) * 100:.2f}%")
            with summary_cols[1]:
                st.metric("Total de Classes", metrics.get("total_classes", 30))
            with summary_cols[2]:
                st.metric("Total de Features", metrics.get("total_features", 28))
            with summary_cols[3]:
                st.metric("Algoritmo", "Random Forest")
            
            # Informações adicionais
            st.markdown("### 🔧 Configuração do Modelo")
            config_info = {
                "n_estimators": "100",
                "max_depth": "15",
                "criterion": "gini",
                "class_weight": "balanced",
                "validação": "K-Fold Stratificado (5 folds)",
            }
            config_df = pd.DataFrame(list(config_info.items()), columns=["Parâmetro", "Valor"])
            st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("Performance por Classe (Precision, Recall, F1-Score)")
            
            classes_metrics = metrics.get("classes_metrics", {})
            
            if classes_metrics:
                # Criar DataFrame com métricas por classe
                metrics_data = []
                for class_id, class_metrics in classes_metrics.items():
                    metrics_data.append({
                        "ID do Bovino": class_id,
                        "Precision": class_metrics.get("precision", 0),
                        "Recall": class_metrics.get("recall", 0),
                        "F1-Score": class_metrics.get("f1-score", 0),
                        "Support": class_metrics.get("support", 0),
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(
                    metrics_df.style.format({
                        "Precision": "{:.3f}",
                        "Recall": "{:.3f}",
                        "F1-Score": "{:.3f}",
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Gráficos de performance
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_precision = px.bar(
                        metrics_df,
                        x="ID do Bovino",
                        y="Precision",
                        title="Precision por Classe",
                        color="Precision",
                        color_continuous_scale="Greens",
                    )
                    fig_precision.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_precision, use_container_width=True)
                
                with col2:
                    fig_recall = px.bar(
                        metrics_df,
                        x="ID do Bovino",
                        y="Recall",
                        title="Recall por Classe",
                        color="Recall",
                        color_continuous_scale="Blues",
                    )
                    fig_recall.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_recall, use_container_width=True)
            else:
                st.warning(
                    "⚠️ Métricas por classe não disponíveis. "
                    "Execute o treinamento do modelo para gerar as métricas."
                )
        
        with tab3:
            st.subheader("Visualizações Adicionais")
            
            # Buscar imagens de visualizações salvas
            from pathlib import Path
            rf_dir = Path(__file__).parent.parent / "models" / "random_forest"
            
            col1, col2 = st.columns(2)
            
            with col1:
                confusion_matrix_path = rf_dir / "confusion_matrix.png"
                if confusion_matrix_path.exists():
                    st.markdown("### Matriz de Confusão")
                    img = Image.open(confusion_matrix_path)
                    st.image(img, use_column_width=True)
                else:
                    st.info("📊 Matriz de confusão será gerada após o treinamento")
            
            with col2:
                feature_importance_path = rf_dir / "feature_importance.png"
                if feature_importance_path.exists():
                    st.markdown("### Importância das Features (Top 20)")
                    img = Image.open(feature_importance_path)
                    st.image(img, use_column_width=True)
                else:
                    st.info("📊 Gráfico de importância será gerado após o treinamento")
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar métricas: {str(e)}")
        st.warning(
            "⚠️ Execute o treinamento do modelo Random Forest primeiro:\n"
            "```bash\npython src/models/train_random_forest.py\n```"
        )


# ============================================================================
# PÁGINA: SOBRE
# ============================================================================

def page_about():
    """Página com informações sobre o projeto"""
    st.header("ℹ️ Sobre Este Projeto")
    
    st.markdown("""
    ## 🐄 Sistema de Classificação Bovina com IA
    
    Esta aplicação utiliza **Deep Learning** e **Machine Learning** para identificar bovinos
    através de análise de imagens com visão computacional.
    
    ### 🎯 Objetivo
    
    Classificar bovinos em 30 classes diferentes (identificadores únicos de animais)
    através de:
    1. Detecção automática de keypoints corporais usando **YOLO Pose**
    2. Extração de features geométricas (distâncias e ângulos entre keypoints)
    3. Classificação usando **Random Forest**
    
    ### 🔬 Tecnologia Utilizada
    
    - **YOLO v8 Pose**: Detecção de 8 keypoints corporais da vaca
    - **Random Forest**: Classificação em 30 classes
    - **Features Geométricas**: 28 features (17 distâncias + 11 ângulos)
    - **Streamlit**: Interface web moderna e responsiva
    
    ### 📍 Keypoints Corporais Detectados
    
    A vaca possui 8 pontos de referência principais que são detectados automaticamente:
    """)
    
    # Tabela de keypoints
    keypoints_data = []
    for i, name in KEYPOINT_NAMES.items():
        keypoints_data.append({
            "ID": i,
            "Nome": name,
            "Descrição": [
                "Ponto mais alto do corpo da vaca",
                "Região das costas",
                "Parte superior da anca",
                "Parte inferior da anca",
                "Articulação da anca",
                "Início da cauda",
                "Pino superior da cintura",
                "Pino inferior da cintura",
            ][i]
        })
    
    keypoints_df = pd.DataFrame(keypoints_data)
    st.dataframe(keypoints_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### 📐 Features Geométricas
    
    Com base nos 8 keypoints, são calculadas **28 features**:
    - **17 Distâncias**: Distância espacial entre pares de keypoints (em pixels)
    - **11 Ângulos**: Ângulos formados por triângulos de 3 keypoints (em graus)
    
    #### Exemplos de Distâncias
    """)
    
    st.code("""
    dist_withers_back - Distância entre cernelha e costas
    dist_hook_up_hook_down - Distância entre ganchos superior e inferior
    dist_hip_tail_head - Distância entre anca e base de cauda
    ... e mais 14 distâncias
    """)
    
    st.markdown("""
    #### Exemplos de Ângulos
    """)
    
    st.code("""
    angle_withers_back_hook_up - Ângulo no vértice "costas"
    angle_hook_up_hook_down_hip - Ângulo no vértice "gancho inferior"
    ... e mais 9 ângulos
    """)
    
    st.markdown("""
    ### 📊 Dataset
    
    O modelo foi treinado com:
    - **1500 imagens** de bovinos rotuladas
    - **30 classes** (IDs únicos de bovinos)
    - **K-Fold Stratificado** (5 folds) para validação robusta
    - **100% de sucesso** na extração de features geométricas
    
    ### 🚀 Como Usar
    
    1. Acesse a aba **"Predição"**
    2. Faça upload de uma imagem clara da vaca (vista lateral recomendada)
    3. Clique em **"Processar Imagem e Fazer Predição"**
    4. O sistema irá:
       - Detectar os 8 keypoints corporais
       - Extrair as 28 features geométricas
       - Fazer a predição usando Random Forest
       - Exibir a classe predita e confiança
    5. Visualize métricas detalhadas na aba **"Métricas do Modelo"**
    
    ### 📚 Documentação
    
    Para mais informações sobre o projeto, consulte o README principal do projeto.
    
    ### 👨‍💻 Stack Técnico
    
    - **Backend**: Python 3.10+
    - **Frontend**: Streamlit
    - **ML/DL**: Scikit-learn, PyTorch, Ultralytics
    - **Visualização**: Plotly, Matplotlib
    - **Processamento de Imagem**: OpenCV, PIL
    """)


# ============================================================================
# MENU PRINCIPAL E NAVEGAÇÃO
# ============================================================================

def main():
    """Função principal com navegação"""
    
    # Sidebar com navegação
    with st.sidebar:
        st.markdown("## 🐄 Menu Principal")
        
        selected = option_menu(
            menu_title=None,
            options=["🎯 Predição", "📊 Métricas", "ℹ️ Sobre"],
            icons=["bullseye", "bar-chart", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        
        # Histórico de predições
        if len(st.session_state.prediction_history) > 0:
            st.markdown("### 📜 Histórico Recente")
            for idx, pred in enumerate(st.session_state.prediction_history[-5:], 1):
                st.write(
                    f"**{idx}. ID {pred['predicted_class']}** "
                    f"(confiança: {format_percentage(pred['confidence'])})"
                )
            
            if st.button("🗑️ Limpar Histórico"):
                st.session_state.prediction_history = []
                st.rerun()
        
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #95A5A6; font-size: 0.85em;'>
            <p>🐄 Classificação Bovina com IA</p>
            <p>Versão 1.0 | 2026</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Renderizar página selecionada
    if selected == "🎯 Predição":
        page_prediction()
    elif selected == "📊 Métricas":
        page_metrics()
    elif selected == "ℹ️ Sobre":
        page_about()


if __name__ == "__main__":
    main()
