# Relatório de Análise Descritiva: Identificação Biométrica de Bovinos via Características Geométricas

## 1. Introdução
A identificação precisa de animais em rebanhos bovinos é uma necessidade crescente para a zootecnia de precisão, permitindo melhor rastreabilidade, bem-estar animal e automação de processos produtivos. O presente documento descreve a metodologia de extração de características (features) baseada na anatomia do animal (keypoints sensíveis à biometria) no desenvolvimento de um classificador de Machine Learning voltado para predição individual dos animais.

O uso de features contínuas baseadas no processamento de imagens confere ao projeto **invariância à iluminação**, **escala** e **rotação**, além de reduzir severamente a dimensionalidade quando comparado ao processamento bruto de pixels através de Redes Neurais Convolucionais. 

## 2. Metodologia de Mapeamento Anatômico
Neste projeto, utilizamos algoritmos de Deep Learning (como a família YOLOv11 Pose) para criar um esqueleto anatômico simplificado do bovino. O mapeamento baseia-se em 8 (oito) pontos-chave fisiológicos fundamentais, que capturam o perfil anatômico do dorso até os quartos traseiros:

1. **Withers** (Cernelha): Ponto mais alto do dorso, na transição com o pescoço.
2. **Back** (Dorso): Parte central superior do tronco do animal.
3. **Hook up** (Ílio superior): Extremidade superior do osso ilíaco.
4. **Hook down** (Ílio inferior): Projeção inferior mais lateral do ílio.
5. **Hip** (Quadril): Articulação coxofemoral principal.
6. **Tail head** (Inserção da cauda): Base de união da cauda com a pélvis.
7. **Pin up** (Ísquio superior): Ponto superior da tuberosidade isquiática.
8. **Pin down** (Ísquio inferior): Ponto inferior da tuberosidade isquiática.

## 3. Extração das Features Geométricas
A anatomia óssea constitui uma espécie de impressão digital biométrica dos animais. Para alimentar nossos modelos preditivos contornando vieses de proximidade de câmera, as coordenadas `(x, y)` brutas são processadas e traduzidas em atributos puramente relacionais.

### 3.1. Distâncias Euclidianas (17 features)
As distâncias lineares são processadas utilizando a norma Euclidiana entre pares predefinidos de keypoints mapeados, o que forma uma "malha" entre os tecidos:
* `dist_withers_back`, `dist_withers_hook_up`, `dist_withers_hook_down`, entre outras 14 interações diretas.

### 3.2. Ângulos Trigonométricos (11 features)
Para capturar o grau de encurvamento, inclinação óssea da pélvis e morfologia corporal, formam-se arcos utilizando produtos da geometria analítica. Um ângulo é formado pelo vetor de três pontos adjacentes:
* `angle_withers_back_hook_up`, `angle_hook_up_hook_down_tail_head`, e mais 9 combinações de tríades fundamentais.

## 4. Viabilidade Analítica da Classificação
Ao submeter o espaço vetorial de 28 features (17 distâncias e 11 ângulos) à **Análise Exploratória de Dados (EDA)**, verifica-se que:

1. **Baixa correlação absoluta entre grupos distintos**: Distâncias dorsais correlacionam-se fracamente com métricas isquiáticas, ofertando à IA fatores amplos e isolados de decisão.
2. **Clusterização natural**: Ao observarmos o *Boxplot* da distribuição angular por exemplo do arco *(hook_up, tail_head, pin_up)*, percebe-se que cada indivíduo (classe) ocupa zonas restritas de variância, indicando alta usabilidade como limite de decisão nas divisões de uma *Random Forest* ou mapeamentos espaciais de uma matriz de *Support Vector Machine (SVM)*.

## 5. Conclusão
O ecossistema numérico derivado da topologia vetorial reflete fielmente as nuances fenotípicas de cada espécime monitorado. A abordagem provou ser eficiente não só na dimensão computacional reduzida de processamento, mas como um previsor fiel que garante ao sistema de Inteligência Artificial uma alta assertividade no reconhecimento e catalogação passiva das matrizes/animais do rebanho.
