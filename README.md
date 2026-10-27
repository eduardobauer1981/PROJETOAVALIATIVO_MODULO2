# Miniprojeto Módulo 2 — Classificação de Dígitos Manuscritos (MNIST)

Projeto de classificação de imagens usando o dataset **MNIST**, comparando três algoritmos de machine learning (Random Forest, KNN e Rede Neural MLP) e explorando seus limites de generalização.

## Bibliotecas utilizadas

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn` (`fetch_openml`, `train_test_split`, `RandomForestClassifier`, `KNeighborsClassifier`, `MLPClassifier`, `Perceptron`, `GridSearchCV`, `RandomizedSearchCV`, métricas)
- `time`

## Estrutura do projeto

### Fase 1 — Carregamento e Análise Exploratória de Imagens (EDA)
- Importação do dataset MNIST via `fetch_openml`
- Verificação da dimensionalidade das matrizes (X e y) e distribuição das classes: 70.000 amostras, 784 features (pixels) cada, com quantidade de amostras bem próxima entre os 10 dígitos
- Grade visual 2x5 com `matplotlib` mostrando exemplos dos dígitos (fundo preto, traço branco)
- Interpretação da estrutura dos dados: escala de intensidade de pixels (0 a 255) e representação vetorial das imagens 28x28 achatadas em 784 features

### Fase 2 — Pipeline de Pré-processamento e Divisão dos Dados
- Divisão estratificada: 80% treino / 20% teste
- Normalização/escalonamento dos pixels para a escala [0.0, 1.0]
- Justificativa: redes neurais e algoritmos de otimização convergem mais rápido e de forma mais estável com valores de entrada numa escala pequena

### Fase 3 — Implementação e Treinamento dos 3 Modelos
- **Random Forest**: treinado e depois otimizado com `RandomizedSearchCV` (n_iter=30), obtendo um modelo melhor que o inicial
- **KNN**: treinado e otimizado via busca de hiperparâmetros, com melhora de acurácia
- **Rede Neural MLP (Perceptron Multicamadas)**: implementado como terceiro modelo de comparação

### Fase 4 — Avaliação Comparativa de Desempenho
- Matrizes de confusão com mapa de calor para os três modelos
- Tabela comparativa com acurácia global, precisão média ponderada, revocação/sensibilidade média ponderada e F1-score ponderado
- Conclusão: o KNN tem o pior tempo de predição do grupo (recalcula distância para todos os pontos de treino a cada predição); considerando acurácia e tempo, o melhor modelo é a Rede Neural

### Fase 5 — Desafios

**1. Treinamento restrito com classes ocultadas (class masking)**
- Dígitos 8 e 9 removidos do treino; avaliação de para onde o modelo redireciona essas classes no teste
- Dígito 8 → confundido majoritariamente com "3" nos três modelos (MLP mais "convicto", 58%); segunda opção diverge entre RF/MLP (→ 2) e KNN (→ 5)
- Dígito 9 → confundido majoritariamente com "4" (Random Forest extremamente concentrado, 84.7%; KNN mais disperso entre 4 e 7)
- Conclusão: essas confusões refletem uma limitação inerente da tarefa (similaridade visual real entre os dígitos), não uma peculiaridade de um algoritmo específico. Nenhum modelo demonstra "incerteza" — todos os classificadores fechados são forçados a "inventar" uma resposta

**2. Teste de generalização extrema (inferência OOD — Out-of-Distribution)**
- Modelos submetidos a ruído aleatório, sem estrutura de dígito
- Random Forest: colapsa numa classe, mas com confiança baixa — risco moderado (a baixa confiança é um sinal de alerta útil)
- KNN: distribui as previsões, confiança intermediária — risco moderado (dispersão sugere instabilidade)
- MLP: colapsa com confiança quase perfeita — risco alto, o pior cenário possível (erro silencioso e "confiante")
- Conclusão: aplicações reais precisam de um mecanismo dedicado de detecção de OOD, não bastando confiar na probabilidade (`predict_proba`) do próprio modelo

**3. Inferência com imagens manuscritas próprias**
- Testes com fotos de dígitos escritos à mão revelaram que, após o pré-processamento, as imagens não se pareciam com um dígito MNIST do ponto de vista dos modelos — reagindo de forma semelhante ao ruído puro
- Causas prováveis: dígito não centralizado/desproporcional, inversão de cor incorreta ou inconsistente, espessura de traço e nitidez
- Foi feita uma nova tentativa com imagem editada (fundo mais branco, traços mais grossos), com melhora nos resultados (ex.: Random Forest e MLP acertando o dígito "8" com confiança de 37.7% e 100%, respectivamente; KNN ainda errando)

## Conclusão Geral

Acurácia alta em dados de teste não garante segurança em produção. Um modelo pode ter ótima performance nos dados que "conhece" e ainda assim falhar silenciosamente — e com alta confiança — diante de entradas inesperadas. Por isso, sistemas reais de classificação geralmente precisam de mecanismos adicionais de detecção de anomalias/OOD, não apenas confiar na saída de probabilidade do próprio modelo.
