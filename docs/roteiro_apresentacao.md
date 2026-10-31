# Roteiro de Apresentação em Vídeo (8 minutos)
## Projeto: Classificação de Dígitos MNIST com Random Forest, KNN e MLP

---

## 1. Abertura e Contexto (0:00 – 0:45)

**Fala sugerida:**
"Olá! Neste vídeo vou apresentar o desenvolvimento de um projeto de classificação de imagens usando o dataset MNIST — o clássico conjunto de dígitos manuscritos de 0 a 9. O objetivo foi comparar diferentes algoritmos de machine learning, entender suas forças e limitações, e explorar até onde esses modelos conseguem generalizar quando confrontados com situações fora do que aprenderam."

**Mostrar na tela:** título do projeto, nome, e uma imagem de exemplo de dígitos do MNIST.

---

## 2. Dados e Análise Exploratória (0:45 – 2:00)

**Pontos a cobrir:**
- Dataset carregado via `scikit-learn` (`fetch_openml`): 70.000 imagens de dígitos manuscritos
- Cada imagem: matriz 28×28 pixels, achatada em vetor de 784 features
- Pixels em escala de cinza, de 0 (preto) a 255 (branco)
- Classes bem balanceadas (cada dígito com ~9-11% do total)
- Divisão estratificada: 80% treino / 20% teste

**Mostrar na tela:** grade 2×5 com amostras aleatórias de cada dígito; gráfico de distribuição das classes.

**Fala sugerida:**
"Antes de treinar qualquer modelo, fiz uma análise exploratória: confirmei que as classes estão bem balanceadas, visualizei amostras de cada dígito e organizei os dados em uma representação vetorial de 784 posições, normalizando os valores de pixel para a faixa de 0 a 1."

---

## 3. Modelos Treinados (2:00 – 4:00)

**Random Forest**
- Modelo baseado em árvores, rápido para treinar, insensível à escala dos dados
- Otimização com `RandomizedSearchCV` — resultado interessante: a busca com poucas iterações **piorou** a acurácia em relação ao modelo padrão, mostrando que otimização de hiperparâmetros não é garantia de melhora

**KNN**
- Classifica por proximidade aos vizinhos mais próximos
- Treino instantâneo, mas predição mais lenta
- Otimizado com `GridSearchCV`, testando número de vizinhos e pesos

**MLP (Perceptron Multicamadas)**
- Rede neural com duas camadas ocultas
- Maior tempo de treino entre os três, mas boa acurácia geral

**Mostrar na tela:** tabela comparativa de acurácia, tempo de treino e tempo de predição dos três modelos; matrizes de confusão lado a lado.

**Fala sugerida:**
"Treinei três modelos diferentes: Random Forest, KNN e uma rede neural MLP. Cada um tem um comportamento distinto — o Random Forest é rápido e robusto, o KNN é simples mas lento na hora de prever, e o MLP entrega a melhor acurácia, ao custo de um treinamento mais demorado."

---

## 4. Teste com Classes Ocultas (4:00 – 5:15)

**O experimento:** remover os dígitos 8 e 9 do conjunto de treino e observar o que cada modelo faz quando esses dígitos aparecem no teste.

**Resultado principal:**
- Os três modelos convergiram nas mesmas confusões: dígito 8 → confundido majoritariamente com "3"; dígito 9 → confundido majoritariamente com "4"
- Isso revela uma similaridade visual real entre esses dígitos, e não um viés de um algoritmo específico

**Mostrar na tela:** matriz de confusão com as classes ocultas destacadas.

**Fala sugerida:**
"Um dos experimentos mais reveladores foi ocultar as classes 8 e 9 durante o treino. Como nenhum classificador tradicional tem a opção de dizer 'não sei', os três modelos foram forçados a redirecionar essas amostras para classes conhecidas — e todos convergiram nas mesmas confusões, o que confirma que isso reflete semelhança visual real entre os dígitos."

---

## 5. Inferência com Dados Fora da Distribuição — OOD (5:15 – 6:45)

**O experimento:** submeter os modelos a ruído aleatório puro, sem nenhuma estrutura de dígito.

**Resultado principal (o mais importante do projeto):**
- Random Forest: colapsou em uma única classe, mas com confiança baixa (27%)
- KNN: distribuiu as previsões entre várias classes, confiança intermediária (62%)
- MLP: colapsou em uma única classe com **99,89% de confiança** — o cenário de maior risco

**Mostrar na tela:** gráfico comparando a confiança média dos três modelos diante do ruído.

**Fala sugerida:**
"O teste mais crítico foi apresentar ruído puro aos modelos — imagens sem nenhuma estrutura de dígito. O resultado mais preocupante veio do MLP: mesmo diante de um ruído completamente aleatório, ele afirmou com quase 100% de confiança que aquilo era um dígito '5'. Isso demonstra um problema conhecido em redes neurais: a saída sempre soma 100% entre as classes conhecidas, então o modelo nunca sinaliza incerteza real diante do desconhecido — o que é um risco sério em aplicações de produção."

---

## 6. Teste com Manuscritos Próprios (6:45 – 7:30)

**O experimento:** fotografar dígitos escritos à mão e submeter aos três modelos.

**Resultado:** as predições ficaram muito parecidas com o teste de ruído, evidenciando que o pré-processamento (inversão de cor, centralização, escala) é decisivo — pequenas diferenças na captura da imagem real prejudicam bastante a performance dos modelos treinados em condições controladas do MNIST.

**Fala sugerida:**
"Por fim, testei os modelos com dígitos escritos por mim mesmo, fotografados com celular. Os resultados mostraram como a qualidade e o pré-processamento da imagem — cor, centralização, iluminação — têm um impacto enorme: mesmo dígitos claros para um humano geraram respostas próximas às do teste de ruído, reforçando que esses modelos são sensíveis a qualquer desvio das condições originais de treino."

---

## 7. Conclusão e Aprendizados (7:30 – 8:00)

**Fala sugerida:**
"Em resumo, este projeto mostrou que acurácia alta em dados de teste não garante segurança em produção. Modelos como o MLP, apesar de mais precisos, podem falhar silenciosamente e com alta confiança diante de dados inesperados. Isso reforça a importância de mecanismos adicionais de detecção de anomalias em sistemas reais de classificação, que vão além de simplesmente confiar na probabilidade que o próprio modelo apresenta. Obrigado por assistir!"

**Mostrar na tela:** slide de encerramento com os 3-4 principais aprendizados em bullet points.

---

## Notas de Produção
- Tempo total estimado: ~8 minutos (ajuste o ritmo de fala conforme necessário)
- Recomenda-se gravar a tela mostrando os gráficos/matrizes enquanto narra cada seção
- Priorize mostrar visualmente as matrizes de confusão e o gráfico de confiança OOD — são os pontos mais impactantes para a audiência
