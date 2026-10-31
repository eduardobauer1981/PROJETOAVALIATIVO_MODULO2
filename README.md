# Reconhecimento de Dígitos Manuscritos (MNIST) — Modelagem e API FastAPI

Projeto avaliativo do Módulo 2. Compara três algoritmos de machine learning na classificação de dígitos manuscritos (MNIST) e serve o modelo escolhido, uma rede neural MLP, por meio de uma API REST construída com FastAPI.

O projeto tem duas partes:

1. **Modelagem (notebook):** análise exploratória, treino, otimização, comparação de modelos e testes de generalização.
2. **API (`app/`):** serviço em camadas, com o modelo carregado uma única vez na RAM, validação estrita de dados e tratamento padronizado de erros.

## Resultados da modelagem

Avaliação no conjunto de teste (20% do MNIST, divisão estratificada, 14.000 imagens):

| Modelo | Acurácia | Tempo de predição (teste inteiro) |
|---|---|---|
| Random Forest (otimizado com `RandomizedSearchCV`) | 96,86% | 0,641 s |
| KNN (otimizado com `GridSearchCV`) | 97,08% | 11,496 s |
| **MLP (rede neural), modelo selecionado** | **97,94%** | **0,036 s** |

Os modelos sem otimização ficaram em 96,72% (Random Forest) e 96,96% (KNN). O MLP teve a melhor acurácia e o menor tempo de predição, por isso foi o escolhido para a API.

### Principais achados dos desafios

- **Classes ocultas no treino (dígitos 8 e 9):** nenhum modelo consegue dizer "não conheço". Os três redirecionaram o 8 principalmente para o 3 (40% a 58% dos casos) e o 9 para o 4 (55% a 85%). A causa é a semelhança visual entre os dígitos, e não um viés de algoritmo.
- **Ruído aleatório (OOD):** o MLP classificou 100% do ruído como "5" com **99,89% de confiança**. O Random Forest também colapsou numa classe, mas com 27% de confiança, e o KNN espalhou as previsões. Acurácia alta não garante segurança diante de dados desconhecidos.
- **Manuscritos próprios:** o pré-processamento da foto (inversão de cores, centralização, espessura do traço e rotação) pesa tanto quanto o modelo.

## Arquitetura da API

```
Requisição HTTP
      |
      v
  routers/   -> endpoints (APIRouter)
      |
      v
  schemas/   -> contratos Pydantic: validação de entrada e saída
      |
      v
  services/  -> carrega o .joblib e faz a inferência (sem HTTP)
      |
      v
  models/mlp_mnist.joblib
```

Cada camada só conhece a de baixo. O service não sabe que existe HTTP, e o router não sabe como a rede neural funciona.

### Decisões técnicas

- **Carregamento único (lifespan):** o modelo é lido do disco uma vez, na inicialização, com `@asynccontextmanager`. Nenhuma rota lê o `.joblib`. O carregamento é idempotente e faz uma predição de aquecimento.
- **Validação estrita (Pydantic):** `Field` com `ge`, `le`, `gt`, limites de tamanho e descrições; `strict=True` (o texto `"128"` não vira número) e `extra="forbid"`. Dados inválidos são barrados antes de chegar ao modelo.
- **Tratamento centralizado de erros:** todo erro vira JSON padronizado. Detalhes internos ficam só no log, e o usuário recebe um `id_erro` para consulta.
- **API resiliente:** se o modelo estiver ausente, corrompido ou incompatível, a API sobe em modo `degradado` em vez de cair.
- **`limiar_confianca`:** regra de negócio ligada ao achado do teste OOD. A resposta traz `confiavel: false` quando a confiança fica abaixo do limiar.

## Estrutura do repositório

```
.
├── digitos_manuscritos.ipynb      # modelagem: EDA, treino, comparação, desafios
├── app/
│   ├── main.py                    # entrada: lifespan, tratadores e roteadores
│   ├── config.py                  # caminho do modelo e constantes
│   ├── erros.py                   # mapeamento de exceções para status HTTP
│   ├── routers/
│   │   ├── health.py              # GET /health
│   │   ├── predicao.py            # POST /v1/predict
│   │   └── dependencias.py
│   ├── schemas/
│   │   ├── predicao.py            # contratos de entrada e saída
│   │   └── erros.py               # contratos de erro (400, 422, 500...)
│   └── services/
│       └── modelo_service.py      # carregamento e inferência
├── models/
│   └── mlp_mnist.joblib           # modelo treinado (gerado pelo notebook)
├── tests/                         # suíte pytest (46 testes)
├── testar_api.py                  # envia uma imagem sua para a API
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

## Como executar

**Requisitos:** Python 3.10 ou superior.

1. **Crie e ative o ambiente virtual** (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. **Instale as dependências:**

```powershell
python -m pip install -r requirements.txt
```

3. **Coloque o modelo em `models/mlp_mnist.joblib`.** Salve-o no notebook com `joblib.dump(modelo_mlp, ...)`. Instale a **mesma versão do scikit-learn** usada no treino (veja com `pip show scikit-learn` no ambiente do notebook), senão o arquivo pode falhar ao carregar.

4. **Suba a API** (na pasta que contém `app/`):

```powershell
python -m uvicorn app.main:app --reload
```

A documentação interativa fica em `http://127.0.0.1:8000/docs`.

## Endpoints

### `GET /health`

Checagem de integridade. Sempre responde **200** enquanto a API estiver no ar.

```json
{
  "status": "ok",
  "servico_ativo": true,
  "modelo_carregado": true,
  "modelo_tipo": "MLPClassifier",
  "mensagem": "Serviço ativo e modelo carregado na memória."
}
```

Se o modelo não estiver na RAM, `status` vira `"degradado"` e `modelo_carregado` fica `false`.

### `POST /v1/predict`

Recebe uma imagem 28x28 achatada em 784 pixels (0 a 255, fundo preto e traço branco, como no MNIST).

| Campo | Tipo | Regra |
|---|---|---|
| `pixels` | lista de números | obrigatório, exatamente 784 valores de 0 a 255 |
| `identificador` | texto | opcional, de 1 a 50 caracteres |
| `limiar_confianca` | número | opcional, maior que 0 e até 1 (padrão 0,5) |
| `top_k` | inteiro | opcional, de 1 a 10 (padrão 3) |

Resposta (200):

```json
{
  "identificador": "digito_01.png",
  "digito_previsto": 7,
  "confianca": 0.98,
  "confiavel": true,
  "melhores_previsoes": [
    {"digito": 7, "probabilidade": 0.98},
    {"digito": 1, "probabilidade": 0.01},
    {"digito": 2, "probabilidade": 0.005}
  ],
  "probabilidades": [0.0, 0.01, 0.005, 0.0, 0.0, 0.0, 0.0, 0.98, 0.0, 0.005]
}
```

Os valores acima são ilustrativos.

### Status codes

| Status | Quando acontece |
|---|---|
| 200 | `/health` e `/v1/predict` com sucesso |
| 400 | JSON malformado (o corpo nem é um JSON) |
| 422 | JSON válido, mas fora do contrato (tipo, faixa ou tamanho inválido) |
| 404 e 405 | Rota inexistente ou método HTTP não permitido |
| 503 | Modelo indisponível (não está na RAM) |
| 500 | Falha inesperada (resposta com `id_erro`, sem detalhes internos) |

## Testando

### Testes automáticos (46 testes)

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

Os testes treinam um MLP minúsculo na hora, então não dependem do `.joblib` real.

| Arquivo | O que prova |
|---|---|
| `test_ciclo_vida.py` | O modelo é lido do disco uma única vez, mesmo após várias requisições. Nenhuma rota importa `joblib` ou `pickle`. |
| `test_validacao_422.py` | Dados inválidos retornam 422 (ou 400) e o modelo é chamado 0 vezes. Os limites exatos são aceitos. |
| `test_endpoints.py` | `/health`, `/v1/predict` e os códigos 400, 404, 405, 500 e 503. Depois de um erro 500, a API continua respondendo. |

### Teste manual

```powershell
# Integridade (modelo_carregado deve ser true)
Invoke-RestMethod http://127.0.0.1:8000/health

# Enviar uma foto de um dígito escrito por você
python testar_api.py "C:\caminho\imagem.jpg"
# se a foto sair deitada: python testar_api.py "C:\caminho\imagem.jpg" --rotacao 90

# Ver um erro 422
$ruim = @{ pixels = @(0, 1, 2); top_k = 11 } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/predict -Method Post -ContentType "application/json" -Body $ruim
```

## Limitações e cuidados

- O modelo foi treinado com imagens do MNIST (dígitos centralizados, fundo preto). Fotos reais exigem pré-processamento cuidadoso, e o `testar_api.py` aplica o mesmo usado no notebook.
- Classificadores fechados sempre devolvem uma classe, mesmo para entradas que não são dígitos. Use o campo `confiavel` com cautela: o MLP pode ter confiança altíssima em dados desconhecidos.
- Com `uvicorn --workers N`, cada processo carrega a sua própria cópia do modelo na RAM.

## Tecnologias

Python, scikit-learn, NumPy, pandas, matplotlib, seaborn, joblib, FastAPI, Uvicorn, Pydantic e pytest.
