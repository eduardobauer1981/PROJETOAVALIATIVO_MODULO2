"""Configurações da aplicação (caminhos e constantes)."""

import os
from pathlib import Path

# Raiz do projeto = pasta que contém "app/" e "models/"
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Caminho do modelo serializado. Pode ser sobrescrito pela variável de ambiente MODELO_PATH.
CAMINHO_MODELO = Path(
    os.getenv("MODELO_PATH", RAIZ_PROJETO / "models" / "mlp_mnist.joblib")
)

NUM_PIXELS = 784  # 28 x 28
