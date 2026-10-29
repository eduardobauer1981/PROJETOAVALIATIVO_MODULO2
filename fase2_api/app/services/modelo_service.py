"""Serviço de inferência: carrega o modelo serializado e executa a predição.

Não conhece FastAPI nem HTTP: recebe números, devolve números.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModeloNaoCarregadoError(RuntimeError):
    """Levantada ao tentar prever sem o modelo carregado."""


@dataclass(frozen=True)
class ResultadoPredicao:
    digito: int
    confianca: float
    probabilidades: list[float]  # índice = dígito (0 a 9)


class ModeloService:
    def __init__(self, caminho_modelo: Path):
        self._caminho = Path(caminho_modelo)
        self._modelo = None

    @property
    def carregado(self) -> bool:
        return self._modelo is not None

    def carregar(self) -> None:
        """Lê o arquivo .joblib para a memória (chamado uma vez, na inicialização)."""
        if not self._caminho.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {self._caminho}")
        self._modelo = joblib.load(self._caminho)
        logger.info("Modelo carregado de %s", self._caminho)

    def descarregar(self) -> None:
        self._modelo = None

    def prever(self, pixels: Sequence[float]) -> ResultadoPredicao:
        if self._modelo is None:
            raise ModeloNaoCarregadoError("O modelo ainda não foi carregado.")

        # Mesmo pré-processamento do treino: vetor (1, 784) normalizado para 0-1
        entrada = np.asarray(pixels, dtype=np.float32).reshape(1, -1) / 255.0

        proba = self._modelo.predict_proba(entrada)[0]

        # Garante a ordem 0..9 usando as classes que o modelo conhece
        completa = np.zeros(10, dtype=float)
        for classe, p in zip(self._modelo.classes_, proba):
            completa[int(classe)] = p

        digito = int(np.argmax(completa))
        return ResultadoPredicao(
            digito=digito,
            confianca=float(completa[digito]),
            probabilidades=[float(p) for p in completa],
        )
