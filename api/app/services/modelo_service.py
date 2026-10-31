"""Serviço de inferência: carrega o modelo serializado e executa a predição.

Não conhece FastAPI nem HTTP: recebe números, devolve números.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from app.config import NUM_PIXELS

logger = logging.getLogger(__name__)


class ModeloNaoCarregadoError(RuntimeError):
    """Levantada ao tentar prever sem o modelo carregado."""


@dataclass(frozen=True)
class ResultadoPredicao:
    digito: int
    confianca: float
    probabilidades: list[float]  # índice = dígito (0 a 9)

    def melhores(self, k: int) -> list[tuple[int, float]]:
        """Os k dígitos mais prováveis, do maior para o menor."""
        ordem = sorted(range(10), key=lambda d: self.probabilidades[d], reverse=True)[:k]
        return [(d, self.probabilidades[d]) for d in ordem]


class ModeloService:
    def __init__(self, caminho_modelo: Path):
        self._caminho = Path(caminho_modelo)
        self._modelo = None

    @property
    def carregado(self) -> bool:
        return self._modelo is not None

    def carregar(self) -> None:
        """Lê o arquivo .joblib para a RAM. Chamado UMA vez, no lifespan da aplicação.

        É idempotente: se o modelo já está na memória, não lê o disco de novo.
        """
        if self._modelo is not None:
            logger.info("Modelo já está em memória; nenhuma leitura de disco.")
            return
        if not self._caminho.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {self._caminho}")

        inicio = time.perf_counter()
        self._modelo = joblib.load(self._caminho)
        self._aquecer()
        logger.info(
            "Modelo carregado de %s em %.2f s", self._caminho, time.perf_counter() - inicio
        )

    def _aquecer(self) -> None:
        """Faz uma predição de teste na inicialização.

        Assim a primeira requisição real não paga o custo da 'primeira chamada' e,
        se o modelo for incompatível (ex.: número de features), o erro aparece
        na subida da API, e não na mão do usuário.
        """
        self._modelo.predict_proba(np.zeros((1, NUM_PIXELS), dtype=np.float32))

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
        completa = np.clip(completa, 0.0, 1.0)

        digito = int(np.argmax(completa))
        return ResultadoPredicao(
            digito=digito,
            confianca=float(completa[digito]),
            probabilidades=[float(p) for p in completa],
        )
