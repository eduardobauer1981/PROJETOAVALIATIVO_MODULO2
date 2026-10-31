"""Contratos de dados da API (Pydantic): o que ela ACEITA e o que ela DEVOLVE.

Tudo que não respeitar estas regras é barrado pelo Pydantic com HTTP 422,
antes de chegar ao serviço do modelo.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.config import NUM_PIXELS

# Cada pixel: número finito entre 0 (preto) e 255 (branco)
Pixel = Annotated[float, Field(ge=0, le=255, allow_inf_nan=False)]


# ----------------------------------------------------------------------------
# ENTRADA
# ----------------------------------------------------------------------------
class PredicaoRequest(BaseModel):
    """Corpo da requisição de predição."""

    model_config = ConfigDict(
        extra="forbid",  # rejeita campos não previstos no contrato
        strict=True,     # sem coerção: "128" (texto) NÃO vira número
        json_schema_extra={
            "examples": [
                {
                    "identificador": "digito_01.png",
                    "pixels": [0] * NUM_PIXELS,
                    "limiar_confianca": 0.5,
                    "top_k": 3,
                }
            ]
        },
    )

    pixels: list[Pixel] = Field(
        ...,
        min_length=NUM_PIXELS,
        max_length=NUM_PIXELS,
        description=(
            "Imagem 28x28 achatada linha a linha em 784 valores de 0 a 255, "
            "com fundo preto (0) e traço branco (255), como no MNIST."
        ),
    )
    identificador: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="Texto opcional (1 a 50 caracteres) para identificar a imagem, ex.: nome do arquivo.",
    )
    limiar_confianca: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description=(
            "Regra de negócio: se a confiança da previsão for menor que este valor, "
            "a resposta marca 'confiavel' como falso. Aceita valores maiores que 0 e até 1."
        ),
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Quantas previsões mais prováveis devolver (de 1 a 10).",
    )


# ----------------------------------------------------------------------------
# SAÍDA
# ----------------------------------------------------------------------------
class PrevisaoClasse(BaseModel):
    """Uma candidata de previsão: dígito e sua probabilidade."""

    digito: int = Field(..., ge=0, le=9, description="Dígito (0 a 9).")
    probabilidade: float = Field(..., ge=0, le=1, description="Probabilidade atribuída ao dígito.")


class PredicaoResponse(BaseModel):
    """Corpo da resposta da predição."""

    identificador: str | None = Field(None, description="Eco do identificador enviado, se houver.")
    digito_previsto: int = Field(..., ge=0, le=9, description="Dígito previsto (0 a 9).")
    confianca: float = Field(..., ge=0, le=1, description="Probabilidade da classe prevista.")
    confiavel: bool = Field(
        ..., description="Verdadeiro se a confiança atingiu o limiar_confianca da requisição."
    )
    melhores_previsoes: list[PrevisaoClasse] = Field(
        ..., min_length=1, max_length=10, description="As top_k previsões, da mais para a menos provável."
    )
    probabilidades: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Probabilidade de cada dígito, na ordem 0 a 9.",
    )


class SaudeResponse(BaseModel):
    """Resposta do endpoint de verificação de saúde."""

    status: str
    modelo_carregado: bool
