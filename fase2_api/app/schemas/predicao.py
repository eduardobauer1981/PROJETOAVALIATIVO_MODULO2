"""Modelos Pydantic: validação estrita do JSON de entrada e de saída."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.config import NUM_PIXELS

# Cada pixel: número finito entre 0 (preto) e 255 (branco)
Pixel = Annotated[float, Field(ge=0, le=255, allow_inf_nan=False)]


class PredicaoRequest(BaseModel):
    """Corpo da requisição: imagem 28x28 achatada em 784 valores."""

    model_config = ConfigDict(
        extra="forbid",  # rejeita campos não previstos
        json_schema_extra={"examples": [{"pixels": [0] * NUM_PIXELS}]},
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


class PredicaoResponse(BaseModel):
    """Corpo da resposta da predição."""

    digito_previsto: int = Field(..., ge=0, le=9, description="Dígito previsto (0 a 9).")
    confianca: float = Field(..., ge=0, le=1, description="Probabilidade da classe prevista.")
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
