"""Contrato da resposta de erro de validação (HTTP 422)."""

from pydantic import BaseModel, Field


class ErroResponse(BaseModel):
    """Formato único dos erros que não são de validação (400, 404, 405, 500, 503...)."""

    detalhe: str = Field(..., description="O que aconteceu, em linguagem clara.")
    status: int = Field(..., ge=400, le=599, description="Código HTTP do erro.")
    id_erro: str | None = Field(
        None, description="Código para localizar o erro nos logs do servidor (presente nos erros 500)."
    )


class ErroCampo(BaseModel):
    campo: str = Field(..., description="Caminho do campo com problema, ex.: 'pixels.5' ou 'top_k'.")
    mensagem: str = Field(..., description="O que está errado.")
    tipo: str = Field(..., description="Código do erro do Pydantic, ex.: 'less_than_equal'.")


class ErroValidacaoResponse(BaseModel):
    detalhe: str = Field(..., description="Resumo do erro.")
    total_erros: int = Field(..., ge=1, description="Quantidade total de problemas encontrados.")
    erros: list[ErroCampo] = Field(..., description="Os primeiros problemas encontrados (no máximo 10).")
