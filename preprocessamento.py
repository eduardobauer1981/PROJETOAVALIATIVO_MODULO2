"""Pré-processamento de fotos/desenhos de dígitos no estilo do MNIST.

O modelo foi treinado com imagens 28x28, fundo preto, traço branco, dígito
centralizado e ocupando uma caixa de 20x20. Esta função aproxima qualquer
imagem dessas condições (sem depender do FastAPI nem do modelo).
"""

import numpy as np
from PIL import Image, ImageFilter, ImageOps

TAMANHO = 28
CAIXA_DIGITO = 20  # o dígito é ajustado a uma caixa de 20x20 dentro dos 28x28
LIMITE_RUIDO = 0.25  # intensidades abaixo disso (sombras, textura do papel) viram fundo
CENTRO = (TAMANHO - 1) / 2  # 13,5


class SemTracoError(ValueError):
    """Levantada quando a imagem não tem um traço distinguível do fundo."""


def _deslocar(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Desloca a matriz preenchendo com zeros (sem 'dar a volta' como o np.roll)."""
    h, w = arr.shape
    saida = np.zeros_like(arr)
    destino_y = slice(max(dy, 0), h + min(dy, 0))
    origem_y = slice(max(-dy, 0), h + min(-dy, 0))
    destino_x = slice(max(dx, 0), w + min(dx, 0))
    origem_x = slice(max(-dx, 0), w + min(-dx, 0))
    saida[destino_y, destino_x] = arr[origem_y, origem_x]
    return saida


def _mediana_da_borda(arr: np.ndarray) -> float:
    borda = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.median(borda))


def preparar_imagem(
    img: Image.Image,
    inverter: str = "auto",
    rotacao: int = 0,
    engrossar: int = 0,
) -> tuple[list[int], Image.Image]:
    """Converte uma imagem qualquer em 784 pixels (0-255) no formato do MNIST.

    inverter:  "auto" (detecta fundo claro), "sim" ou "nao".
    rotacao:   graus no sentido anti-horário (0, 90, 180, 270), além da correção
               automática de orientação das fotos de celular (EXIF).
    engrossar: 0 a 2 passadas para engrossar o traço (canetas finas viram traço fraco).

    Retorna (lista de 784 inteiros, imagem 28x28 do que o modelo verá).
    Levanta SemTracoError se não houver um traço visível.
    """
    img = ImageOps.exif_transpose(img).convert("L")
    if rotacao:
        img = img.rotate(rotacao, expand=True)
    arr = np.asarray(img, dtype=np.float32)

    # 1) Dígito claro sobre fundo escuro (como no MNIST)
    if inverter == "sim":
        precisa_inverter = True
    elif inverter == "nao":
        precisa_inverter = False
    else:
        precisa_inverter = _mediana_da_borda(arr) > 127
    if precisa_inverter:
        arr = 255.0 - arr

    # 2) Normaliza o contraste a partir do nível do fundo e remove sombras/ruído
    fundo = _mediana_da_borda(arr)
    topo = float(np.percentile(arr, 99.5))
    if topo - fundo < 20:
        raise SemTracoError("Não encontrei um traço na imagem (contraste muito baixo).")
    arr = np.clip((arr - fundo) / (topo - fundo), 0.0, 1.0)
    arr[arr < LIMITE_RUIDO] = 0.0

    # 3) Recorta o dígito e ajusta à caixa de 20x20 mantendo a proporção
    linhas = np.where(arr.any(axis=1))[0]
    colunas = np.where(arr.any(axis=0))[0]
    if linhas.size == 0 or colunas.size == 0:
        raise SemTracoError("Não encontrei um traço na imagem.")
    recorte = arr[linhas[0] : linhas[-1] + 1, colunas[0] : colunas[-1] + 1]
    h, w = recorte.shape
    escala = CAIXA_DIGITO / max(h, w)
    novo_w, novo_h = max(1, round(w * escala)), max(1, round(h * escala))
    miniatura = Image.fromarray((recorte * 255).astype(np.uint8)).resize(
        (novo_w, novo_h), Image.Resampling.LANCZOS
    )
    for _ in range(max(0, min(engrossar, 2))):
        miniatura = miniatura.filter(ImageFilter.MaxFilter(3))

    # 4) Coloca no centro de uma tela 28x28 e centraliza pelo centro de massa
    tela = np.zeros((TAMANHO, TAMANHO), dtype=np.float32)
    topo_y, esq_x = (TAMANHO - novo_h) // 2, (TAMANHO - novo_w) // 2
    tela[topo_y : topo_y + novo_h, esq_x : esq_x + novo_w] = np.asarray(miniatura, dtype=np.float32)

    ys, xs = np.indices(tela.shape)
    massa = tela.sum()
    cy, cx = (ys * tela).sum() / massa, (xs * tela).sum() / massa
    tela = _deslocar(tela, int(round(CENTRO - cy)), int(round(CENTRO - cx)))

    final = np.clip(tela, 0, 255).astype(np.uint8)
    return final.flatten().astype(int).tolist(), Image.fromarray(final)
