"""Pré-processamento de fotos/desenhos de dígitos no estilo do MNIST.

O modelo foi treinado com imagens 28x28, fundo preto, traço branco, dígito
centralizado numa caixa de 20x20 e com traço grosso (cerca de 13% da altura).
Esta função aproxima fotos reais dessas condições, mesmo com sombras, papel
cinza e mesa escura aparecendo na foto (sem depender do FastAPI nem do modelo).

Etapas:
 1. corrige a orientação (EXIF) e reduz fotos muito grandes;
 2. mede quanta "tinta" há em cada pixel comparando com o papel ao redor
    (assim sombra e iluminação desigual não viram traço);
 3. isola o dígito: descarta regiões coladas na borda (mesa, sombra) e ruídos;
 4. engrossa canetas finas até a espessura típica do MNIST;
 5. ajusta a uma caixa de 20x20 e centraliza pelo centro de massa em 28x28.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage as ndi

TAMANHO = 28
CAIXA_DIGITO = 20           # o dígito é ajustado a uma caixa de 20x20 dentro dos 28x28
CENTRO = (TAMANHO - 1) / 2  # 13,5
LADO_MAX = 400              # fotos maiores são reduzidas (mais rápido e mais robusto)
PICO_MINIMO = 0.15          # tinta mínima para considerar que existe um traço
LIMITE_TRACO = 0.35         # fração do pico de tinta que conta como traço
FRACAO_COMPONENTE = 0.08    # descarta manchas menores que 8% do maior pedaço de traço
COBERTURA_MAXIMA = 0.45     # se "tinta" cobre mais que isso, as cores estão invertidas
ESPESSURA_ALVO = 0.13       # espessura do traço / tamanho do dígito (proporção do MNIST)


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


def _impar(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _tinta(base: np.ndarray) -> np.ndarray:
    """Quantidade de tinta (0 a 1) por pixel, para imagens com papel CLARO e tinta escura.

    Cada pixel é comparado com o papel ao redor (filtro de máximo suavizado). Assim uma
    sombra no papel continua sendo "papel", e só o que é mais escuro que a vizinhança vira tinta.
    """
    janela = _impar(max(15, max(base.shape) // 5))
    papel = ndi.maximum_filter(base, size=janela)
    papel = ndi.gaussian_filter(papel, sigma=janela / 3)
    tinta = np.clip(1.0 - base / np.maximum(papel, 1.0), 0.0, 1.0)
    return ndi.gaussian_filter(tinta, sigma=0.7)


def _analisar(base: np.ndarray):
    """Devolve (tinta normalizada pelo pico, cobertura) ou None se não houver traço."""
    tinta = _tinta(base)
    pico = float(tinta.max())
    if pico < PICO_MINIMO:
        return None
    norm = tinta / pico
    return norm, float((norm > LIMITE_TRACO).mean())


def _isolar_digito(norm: np.ndarray) -> np.ndarray:
    """Mantém só o(s) pedaço(s) de traço que formam o dígito."""
    mascara = norm > LIMITE_TRACO
    rotulos, n = ndi.label(mascara, structure=np.ones((3, 3)))
    if n == 0:
        raise SemTracoError("Não encontrei um traço na imagem.")
    areas = ndi.sum(mascara, rotulos, index=np.arange(1, n + 1))

    # Regiões coladas na borda da imagem costumam ser mesa ou sombra, não o dígito.
    na_borda = set(np.unique(np.concatenate([rotulos[0], rotulos[-1], rotulos[:, 0], rotulos[:, -1]]))) - {0}
    candidatos = [i for i in range(1, n + 1) if i not in na_borda] or list(range(1, n + 1))

    maior = max(areas[i - 1] for i in candidatos)
    manter = [i for i in candidatos if areas[i - 1] >= FRACAO_COMPONENTE * maior]
    regiao = ndi.binary_dilation(np.isin(rotulos, manter), iterations=2)  # inclui a borda suave do traço
    return np.where(regiao, norm, 0.0)


def _espessura(mascara: np.ndarray) -> float:
    """Espessura média do traço: 2 x área / perímetro (vale para traços finos e compridos)."""
    perimetro = (mascara & ~ndi.binary_erosion(mascara)).sum()
    return 2.0 * mascara.sum() / max(int(perimetro), 1)


def preparar_imagem(
    img: Image.Image,
    inverter: str = "auto",
    rotacao: int = 0,
    engrossar: int = 0,
) -> tuple[list[int], Image.Image]:
    """Converte uma imagem qualquer em 784 pixels (0-255) no formato do MNIST.

    inverter:  "auto" (detecta sozinho qual é o fundo), "sim" (papel claro com tinta
               escura) ou "nao" (fundo escuro com traço claro, como o MNIST).
    rotacao:   graus no sentido anti-horário (0, 90, 180, 270), além da correção
               automática de orientação das fotos de celular (EXIF).
    engrossar: 0 a 2 passadas extras para engrossar o traço (além do ajuste automático).

    Retorna (lista de 784 inteiros, imagem 28x28 do que o modelo verá).
    Levanta SemTracoError se não houver um traço visível.
    """
    img = ImageOps.exif_transpose(img).convert("L")
    if rotacao:
        img = img.rotate(rotacao, expand=True)
    img.thumbnail((LADO_MAX, LADO_MAX), Image.Resampling.LANCZOS)
    cinza = np.asarray(img, dtype=np.float32)

    # 1) Polaridade: papel claro com tinta escura (foto) ou fundo escuro com traço claro (MNIST)?
    opcoes = {True: cinza, False: 255.0 - cinza}  # chave: "papel claro?"
    if inverter == "sim":
        papel_claro = True
    elif inverter == "nao":
        papel_claro = False
    else:
        # A polaridade certa é a que deixa MENOS área coberta de tinta (o traço é fino).
        analises = {p: _analisar(b) for p, b in opcoes.items()}
        validas = {p: a for p, a in analises.items() if a is not None}
        if not validas:
            raise SemTracoError("Não encontrei um traço na imagem (contraste muito baixo).")
        papel_claro = min(validas, key=lambda p: validas[p][1])

    resultado = _analisar(opcoes[papel_claro])
    if resultado is None:
        raise SemTracoError("Não encontrei um traço na imagem (contraste muito baixo).")
    norm, cobertura = resultado
    if cobertura > COBERTURA_MAXIMA:
        raise SemTracoError(
            "A imagem parece estar com as cores invertidas ou sem um fundo uniforme. "
            "Tente mudar a opção de inversão de cores."
        )

    # 2) Isola o dígito e recorta
    digito = _isolar_digito(norm)
    linhas, colunas = np.where(digito > 0)
    recorte = digito[linhas.min() : linhas.max() + 1, colunas.min() : colunas.max() + 1]
    lado = max(recorte.shape)

    # 3) Engrossa canetas finas até a proporção do MNIST (ainda em alta resolução)
    espessura = _espessura(recorte > LIMITE_TRACO)
    raio = int(round((ESPESSURA_ALVO * lado - espessura) / 2))
    if raio >= 1:
        recorte = np.pad(recorte, raio)
        recorte = ndi.grey_dilation(recorte, size=(2 * raio + 1, 2 * raio + 1))

    # 4) Ajusta à caixa de 20x20 mantendo a proporção
    h, w = recorte.shape
    escala = CAIXA_DIGITO / max(h, w)
    novo_w, novo_h = max(1, round(w * escala)), max(1, round(h * escala))
    miniatura = Image.fromarray((np.clip(recorte, 0, 1) * 255).astype(np.uint8)).resize(
        (novo_w, novo_h), Image.Resampling.LANCZOS
    )
    for _ in range(max(0, min(engrossar, 2))):
        miniatura = miniatura.filter(ImageFilter.MaxFilter(3))

    # 5) Coloca no centro de uma tela 28x28 e centraliza pelo centro de massa
    tela = np.zeros((TAMANHO, TAMANHO), dtype=np.float32)
    topo_y, esq_x = (TAMANHO - novo_h) // 2, (TAMANHO - novo_w) // 2
    tela[topo_y : topo_y + novo_h, esq_x : esq_x + novo_w] = np.asarray(miniatura, dtype=np.float32)

    ys, xs = np.indices(tela.shape)
    massa = tela.sum()
    cy, cx = (ys * tela).sum() / massa, (xs * tela).sum() / massa
    tela = _deslocar(tela, int(round(CENTRO - cy)), int(round(CENTRO - cx)))

    tela = tela / tela.max() * 255.0  # o traço mais forte vira 255, como no MNIST
    final = np.clip(tela, 0, 255).astype(np.uint8)
    return final.flatten().astype(int).tolist(), Image.fromarray(final)
