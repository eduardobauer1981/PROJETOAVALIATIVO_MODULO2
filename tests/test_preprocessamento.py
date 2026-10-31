"""Testes do pré-processamento da interface (fotos/desenhos -> formato MNIST)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from interface.preprocessamento import SemTracoError, preparar_imagem


def desenhar(fundo=255, traco=0, tamanho=300, pontos=((80, 60), (220, 60), (140, 240)), largura=18):
    """Imagem sintética com um traço em forma de '7'."""
    img = Image.new("L", (tamanho, tamanho), fundo)
    ImageDraw.Draw(img).line(pontos, fill=traco, width=largura)
    return img


def centro_de_massa(pixels):
    tela = np.array(pixels, dtype=float).reshape(28, 28)
    ys, xs = np.indices(tela.shape)
    return (ys * tela).sum() / tela.sum(), (xs * tela).sum() / tela.sum()


def test_saida_tem_784_pixels_inteiros_na_faixa_0_255():
    pixels, previa = preparar_imagem(desenhar())

    assert len(pixels) == 784
    assert all(isinstance(p, int) and 0 <= p <= 255 for p in pixels)
    assert previa.size == (28, 28)


def test_fundo_claro_e_invertido_automaticamente_e_o_fundo_fica_preto():
    pixels, _ = preparar_imagem(desenhar(fundo=255, traco=0))
    tela = np.array(pixels).reshape(28, 28)

    assert tela[0, 0] == 0 and tela[27, 27] == 0  # cantos pretos
    assert tela.max() > 200                       # traço claro


def test_fundo_claro_e_escuro_dao_o_mesmo_resultado():
    claro, _ = preparar_imagem(desenhar(fundo=255, traco=0))
    escuro, _ = preparar_imagem(desenhar(fundo=0, traco=255))

    diferenca = np.abs(np.array(claro) - np.array(escuro)).mean()
    assert diferenca < 5


def test_inversao_manual_errada_avisa_que_nao_ha_traco():
    # fundo escuro + traço claro, mas forçando inverter: o fundo vira claro e o traço some
    with pytest.raises(SemTracoError):
        preparar_imagem(desenhar(fundo=0, traco=255), inverter="sim")
    # fundo claro + traço escuro, mas dizendo que NÃO precisa inverter: mesma situação
    with pytest.raises(SemTracoError):
        preparar_imagem(desenhar(fundo=255, traco=0), inverter="nao")


def test_inversao_manual_correta_funciona():
    forcado, _ = preparar_imagem(desenhar(fundo=255, traco=0), inverter="sim")
    automatico, _ = preparar_imagem(desenhar(fundo=255, traco=0), inverter="auto")
    assert forcado == automatico


@pytest.mark.parametrize("posicao", [((10, 10), (60, 10), (30, 90)), ((200, 200), (290, 200), (250, 290))])
def test_centraliza_pelo_centro_de_massa_mesmo_com_traco_em_qualquer_canto(posicao):
    pixels, _ = preparar_imagem(desenhar(pontos=posicao))
    cy, cx = centro_de_massa(pixels)

    assert abs(cy - 13.5) <= 1.5
    assert abs(cx - 13.5) <= 1.5


def test_digito_ocupa_caixa_de_20x20_no_maximo():
    pixels, _ = preparar_imagem(desenhar())
    tela = np.array(pixels).reshape(28, 28)
    linhas, colunas = np.where(tela > 0)

    assert linhas.max() - linhas.min() + 1 <= 22  # 20 + folga do deslocamento pelo centro de massa
    assert colunas.max() - colunas.min() + 1 <= 22


def test_ruido_de_fundo_leve_e_ignorado():
    rng = np.random.default_rng(1)
    base = np.array(desenhar(), dtype=float)
    ruidosa = Image.fromarray(np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8))

    pixels, _ = preparar_imagem(ruidosa)
    tela = np.array(pixels).reshape(28, 28)
    assert tela[0, 0] == 0 and tela[27, 27] == 0


def test_engrossar_aumenta_a_area_do_traco():
    fino, _ = preparar_imagem(desenhar(largura=6), engrossar=0)
    grosso, _ = preparar_imagem(desenhar(largura=6), engrossar=2)
    assert (np.array(grosso) > 0).sum() > (np.array(fino) > 0).sum()


def test_imagem_em_branco_levanta_sem_traco():
    with pytest.raises(SemTracoError):
        preparar_imagem(Image.new("L", (200, 200), 255))
    with pytest.raises(SemTracoError):
        preparar_imagem(Image.new("L", (200, 200), 0))


def test_rotacao_gira_a_imagem():
    sem, _ = preparar_imagem(desenhar())
    com, _ = preparar_imagem(desenhar(), rotacao=90)
    assert sem != com


def test_orientacao_exif_de_foto_de_celular_e_corrigida():
    deitada = desenhar()
    exif = deitada.getexif()
    exif[274] = 6  # Orientation: precisa girar 90° para ficar em pé
    corrigida, _ = preparar_imagem(deitada)

    manual, _ = preparar_imagem(desenhar().rotate(-90, expand=True))
    assert np.abs(np.array(corrigida) - np.array(manual)).mean() < 5


def test_aceita_imagens_coloridas_e_com_canal_alfa():
    colorida = desenhar().convert("RGB")
    com_alfa = desenhar().convert("RGBA")
    assert len(preparar_imagem(colorida)[0]) == 784
    assert len(preparar_imagem(com_alfa)[0]) == 784


# ---------------------------------------------------------------------------
# Cenários de foto real (regressão): sombra, mesa escura, caneta fina, foto enorme
# ---------------------------------------------------------------------------
def espessura(pixels):
    from interface.preprocessamento import _espessura

    return _espessura(np.array(pixels).reshape(28, 28) > 90)


def test_sombra_no_papel_nao_vira_traco():
    uniforme, _ = preparar_imagem(desenhar())
    base = np.array(desenhar(), dtype=float)
    com_sombra = Image.fromarray((base * np.linspace(0.35, 1.0, 300)[None, :]).astype(np.uint8))

    resultado, _ = preparar_imagem(com_sombra)
    assert np.abs(np.array(resultado) - np.array(uniforme)).mean() < 3


def test_mesa_escura_na_borda_da_foto_e_ignorada():
    uniforme, _ = preparar_imagem(desenhar())
    base = np.array(desenhar(), dtype=float)
    base[:40, :] = 15  # faixa preta no topo, como a mesa na imagem_b
    base *= np.linspace(0.4, 1.0, 300)[None, :]  # e ainda com sombra

    resultado, _ = preparar_imagem(Image.fromarray(base.astype(np.uint8)))
    assert np.abs(np.array(resultado) - np.array(uniforme)).mean() < 3


@pytest.mark.parametrize("largura", [3, 18, 40])
def test_espessura_do_traco_fica_proxima_da_do_mnist(largura):
    pixels, _ = preparar_imagem(desenhar(largura=largura))
    assert 2.0 <= espessura(pixels) <= 4.5


def test_foto_enorme_e_processada_rapido():
    import time

    grande = desenhar(tamanho=3000, pontos=((800, 600), (2200, 600), (1400, 2400)), largura=180)
    inicio = time.perf_counter()
    pixels, _ = preparar_imagem(grande)

    assert time.perf_counter() - inicio < 3
    assert len(pixels) == 784


def test_cores_invertidas_manualmente_geram_erro_em_vez_de_lixo():
    with pytest.raises(SemTracoError):
        preparar_imagem(desenhar(), inverter="nao")


FOTOS = ["imagem_a.jpg", "imagem_b.jpg", "imagem_c.jpg", "imagem_ajustada.jpg"]


@pytest.mark.parametrize("nome", FOTOS)
def test_fotos_reais_viram_digitos_limpos_e_centralizados(nome):
    caminho = Path(__file__).parent / "fixtures" / nome
    pixels, _ = preparar_imagem(Image.open(caminho))
    tela = np.array(pixels).reshape(28, 28)

    # fundo preto: os quatro cantos estão vazios e o traço não "vaza" pela imagem
    for cantos in (tela[:4, :4], tela[:4, -4:], tela[-4:, :4], tela[-4:, -4:]):
        assert cantos.max() == 0
    assert (tela > 0).mean() < 0.35

    # dígito centralizado, ocupando a caixa de 20x20 e com traço de espessura parecida com a do MNIST
    cy, cx = centro_de_massa(pixels)
    assert abs(cy - 13.5) <= 1.5 and abs(cx - 13.5) <= 1.5
    linhas, colunas = np.where(tela > 0)
    assert max(linhas.max() - linhas.min(), colunas.max() - colunas.min()) + 1 >= 18
    assert 2.0 <= espessura(pixels) <= 4.5
    assert tela.max() == 255
