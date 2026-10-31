"""Interface web: envie uma imagem (ou tire uma foto) e veja o dígito reconhecido.

Uso (com a API rodando em outro terminal):
    python -m streamlit run interface/app_streamlit.py
"""

import os

import pandas as pd
import streamlit as st
from PIL import Image

from cliente_api import ApiIndisponivelError, ApiRespondeuErroError, consultar_health, prever
from preprocessamento import SemTracoError, preparar_imagem

st.set_page_config(page_title="Reconhecimento de Dígitos", page_icon="🔢", layout="wide")

# ----------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("⚙️ Configurações")
    url_api = st.text_input("Endereço da API", os.getenv("API_URL", "http://127.0.0.1:8000"))

    saude = consultar_health(url_api)
    if saude is None:
        st.error("API fora do ar. Suba o servidor com:\n\n`python -m uvicorn app.main:app --reload`")
    elif saude.get("modelo_carregado"):
        st.success(f"API ativa • modelo {saude.get('modelo_tipo')} na memória")
    else:
        st.warning("API ativa, mas o modelo NÃO está carregado (modo degradado).")

    st.subheader("Predição")
    limiar = st.slider("Limiar de confiança", 0.01, 1.0, 0.5, 0.01,
                       help="Abaixo deste valor a resposta é marcada como não confiável.")
    top_k = st.slider("Quantas previsões mostrar", 1, 10, 3)

    st.subheader("Preparo da imagem")
    inverter = st.radio(
        "Inverter cores", ["auto", "sim", "nao"], horizontal=True,
        help="O modelo espera traço claro sobre fundo escuro. 'auto' detecta fundo claro.",
    )
    engrossar = st.slider("Engrossar o traço", 0, 2, 0, help="Ajuda com canetas finas.")
    rotacao = st.selectbox("Girar imagem (graus, anti-horário)", [0, 90, 180, 270])

# --------------------------------------------------------------------- conteúdo
st.title("🔢 Reconhecimento de Dígitos Manuscritos")
st.write("Envie a foto de um dígito (0 a 9) e a rede neural vai dizer qual é.")

aba_arquivo, aba_camera = st.tabs(["📁 Enviar imagem", "📷 Usar câmera"])
with aba_arquivo:
    arquivo = st.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"])
with aba_camera:
    foto = st.camera_input("Tire uma foto do dígito")

fonte = arquivo or foto
if fonte is None:
    st.info("Dica: escreva um único dígito grande, com caneta grossa, em papel branco e boa iluminação.")
    st.stop()

imagem = Image.open(fonte)
try:
    pixels, previa = preparar_imagem(imagem, inverter=inverter, rotacao=rotacao, engrossar=engrossar)
except SemTracoError as erro:
    st.image(imagem, width=300)
    st.warning(f"{erro} Tente outra imagem, ajuste a inversão de cores ou a iluminação.")
    st.stop()

col_original, col_modelo = st.columns(2)
with col_original:
    st.subheader("Imagem original")
    st.image(imagem, width=300)
with col_modelo:
    st.subheader("O que o modelo vê (28x28)")
    st.image(previa.resize((224, 224), Image.Resampling.NEAREST), width=224)
    st.caption("Se o dígito não estiver nítido aqui, ajuste as opções de preparo na barra lateral.")

st.divider()
try:
    with st.spinner("Consultando a API..."):
        resultado = prever(url_api, pixels, limiar, top_k, getattr(fonte, "name", None))
except ApiIndisponivelError:
    st.error(f"Não consegui falar com a API em {url_api}. Ela está rodando?")
    st.stop()
except ApiRespondeuErroError as erro:
    if erro.status == 503:
        st.error("Modelo indisponível na API. Confira o endpoint /health.")
    elif erro.status in (400, 422):
        st.error(f"A API recusou os dados: {erro.detalhe}")
        for e in erro.erros:
            st.write(f"• **{e['campo']}**: {e['mensagem']}")
    else:
        st.error(f"Erro {erro.status}: {erro.detalhe}" + (f" (id do erro: {erro.id_erro})" if erro.id_erro else ""))
    st.stop()

col_a, col_b = st.columns(2)
col_a.metric("Dígito previsto", resultado["digito_previsto"])
col_b.metric("Confiança", f"{resultado['confianca']:.1%}")

if resultado["confiavel"]:
    st.success("Confiança acima do limiar escolhido.")
else:
    st.warning(
        "Confiança abaixo do limiar: trate esta previsão com cautela. "
        "O modelo pode estar errado mesmo com confiança alta em imagens muito diferentes dos dígitos do MNIST."
    )

st.subheader("Probabilidade por dígito")
st.bar_chart(pd.DataFrame({"Probabilidade": resultado["probabilidades"]},
                          index=[str(i) for i in range(10)]))

st.subheader(f"As {len(resultado['melhores_previsoes'])} previsões mais prováveis")
st.dataframe(
    pd.DataFrame(resultado["melhores_previsoes"]).rename(
        columns={"digito": "Dígito", "probabilidade": "Probabilidade"}
    ),
    hide_index=True,
)
