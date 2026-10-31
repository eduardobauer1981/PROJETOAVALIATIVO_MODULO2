"""
testar_api.py
Envia uma imagem de dígito para a API e mostra a resposta.
(Pelo /docs seria preciso digitar 784 números à mão; este script faz isso por você.)

Uso (com a API rodando):
    python testar_api.py "C:\\caminho\\imagem.jpg"
    python testar_api.py "C:\\caminho\\imagem.jpg" --rotacao 90
"""

import argparse
import json
import urllib.error
import urllib.request

from PIL import Image

URL = "http://127.0.0.1:8000/predicao/digito"


def imagem_para_pixels(caminho: str, rotacao: int = 0) -> list[int]:
    """Mesmo pré-processamento do notebook: cinza, 28x28 e cores invertidas."""
    img = Image.open(caminho).convert("L")
    if rotacao:
        img = img.rotate(rotacao, expand=True)
    img = img.resize((28, 28), Image.LANCZOS)
    return [255 - p for p in img.tobytes()]  # fundo preto, traço branco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagem")
    parser.add_argument("--rotacao", type=int, default=0)
    args = parser.parse_args()

    corpo = json.dumps({"pixels": imagem_para_pixels(args.imagem, args.rotacao)}).encode()
    requisicao = urllib.request.Request(
        URL, data=corpo, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(requisicao) as resposta:
            dados = json.loads(resposta.read())
    except urllib.error.HTTPError as erro:
        print(f"Erro {erro.code}: {erro.read().decode()}")
        return
    except urllib.error.URLError:
        print("Não consegui conectar. A API está rodando? (uvicorn app.main:app --reload)")
        return

    print(f"Dígito previsto: {dados['digito_previsto']}  (confiança: {dados['confianca'] * 100:.1f}%, confiável: {'sim' if dados['confiavel'] else 'não'})")
    for digito, p in enumerate(dados["probabilidades"]):
        print(f"  {digito}: {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
