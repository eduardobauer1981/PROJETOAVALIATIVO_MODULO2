"""
testar_api.py
Envia uma imagem de dígito para a API e mostra a resposta.
Usa o mesmo pré-processamento da interface web (interface/preprocessamento.py).

Uso (com a API rodando, na pasta do projeto):
    python testar_api.py "C:\\caminho\\imagem.jpg"
    python testar_api.py imagem.jpg --rotacao 90 --engrossar 1
"""

import argparse
import json
import urllib.error
import urllib.request

from PIL import Image, UnidentifiedImageError

from interface.preprocessamento import SemTracoError, preparar_imagem

URL = "http://127.0.0.1:8000/v1/predict"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagem")
    parser.add_argument("--rotacao", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--engrossar", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--inverter", choices=["auto", "sim", "nao"], default="auto")
    args = parser.parse_args()

    try:
        pixels, _ = preparar_imagem(
            Image.open(args.imagem), inverter=args.inverter, rotacao=args.rotacao, engrossar=args.engrossar
        )
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        print(f"Erro: não consegui abrir '{args.imagem}' como imagem. Confira o caminho e o formato.")
        return
    except SemTracoError as erro:
        print(f"Erro: {erro}")
        return

    corpo = json.dumps({"pixels": pixels}).encode()
    requisicao = urllib.request.Request(URL, data=corpo, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(requisicao) as resposta:
            dados = json.loads(resposta.read())
    except urllib.error.HTTPError as erro:
        print(f"Erro {erro.code}: {erro.read().decode()}")
        return
    except urllib.error.URLError:
        print("Não consegui conectar. A API está rodando? (python -m uvicorn app.main:app --reload)")
        return

    sim_nao = "sim" if dados["confiavel"] else "não"
    print(f"Dígito previsto: {dados['digito_previsto']}  (confiança: {dados['confianca'] * 100:.1f}%, confiável: {sim_nao})")
    for digito, p in enumerate(dados["probabilidades"]):
        print(f"  {digito}: {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
