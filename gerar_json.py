"""Gera o corpo JSON do POST /v1/predict a partir de uma imagem (para colar no /docs).

Usa o mesmo pré-processamento da interface web (interface/preprocessamento.py):
corrige a orientação da foto, ignora sombras e mesa, centraliza e engrossa o traço.

Uso (na pasta do projeto):
    python gerar_json.py "C:\\caminho\\imagem.jpg" | clip
    python gerar_json.py imagem.jpg --top-k 5 --inverter auto --engrossar 1
"""

import argparse
import json

from PIL import Image, UnidentifiedImageError

from interface.preprocessamento import SemTracoError, preparar_imagem


def main():
    parser = argparse.ArgumentParser(description="Converte uma imagem no JSON esperado pela API.")
    parser.add_argument("imagem")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--inverter", choices=["auto", "sim", "nao"], default="auto")
    parser.add_argument("--rotacao", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--engrossar", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    try:
        pixels, _ = preparar_imagem(
            Image.open(args.imagem), inverter=args.inverter, rotacao=args.rotacao, engrossar=args.engrossar
        )
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        raise SystemExit(f"Erro: não consegui abrir '{args.imagem}' como imagem. Confira o caminho e o formato.")
    except SemTracoError as erro:
        raise SystemExit(f"Erro: {erro}")

    print(json.dumps({"pixels": pixels, "top_k": args.top_k}))


if __name__ == "__main__":
    main()
