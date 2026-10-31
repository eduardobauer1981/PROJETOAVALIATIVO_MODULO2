"""
apresentar_metricas_mlp.py
Script auxiliar (item 8.2 do projeto): apresenta as métricas do modelo de
Rede Neural (MLP) que já foram geradas no notebook.

Não treina, não carrega o MNIST e não recalcula nada: só lê o arquivo
reports/metricas_mlp.json e mostra o resultado no terminal.
Usa apenas a biblioteca padrão do Python (sem matplotlib, sem janelas).

Uso (na pasta do projeto):
    python apresentar_metricas_mlp.py
"""

import json
from pathlib import Path

CAMINHO_METRICAS = Path(__file__).resolve().parent / "reports" / "metricas_mlp.json"


def main():
    print(f"Lendo métricas de: {CAMINHO_METRICAS}\n")

    if not CAMINHO_METRICAS.exists():
        print("ERRO: arquivo de métricas não encontrado.")
        print("Rode antes, no notebook, a célula que salva as métricas em JSON.")
        return

    m = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))

    print("=" * 58)
    print(f"  8.2 TABELA DE MÉTRICAS - {m['modelo']} (Rede Neural)")
    print("=" * 58)
    print(f"{'Métrica':<34}{'Valor':>12}")
    print("-" * 46)
    print(f"{'Acurácia global':<34}{m['acuracia'] * 100:>11.2f}%")
    print(f"{'Precisão média ponderada':<34}{m['precisao_ponderada'] * 100:>11.2f}%")
    print(f"{'Revocação/Sensibilidade ponderada':<34}{m['recall_ponderado'] * 100:>11.2f}%")
    print(f"{'F1-Score ponderado':<34}{m['f1_ponderado'] * 100:>11.2f}%")
    print(f"{'Tempo de predição (teste todo)':<34}{m['tempo_predicao_s']:>11.3f}s")

    print("\nDesempenho por classe:")
    print(f"{'Dígito':<8}{'Precisão':>10}{'Recall':>10}{'F1-Score':>10}{'Amostras':>10}")
    print("-" * 48)
    for digito, r in m["por_classe"].items():
        print(f"{digito:<8}{r['precision']:>10.4f}{r['recall']:>10.4f}"
              f"{r['f1-score']:>10.4f}{int(r['support']):>10}")


if __name__ == "__main__":
    main()
