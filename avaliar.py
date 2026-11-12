import json
import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config


def avaliar(colecao: str):
    arquivo = config.BASE / "testes" / f"{colecao}.json"
    if not arquivo.exists():
        raise SystemExit(f"Sem arquivo de testes: {arquivo}")
    testes = json.loads(arquivo.read_text(encoding="utf-8"))

    vs = Chroma(
        persist_directory=str(config.PASTA_BANCOS / colecao),
        embedding_function=OllamaEmbeddings(model=config.MODELO_EMBEDDING),
    )
    total = vs._collection.count()
    if total == 0:
        raise SystemExit(f"Coleção '{colecao}' vazia. Rode: python indexar.py {colecao}")
    print(f"Coleção: {colecao} | modelo: {config.MODELO_EMBEDDING} | chunks: {total}\n")

    top4 = top8 = 0
    for t in testes:
        resultados = vs.similarity_search_with_score(t["pergunta"], k=total)
        posicao = next(
            (pos for pos, (d, _) in enumerate(resultados, start=1)
             if d.metadata.get("secao") in t["secoes"]),
            None,
        )
        top4 += posicao is not None and posicao <= 4
        top8 += posicao is not None and posicao <= 8
        print(f"  posição {str(posicao):>4} | seção {','.join(t['secoes']):>3} | {t['pergunta']}")

    print(f"\nResumo: top 4 = {top4}/{len(testes)} | top 8 = {top8}/{len(testes)}")


if __name__ == "__main__":
    avaliar(sys.argv[1] if len(sys.argv) > 1 else "lgpd")