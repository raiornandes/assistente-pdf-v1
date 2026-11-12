import re
import shutil
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pathlib import Path

import config

def carregar_paginas(caminho):
    leitor = PdfReader(str(caminho))
    return [
        Document(page_content=pagina.extract_text() or "", metadata={"page": i})
        for i, pagina in enumerate(leitor.pages)
    ]

def chunk_generico(paginas, cfg, fonte):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["tamanho"], chunk_overlap=cfg["overlap"]
    )
    chunks = []
    for c in splitter.split_documents(paginas):
        pagina = c.metadata.get("page", 0) + 1
        chunks.append(Document(
            page_content=c.page_content,
            metadata={"fonte": fonte, "secao": f"p.{pagina}", "pagina": pagina},
        ))
    return chunks


def chunk_estruturado(texto, cfg, fonte):
    tam = cfg["tamanho"]
    sub = RecursiveCharacterTextSplitter(chunk_size=tam, chunk_overlap=cfg["overlap"])
    chunks = []

    def adicionar(conteudo, secao):
        chunks.append(Document(page_content=conteudo,
                               metadata={"fonte": fonte, "secao": secao}))

    for bloco in re.split(cfg["padrao_secao"], texto):
        m = re.match(cfg["rotulo_secao"], bloco)
        secao = m.group(1) if m else "inicio"

        # início do doc
        if not m or len(bloco) <= tam:
            adicionar(bloco, secao)
            continue

        # divide em unidades (incisos...) ou, sem padrão por tamanho
        if cfg.get("padrao_unidade"):
            unidades = re.split(cfg["padrao_unidade"], bloco)
        else:
            unidades = sub.split_text(bloco)

        # a primeira unidade vira cabeçalho de todos os pedaços, se for curta
        if len(unidades) > 1 and len(unidades[0]) <= 300:
            cabecalho, corpo = unidades[0], unidades[1:]
        else:
            cabecalho, corpo = f"{cfg.get('prefixo', 'Seção')} {secao}", unidades

        atual = ""
        for u in corpo:
            if atual and len(cabecalho) + len(atual) + len(u) > tam:
                adicionar(cabecalho + "\n" + atual, secao)
                atual = ""
            atual += ("\n" if atual else "") + u
        if atual:
            adicionar(cabecalho + "\n" + atual, secao)
    return chunks


def indexar(nome: str, cfg: dict, caminho_pdf: Path) -> int:
    """Indexa um PDF numa coleção e devolve quantos chunks foram gravados."""
    paginas = carregar_paginas(caminho_pdf)
    if not any(p.page_content.strip() for p in paginas):
        raise ValueError("Não encontrei texto neste PDF (parece escaneado).")

    fonte = caminho_pdf.name
    if cfg["estrategia"] == "estruturado":
        texto = "\n".join(p.page_content for p in paginas)
        chunks = chunk_estruturado(texto, cfg, fonte)
    else:
        chunks = chunk_generico(paginas, cfg, fonte)

    for i, c in enumerate(chunks):
        c.metadata["ordem"] = i

    destino = config.PASTA_BANCOS / nome
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=config.MODELO_EMBEDDING),
        persist_directory=str(destino),
    )
    return vs._collection.count()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python indexar.py <colecao> [caminho.pdf]")
    nome = sys.argv[1]
    cfg = dict(config.COLECOES.get(nome, config.PADRAO))
    if len(sys.argv) > 2:
        cfg["pdf"] = sys.argv[2]
    if "pdf" not in cfg:
        raise SystemExit("Coleção nova: informe o PDF.")
    n = indexar(nome, cfg, config.BASE / cfg["pdf"])
    print(f"Coleção: {nome} | estratégia: {cfg['estrategia']} | indexados: {n}")


if __name__ == "__main__":
    main()
