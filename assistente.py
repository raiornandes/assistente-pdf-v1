import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

import config

TEMPLATE = """Você é um assistente especializado em {dominio}.
Responda a pergunta do usuário usando APENAS as informações do contexto abaixo.
Se a resposta não estiver no contexto, diga claramente que não encontrou essa informação nos documentos.
Liste os itens que estiverem presentes no contexto, em ordem, sem acrescentar nenhum que não esteja escrito nele.
Se a numeração pular, ou o texto disser "os seguintes" e a lista não aparecer no contexto, avise que o contexto está incompleto.
Nunca complete uma lista de memória.
Sempre que possível, cite a seção ou página em que se baseou.

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:"""


class Assistente:
    def __init__(self, colecao: str, dominio: str | None = None):
        cfg = {**config.PADRAO, **config.COLECOES.get(colecao, {})}
        self.dominio = dominio or cfg["dominio"]
        self.vectorstore = Chroma(
            persist_directory=str(config.PASTA_BANCOS / colecao),
            embedding_function=OllamaEmbeddings(model=config.MODELO_EMBEDDING),
        )
        if self.vectorstore._collection.count() == 0:
            raise SystemExit(f"Coleção '{colecao}' vazia ou inexistente. Rode: python indexar.py {colecao}")
        self.llm = OllamaLLM(model=config.MODELO_LLM, num_ctx=config.NUM_CTX)
        self.prompt = ChatPromptTemplate.from_template(TEMPLATE)

    def buscar_contexto(self, pergunta: str, k: int = 8, n_secoes: int = 2) -> list:
        encontrados = self.vectorstore.similarity_search(pergunta, k=k)

        # as N seções mais bem colocadas no ranking
        secoes = []
        for d in encontrados:
            s = d.metadata.get("secao")
            if s not in secoes and s != "inicio":
                secoes.append(s)
        secoes = secoes[:n_secoes]

        # traz TODOS os chunks dessas seções ordenados
        docs = []
        for s in secoes:
            dados = self.vectorstore.get(where={"secao": s})
            grupo = [Document(page_content=t, metadata=m)
                     for t, m in zip(dados["documents"], dados["metadatas"])]
            grupo.sort(key=lambda d: d.metadata["ordem"])
            docs.extend(grupo)

        # completando com os demais resultados
        ja = {d.page_content for d in docs}
        docs.extend(d for d in encontrados if d.page_content not in ja)
        return docs

    def perguntar(self, pergunta: str, k: int = 8) -> dict:
        docs = self.buscar_contexto(pergunta, k=k)
        contexto = "\n\n---\n\n".join(
            f"({d.metadata.get('fonte')}, {d.metadata.get('secao')}): {d.page_content}"
            for d in docs
        )
        resposta = self.llm.invoke(
            self.prompt.format(dominio=self.dominio, contexto=contexto, pergunta=pergunta)
        )
        return {"pergunta": pergunta, "resposta": resposta, "chunks_usados": docs}


if __name__ == "__main__":
    colecao = sys.argv[1] if len(sys.argv) > 1 else "lgpd"
    assistente = Assistente(colecao)
    for p in sys.argv[2:] or ["O que é considerado um dado pessoal sensível?"]:
        r = assistente.perguntar(p)
        print(f"\nPergunta: {r['pergunta']}\nResposta: {r['resposta']}\n{'-' * 80}")
