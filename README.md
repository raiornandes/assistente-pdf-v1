# Assistente de PDF (v1)

Assistente que responde perguntas sobre um PDF, citando o trecho de onde tirou a resposta. Usa **RAG** (Retrieval-Augmented Generation) e roda **100% local**: sem API paga e sem enviar documentos para fora da máquina.

O projeto nasceu como um assistente da **LGPD** (Lei 13.709/2018) e foi generalizado para aceitar qualquer PDF com texto.

## Como funciona

```
PDF -> chunks -> embeddings (bge-m3) -> ChromaDB
                                           |
pergunta -> busca por similaridade -> expansão por seção -> llama3.1 -> resposta + trechos usados
```

- **Ingestão:** o texto do PDF é dividido em chunks. Há duas estratégias: `generico` (por tamanho, com overlap) e `estruturado` (por seção, para textos legais; cada chunk repete o cabeçalho do artigo).
- **Busca:** o `bge-m3` transforma chunks e pergunta em vetores, e o ChromaDB devolve os mais próximos.
- **Expansão por seção:** quando a busca acerta uma seção (ex.: Art. 52), o assistente entrega **todos** os chunks dela, em ordem. Isso resolveu respostas cortadas, como listas de incisos.
- **Geração:** o `llama3.1` (via Ollama) responde apenas com base no contexto recebido.
- **Interface:** app em Streamlit com upload de PDF, chat e a lista dos trechos usados em cada resposta.

## Como rodar

Pré-requisitos: Python 3.10+ e [Ollama](https://ollama.com/download).

```bash
# 1. Modelos locais
ollama pull llama3.1
ollama pull bge-m3

# 2. Ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 3. Interface
streamlit run app.py
```

Sem upload, o app usa a LGPD. Para ela funcionar, coloque o PDF da lei em `dados/lgpd.pdf` e indexe uma vez:

```bash
python indexar.py lgpd
```

Para outro documento pela linha de comando:

```bash
python indexar.py meu_manual dados/manual.pdf
python assistente.py meu_manual "Sua pergunta aqui"
```

## Decisões técnicas e resultados

A busca foi medida com 12 perguntas sobre a LGPD, cada uma com o artigo onde está a resposta (`testes/lgpd.json`). Rode com `python avaliar.py lgpd`.

| Versão | O que mudou | Resultado |
|---|---|---|
| v1 | Chunks de 800 caracteres | O Art. 7º ficava fatiado e os pedaços perdiam o rótulo do artigo; Art. 7º e Art. 52 ficavam fora do top 4 |
| v2 | Um chunk por artigo | Pior: chunks grandes perdiam no ranking para chunks curtos e genéricos |
| v3 | Chunk por inciso, com o cabeçalho do artigo repetido | Melhora forte. Com `nomic-embed-text`: **9/12** no top 4 |
| v3 + `bge-m3` | Troca só o modelo de embedding (mesmos chunks) | **12/12** no top 4 |
| + expansão por seção | Entrega todos os chunks da seção encontrada | Listas completas (Art. 7º, I a X; Art. 52, I a XII) |

Outras coisas que testei e **descartei**:
- Prefixos `search_query:` / `search_document:` do `nomic-embed-text`: posições levemente piores.
- Expandir o artigo "mais frequente" nos resultados: escolhia o artigo errado.

Aprendizados que viraram código:
- Caminhos relativos faziam o Chroma abrir um banco vazio sem avisar. Agora os caminhos partem de `Path(__file__)` e o app recusa coleção vazia.
- Reindexar duplicava chunks. O indexador agora apaga o banco antes de recriar.

## Limitações conhecidas

- **Conjunto de teste pequeno.** 12 perguntas mostram a diferença entre as versões, mas não provam que a busca acerta sempre. Além disso, três delas orientaram o desenho do chunking.
- **Chunking genérico não foi medido.** Só a LGPD tem avaliação. Com a LGPD enviada pelo upload (genérico), as respostas pioraram e houve itens que não estão na lei.
- **O modelo local (llama3.1 8B) às vezes falha mesmo com o contexto certo:** repete trechos, mistura critérios com itens de lista e, de vez em quando, completa com conhecimento próprio. O prompt manda não fazer isso, mas não garante.
- **PDFs escaneados** não funcionam (sem OCR). O app avisa que não encontrou texto.
- **O chunking estruturado** depende de regex para artigos e incisos; só serve para textos nesse formato.
- **Indexação lenta** em PDFs grandes, porque os embeddings rodam na máquina local.

## Roadmap (v2)

- Vários PDFs ao mesmo tempo na mesma consulta
- Mais tipos de consulta (resumo, comparação entre documentos)
- Conjunto de teste maior e avaliação também da resposta final, não só da busca
- Modelo de geração maior ou reranking
- OCR para PDFs escaneados

## Estrutura

```
app.py          interface Streamlit
assistente.py   busca, expansão por seção e geração
indexar.py      leitura do PDF, chunking e indexação
avaliar.py      mede a busca com testes/<colecao>.json
config.py       modelos, caminhos e configuração por coleção
testes/         perguntas e seções esperadas
```
