from pathlib import Path

BASE = Path(__file__).resolve().parent  ## pasta do arquivo
PASTA_BANCOS = BASE / "bancos"

MODELO_EMBEDDING = "bge-m3"
MODELO_LLM = "llama3.1"
NUM_CTX = 8192

PADRAO = {
    "estrategia": "generico",
    "tamanho": 700,
    "overlap": 100,
    "dominio": "os documentos fornecidos",
}

COLECOES = {
    "lgpd": {
        **PADRAO,                      # copia o padrão e sobrescreve abaixo
        "pdf": "dados/lgpd.pdf",
        "dominio": "a LGPD (Lei Geral de Proteção de Dados)",
        "estrategia": "estruturado",
        "prefixo": "Art.",
        "padrao_secao": r"\n(?=Art\. \d+)",
        "rotulo_secao": r"Art\. (\d+(?:-[A-Z])?)",
        "padrao_unidade": r"\n(?=(?:[IVXL]+ - |§ ?\d+º|Parágrafo único))",
    },
}
