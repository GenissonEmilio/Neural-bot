import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configurações ---
ARQUIVO_JSON_FATOS = 'fatos.json'
NOME_MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'
NOME_COLECAO = "fatos_bot"
CAMINHO_DB = "./chroma_db"

print("🤖 [Indexer] Iniciando script de indexação...")

# --- 1. Carregar Modelo de Embedding ---
print(f"🤖 [Indexer] Carregando modelo '{NOME_MODELO_EMBEDDING}'...")
try:
    model = SentenceTransformer(NOME_MODELO_EMBEDDING)
    print("🤖 [Indexer] Modelo carregado.")
except Exception as e:
    print(f"🚨 ERRO ao carregar modelo: {e}")
    exit()

# --- 2. Conectar ao ChromaDB ---
# PersistentClient salva os dados em disco no 'CAMINHO_DB'
try:
    client = chromadb.PersistentClient(path=CAMINHO_DB)

    # Recria (ou cria) a coleção. metadata especifica o método de cálculo (cosseno)
    # Isso é importante para que "distância" menor signifique "mais similar"
    collection = client.get_or_create_collection(
        name=NOME_COLECAO,
        metadata={"hnsw:space": "cosine"}  # Define a métrica de distância como cosseno
    )
    print(f"🤖 [Indexer] Conectado ao ChromaDB e coleção '{NOME_COLECAO}' pronta.")
except Exception as e:
    print(f"🚨 ERRO ao conectar ao ChromaDB: {e}")
    exit()

# --- 3. Carregar Fatos do JSON ---
try:
    with open(ARQUIVO_JSON_FATOS, 'r', encoding='utf-8') as f:
        fatos_lista = json.load(f)
    if not fatos_lista:
        print("🚨 ERRO: 'fatos.json' está vazio.")
        exit()
    print(f"🤖 [Indexer] Carregados {len(fatos_lista)} fatos do JSON.")
except Exception as e:
    print(f"🚨 ERRO ao ler 'fatos.json': {e}")
    exit()

# --- 4. Popular o Banco de Dados ---
print("🤖 [Indexer] Iniciando processo de embedding e indexação (pode demorar)...")

# Prepara os dados para o ChromaDB
ids = []
embeddings = []
documentos = []  # O texto que será retornado (a resposta)
metadados = []  # Dados extras (a pergunta original)

perguntas_fatos = [item['pergunta'] for item in fatos_lista]

try:
    # Gera TODOS os embeddings de uma vez (muito mais rápido)
    embeddings = model.encode(perguntas_fatos, show_progress_bar=True).tolist()

    for i, item in enumerate(fatos_lista):
        ids.append(f"fato_{i}")
        documentos.append(item['resposta'])
        metadados.append({"pergunta_original": item['pergunta']})

    # Adiciona tudo ao banco de dados em lote
    collection.add(
        embeddings=embeddings,
        documents=documentos,
        metadatas=metadados,
        ids=ids
    )

    print("\n✅ SUCESSO! Banco de dados vetorial populado.")
    print(f"Foram adicionados {collection.count()} documentos à coleção '{NOME_COLECAO}'.")

except Exception as e:
    print(f"🚨 ERRO durante a indexação: {e}")