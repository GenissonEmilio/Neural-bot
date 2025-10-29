import time
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset  # <-- A MÁGICA ESTÁ AQUI

# --- Configurações ---
# Vamos usar a Wikipédia em Português!
# Você pode ver outras aqui: https://huggingface.co/datasets/wikipedia
NOME_DATASET_HF = "wikipedia"
CONFIG_DATASET_HF = "20220301.pt"  # 'pt' para português, 'en' para inglês

NOME_MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'
NOME_COLECAO = "fatos_bot"
CAMINHO_DB = "./chroma_db"

# Limite de artigos para processar (para não rodar para sempre no teste)
# Coloque None para processar a Wikipédia INTEIRA (milhões de artigos, pode levar dias!)
LIMITE_DE_ARTIGOS = 2000  # Comece com um número pequeno para testar

# --- Configurações do "Fatiador" (Chunker) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

print("🤖 [Ingestor Wiki] Iniciando script de ingestão...")

# --- 1. Carregar Modelo de Embedding ---
print(f"🤖 [Ingestor Wiki] Carregando modelo '{NOME_MODELO_EMBEDDING}'...")
try:
    model = SentenceTransformer(NOME_MODELO_EMBEDDING)
    print("🤖 [Ingestor Wiki] Modelo carregado.")
except Exception as e:
    print(f"🚨 ERRO ao carregar modelo: {e}")
    exit()

# --- 2. Conectar e Limpar o ChromaDB ---
print(f"🤖 [Ingestor Wiki] Conectando ao ChromaDB em '{CAMINHO_DB}'...")
try:
    client = chromadb.PersistentClient(path=CAMINHO_DB)

    # Apaga a coleção antiga, se existir
    try:
        if NOME_COLECAO in [c.name for c in client.list_collections()]:
            print(f"🤖 [Ingestor Wiki] Coleção '{NOME_COLECAO}' antiga encontrada. Removendo...")
            client.delete_collection(name=NOME_COLECAO)
    except Exception as e:
        print(f"🚨 ERRO ao tentar apagar coleção antiga: {e}")

    collection = client.get_or_create_collection(
        name=NOME_COLECAO,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"🤖 [Ingestor Wiki] Coleção '{NOME_COLECAO}' pronta.")
except Exception as e:
    print(f"🚨 ERRO ao conectar ao ChromaDB: {e}")
    exit()

# --- 3. Carregar e Processar o Dataset da Wikipédia ---
print(f"🤖 [Ingestor Wiki] Carregando dataset '{NOME_DATASET_HF}' ({CONFIG_DATASET_HF})...")
print("Isso pode baixar alguns GBs na primeira vez que você rodar.")

try:
    # streaming=True é crucial! Ele não baixa tudo, ele "puxa" sob demanda.
    dataset = load_dataset(NOME_DATASET_HF, CONFIG_DATASET_HF, split='train', streaming=True)

    print("🤖 [Ingestor Wiki] Dataset carregado. Iniciando processamento...")

    total_chunks = 0
    artigos_processados = 0

    # Prepara lotes (batches) para adicionar ao ChromaDB
    # Isso é MUITO mais rápido do que adicionar 1 por 1
    lote_chunks = []
    lote_metadados = []
    lote_ids = []
    TAMANHO_LOTE = 100

    # Itera sobre o dataset (artigo por artigo)
    for doc in dataset:
        if LIMITE_DE_ARTIGOS is not None and artigos_processados >= LIMITE_DE_ARTIGOS:
            print(f"🤖 [Ingestor Wiki] Atingido o limite de {LIMITE_DE_ARTIGOS} artigos.")
            break

        artigos_processados += 1
        titulo_artigo = doc['title']
        texto_artigo = doc['text']

        # 1. Fatiar (Chunking)
        chunks = text_splitter.split_text(texto_artigo)

        for i, chunk in enumerate(chunks):
            total_chunks += 1
            lote_chunks.append(chunk)
            lote_metadados.append({"fonte": "wikipedia", "titulo": titulo_artigo})
            lote_ids.append(f"wiki_{doc['id']}_chunk_{i}")

            # Quando o lote estiver cheio, processe e adicione ao DB
            if len(lote_chunks) >= TAMANHO_LOTE:
                # 2. Gerar Embeddings (em lote)
                embeddings = model.encode(lote_chunks, show_progress_bar=False).tolist()

                # 3. Adicionar ao ChromaDB (em lote)
                collection.add(
                    embeddings=embeddings,
                    documents=lote_chunks,
                    metadatas=lote_metadados,
                    ids=lote_ids
                )
                print(f"  -> Lote processado. Total de chunks indexados: {total_chunks}")

                # Limpar lotes
                lote_chunks = []
                lote_metadados = []
                lote_ids = []

    # Processar o último lote restante
    if lote_chunks:
        embeddings = model.encode(lote_chunks, show_progress_bar=False).tolist()
        collection.add(
            embeddings=embeddings,
            documents=lote_chunks,
            metadatas=lote_metadados,
            ids=lote_ids
        )
        print(f"  -> Lote final processado. Total de chunks indexados: {total_chunks}")

    print("\n✅ SUCESSO! Banco de dados vetorial populado com a Wikipédia.")
    print(f"Foram processados {artigos_processados} artigos.")
    print(f"Foram adicionados {collection.count()} chunks à coleção '{NOME_COLECAO}'.")

except Exception as e:
    print(f"🚨 ERRO durante o processamento do dataset: {e}")
    import traceback

    traceback.print_exc()