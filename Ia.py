import time
import json
import random
import numpy as np
import chromadb  # <-- Novo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer

# --- 0. Constantes de Confiança ---
LIMITE_CONFIANCA_CONVERSA = 0.7  # (SVC) 70% de certeza que é um 'oi', 'tchau', etc.
LIMITE_DISTANCIA_FATO = 0.5  # (Chroma) Distância de cosseno. MENOR é MELHOR. 0.5 é um bom corte.

# --- 1. CÉREBRO 1: O BOT CONVERSACIONAL (Classificador SVC) ---
# (Esta parte é idêntica à v3, apenas carrega o conhecimento.json e treina o SVC)

ARQUIVO_JSON_CONVERSA = 'conhecimento.json'
print(f"🤖 [Status] Carregando Cérebro Conversacional de '{ARQUIVO_JSON_CONVERSA}'...")
try:
    with open(ARQUIVO_JSON_CONVERSA, 'r', encoding='utf-8') as f:
        base_conhecimento_conversa = json.load(f)
except FileNotFoundError:
    print(f"🚨 ERRO: Arquivo '{ARQUIVO_JSON_CONVERSA}' não encontrado!")
    exit()

frases_de_treino_conversa = []
intentos_de_treino_conversa = []
for intento, dados in base_conhecimento_conversa.items():
    if intento != "fallback" and "exemplos" in dados and dados["exemplos"]:
        for exemplo in dados["exemplos"]:
            frases_de_treino_conversa.append(exemplo)
            intentos_de_treino_conversa.append(intento)

if not frases_de_treino_conversa:
    print("🚨 ERRO: Nenhuma frase de treino carregada do 'conhecimento.json'.")
    exit()

modelo_conversacional = Pipeline([
    ('vetorizador', TfidfVectorizer()),
    ('classificador', SVC(kernel='linear', probability=True))
])
modelo_conversacional.fit(frases_de_treino_conversa, intentos_de_treino_conversa)
print("🤖 [Status] Cérebro Conversacional treinado.")

# --- 2. CÉREBRO 2: O BOT FACTUAL (Conexão com ChromaDB) ---

NOME_MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'
NOME_COLECAO = "fatos_bot"
CAMINHO_DB = "./chroma_db"

print(f"🤖 [Status] Carregando Modelo de Embedding Factual ('{NOME_MODELO_EMBEMBEDDING}')...")
try:
    # Ainda precisamos do modelo para GERAR O VETOR da pergunta do usuário
    modelo_factual_search = SentenceTransformer(NOME_MODELO_EMBEDDING)
    print("🤖 [Status] Modelo Factual carregado.")
except Exception as e:
    print(f"🚨 ERRO ao carregar modelo SentenceTransformer: {e}")
    exit()

print("🤖 [Status] Conectando ao Banco de Dados Vetorial...")
try:
    client = chromadb.PersistentClient(path=CAMINHO_DB)
    collection_fatos = client.get_collection(name=NOME_COLECAO)
    print(f"🤖 [Status] Conectado ao ChromaDB! {collection_fatos.count()} fatos indexados.")
except Exception as e:
    print(f"🚨 ERRO: Não foi possível conectar ao ChromaDB em '{CAMINHO_DB}'.")
    print("🚨 Você executou o script 'popular_db.py' primeiro?")
    exit()


# --- 3. REMOÇÃO DAS FUNÇÕES ANTIGAS ---
# NÃO precisamos mais de:
# - carregar 'fatos.json'
# - 'base_fatos_lista' ou 'base_fatos_embeddings'
# - 'calcular_similaridade_cosseno'
# - 'buscar_fato_semantico'
# O ChromaDB faz tudo isso por nós!

# --- 4. Nova Função de Busca Vetorial ---

def buscar_fato_no_chroma(pergunta_usuario_emb):
    """
    Busca no ChromaDB o fato mais similar à pergunta.
    """
    try:
        # O ChromaDB espera uma lista de vetores (no nosso caso, só um)
        query_emb_list = pergunta_usuario_emb.tolist()

        # Faz a busca! Pede o vizinho mais próximo (n_results=1)
        results = collection_fatos.query(
            query_embeddings=[query_emb_list],
            n_results=1
        )

        # Analisa os resultados
        if not results['documents']:
            return None, 1.0  # Retorna 'None' e distância máxima (1.0)

        resposta = results['documents'][0][0]
        distancia = results['distances'][0][0]

        return resposta, distancia

    except Exception as e:
        print(f"🚨 ERRO durante a busca no ChromaDB: {e}")
        return None, 1.0


# --- 5. A Lógica Híbrida Principal (Modificada) ---

def obter_resposta_hibrida(pergunta_usuario):
    """
    Decide qual cérebro usar para responder.
    """
    pergunta_limpa = pergunta_usuario.lower().strip()
    if not pergunta_limpa:
        return random.choice(base_conhecimento_conversa["fallback"]["respostas"]), "fallback"

    # --- CÉREBRO 1: TENTATIVA CONVERSACIONAL (Igual) ---
    probabilidades = modelo_conversacional.predict_proba([pergunta_limpa])[0]
    max_prob = max(probabilidades)
    melhor_intento_idx = probabilidades.argmax()
    melhor_intento = modelo_conversacional.classes_[melhor_intento_idx]

    if max_prob > LIMITE_CONFIANCA_CONVERSA:
        resposta = random.choice(base_conhecimento_conversa[melhor_intento]["respostas"])
        return resposta, melhor_intento

    # --- CÉREBRO 2: TENTATIVA FACTUAL (Modificada para ChromaDB) ---
    # 1. Transformar a pergunta do usuário em um vetor (embedding)
    # Usamos .encode() que retorna um array numpy, perfeito para a nova função
    pergunta_emb = modelo_factual_search.encode(pergunta_limpa)

    # 2. Buscar a resposta no ChromaDB
    resposta_fato, distancia_fato = buscar_fato_no_chroma(pergunta_emb)

    # ATENÇÃO: Agora comparamos DISTÂNCIA (menor é melhor)
    if resposta_fato and distancia_fato < LIMITE_DISTANCIA_FATO:
        # Encontramos uma resposta factual com boa confiança!
        return resposta_fato, "fato_encontrado"

    # --- FALLBACK REAL ---
    resposta = random.choice(base_conhecimento_conversa["fallback"]["respostas"])
    return resposta, "fallback"


# --- 6. Loop Principal do Chat (Igual ao v3) ---
def iniciar_chat():
    print("🤖 Olá! Eu sou o ChatBot v4 (Híbrido com Banco de Dados Vetorial ChromaDB).")
    print("Eu posso conversar e responder fatos indexados.")
    print("Digite 'tchau' ou 'adeus' para sair.")
    print("-" * 30)

    rodando = True
    while rodando:
        pergunta = input("Você: ")

        resposta, intento = obter_resposta_hibrida(pergunta)

        print("🤖... ", end="", flush=True)
        time.sleep(0.5)
        print(resposta)

        if intento == "despedida":
            rodando = False


# --- Executa o programa ---
if __name__ == "__main__":
    iniciar_chat()