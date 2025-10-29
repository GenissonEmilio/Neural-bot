import time
import json
import random
import numpy as np
import chromadb
import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer

# --- 0. Configurações Globais ---

# --- Configuração do Cérebro Gerador (Gemini) ---
try:
    GOOGLE_API_KEY = "test"
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gerador = genai.GenerativeModel('gemini-1.5-flash-latest')  # Modelo rápido para RAG
    print("🤖 [Status] Cérebro Gerador (Gemini) inicializado.")
except Exception as e:
    print("🚨 ERRO: Não foi possível configurar a API do Gemini. Verifique sua chave de API.")
    print(f"Detalhe: {e}")
    exit()

# --- Constantes de Confiança ---
LIMITE_CONFIANCA_CONVERSA = 0.7  # (SVC) 70% de certeza que é um 'oi', 'tchau', etc.
LIMITE_DISTANCIA_FATO = 0.6  # (Chroma) Aumentamos um pouco, para pegar mais contexto
N_RESULTADOS_BUSCA = 3  # <-- NOVO: Buscar os 3 fatos mais relevantes

# --- 1. CÉREBRO 1: O BOT CONVERSACIONAL (Classificador SVC) ---
# (Esta parte é idêntica ao v4, treinando o classificador de intentos)
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

modelo_conversacional = Pipeline([
    ('vetorizador', TfidfVectorizer()),
    ('classificador', SVC(kernel='linear', probability=True))
])
modelo_conversacional.fit(frases_de_treino_conversa, intentos_de_treino_conversa)
print("🤖 [Status] Cérebro Conversacional treinado.")

# --- 2. CÉREBRO 2: O BOT FACTUAL (Conexão com ChromaDB) ---
# (Esta parte é idêntica ao v4, conectando ao ChromaDB)

NOME_MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'
NOME_COLECAO = "fatos_bot"
CAMINHO_DB = "./chroma_db"

print(f"🤖 [Status] Carregando Modelo de Embedding Factual ('{NOME_MODELO_EMBEDDING}')...")
try:
    modelo_factual_search = SentenceTransformer(NOME_MODELO_EMBEDDING)
    print("🤖 [Status] Modelo Factual (Embedding) carregado.")
except Exception as e:
    print(f"🚨 ERRO ao carregar modelo SentenceTransformer: {e}")
    exit()

print("🤖 [Status] Conectando ao Banco de Dados Vetorial (ChromaDB)...")
try:
    client = chromadb.PersistentClient(path=CAMINHO_DB)
    collection_fatos = client.get_collection(name=NOME_COLECAO)
    print(f"🤖 [Status] Conectado ao ChromaDB! {collection_fatos.count()} fatos indexados.")
except Exception as e:
    print(f"🚨 ERRO: Não foi possível conectar ao ChromaDB em '{CAMINHO_DB}'.")
    exit()


# --- 3. NOVAS FUNÇÕES: GERAÇÃO (RAG) ---

def buscar_contexto_no_chroma(pergunta_usuario_emb):
    """
    Busca no ChromaDB os N fatos mais similares à pergunta.
    """
    try:
        query_emb_list = pergunta_usuario_emb.tolist()
        results = collection_fatos.query(
            query_embeddings=[query_emb_list],
            n_results=N_RESULTADOS_BUSCA  # <-- Modificado para N resultados
        )

        documentos = results.get('documents', [[]])[0]
        distancias = results.get('distances', [[]])[0]

        return documentos, distancias

    except Exception as e:
        print(f"🚨 ERRO durante a busca no ChromaDB: {e}")
        return [], []


def gerar_resposta_com_llm(contexto, pergunta_usuario):
    """
    Usa o Cérebro Gerador (Gemini) para sintetizar uma resposta.
    """
    print("🤖 [Status] Cérebro Gerador (Gemini) está pensando...")

    # Este é o "prompt" que controla o LLM
    prompt_template = f"""
    Você é um assistente prestativo. Sua tarefa é responder à pergunta do usuário
    usando **apenas** o contexto fornecido.
    Se o contexto não contiver a resposta, diga "Desculpe, eu não tenho
    informação sobre isso no meu banco de dados."

    **Contexto Fornecido:**
    ---
    {contexto}
    ---

    **Pergunta do Usuário:**
    {pergunta_usuario}

    **Sua Resposta:**
    """

    try:
        # Gera a resposta
        response = model_gerador.generate_content(prompt_template)
        return response.text.strip(), "fato_gerado"
    except Exception as e:
        print(f"🚨 ERRO ao gerar resposta com LLM: {e}")
        return "Ocorreu um erro ao tentar gerar a resposta.", "fallback"


# --- 4. A Lógica Híbrida Principal (Modificada para RAG) ---

def obter_resposta_hibrida_rag(pergunta_usuario):
    """
    Decide qual cérebro usar: SVC (Conversa) ou RAG (Fatos).
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

    # --- CÉREBRO 2: TENTATIVA FACTUAL (AGORA COM RAG) ---

    # 1. BUSCA (Retrieve): Transformar pergunta em vetor e buscar no Chroma
    pergunta_emb = modelo_factual_search.encode(pergunta_limpa)
    documentos_contexto, distancias_contexto = buscar_contexto_no_chroma(pergunta_emb)

    # Se não encontramos nada ou o fato mais próximo é muito ruim, desistimos
    if not documentos_contexto or distancias_contexto[0] > LIMITE_DISTANCIA_FATO:
        return random.choice(base_conhecimento_conversa["fallback"]["respostas"]), "fallback"

    # 2. AUMENTA (Augment): Formata o contexto para o LLM
    contexto_formatado = "\n".join(f"- {doc}" for doc in documentos_contexto)

    # 3. GERA (Generate): Pede ao Gemini para criar a resposta
    resposta_gerada, intento = gerar_resposta_com_llm(contexto_formatado, pergunta_limpa)
    return resposta_gerada, intento


# --- 5. Loop Principal do Chat ---
def iniciar_chat():
    print("🤖 Olá! Eu sou o ChatBot v5 (RAG com Gemini e ChromaDB).")
    print("Eu posso conversar e gerar respostas complexas sobre meus fatos.")
    print("Digite 'tchau' ou 'adeus' para sair.")
    print("-" * 30)

    rodando = True
    while rodando:
        pergunta = input("Você: ")

        resposta, intento = obter_resposta_hibrida_rag(pergunta)

        # O "delay" agora é real, pois a API leva 1-2 segundos
        print("🤖... ", end="", flush=True)
        # time.sleep(0.5) # Não precisamos mais de um sleep falso
        print(resposta)

        if intento == "despedida":
            rodando = False


# --- Executa o programa ---
if __name__ == "__main__":
    iniciar_chat()