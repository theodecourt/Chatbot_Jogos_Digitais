import PyPDF2
import openai
import faiss
import numpy as np
import os
import pickle
from dotenv import load_dotenv

# Carregar variáveis do arquivo .env
load_dotenv()

# Pegar a chave da OpenAI de forma segura
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("Chave da OpenAI não encontrada! Verifique o arquivo .env.")

# Definir arquivos de armazenamento do índice FAISS e dos chunks de texto
FAISS_INDEX_FILE = "faiss_index.idx"
CHUNKS_FILE = "chunks.pkl"

def extract_text_from_pdf(pdf_path):
    """Extrai todo o texto de um arquivo PDF."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=500, overlap=50):
    """Divide o texto em chunks menores."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start+chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap  # avanço com sobreposição
    return chunks

def get_embeddings(texts):
    """Gera embeddings para uma lista de textos."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Erro ao gerar embeddings: {e}")
        return []

def build_faiss_index(embeddings):
    """Cria um índice FAISS e o salva em disco."""
    if not embeddings:
        print("Nenhum embedding foi gerado. Verifique a entrada.")
        return None
    
    d = len(embeddings[0])  # Dimensão dos embeddings
    index = faiss.IndexFlatL2(d)
    embedding_array = np.array(embeddings).astype("float32")
    index.add(embedding_array)
    
    # Salvar índice FAISS
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    return index

def load_faiss_index():
    """Carrega o índice FAISS do disco, se existir."""
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        print("Índice FAISS carregado do disco.")
        return index
    return None

def save_chunks(chunks):
    """Salva os chunks de texto associados ao índice FAISS."""
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    """Carrega os chunks de texto, se existirem."""
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        print("Chunks de texto carregados do disco.")
        return chunks
    return []

def query_index(query, index, texts, k=3):
    """Busca os chunks mais relevantes para a pergunta do usuário."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding).astype("float32")
        query_embedding = np.expand_dims(query_embedding, axis=0)  # Ajusta para FAISS
        
        distances, indices = index.search(query_embedding, k)
        retrieved_texts = [texts[i] for i in indices[0]]
        return retrieved_texts
    except Exception as e:
        print(f"Erro ao buscar contexto: {e}")
        return []

def generate_answer(query, context):
    """Gera uma resposta baseada no contexto extraído do PDF."""
    prompt = f"Contexto:\n{context}\n\nPergunta: {query}\n\nResposta:"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente que responde perguntas dos jogadores sobre um jogo de tabuleiro. Use o contexto fornecido para responder."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao processar sua pergunta."

def main():
    pdf_path = "notion-manual_morcef.pdf"  # Substitua pelo caminho do seu PDF
    
    # Verificar se o índice já existe para carregamento
    index = load_faiss_index()
    chunks = load_chunks()
    
    if index is None or not chunks:
        print("Processando PDF para gerar embeddings...")
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text, chunk_size=500, overlap=50)
        embeddings = get_embeddings(chunks)
        
        index = build_faiss_index(embeddings)
        save_chunks(chunks)
        
        if index is None:
            print("Erro ao criar o índice FAISS. Encerrando.")
            return
    else:
        print("Usando índice FAISS carregado.")

    # Loop de interação com o usuário
    while True:
        query = input("Digite sua pergunta (ou 'sair' para terminar): ")
        if query.lower() == "sair":
            break
        retrieved_chunks = query_index(query, index, chunks, k=3)
        context = "\n\n".join(retrieved_chunks)
        
        if not context:
            print("Nenhum contexto relevante encontrado. Tente reformular a pergunta.")
            continue
        
        answer = generate_answer(query, context)
        print("\nResposta:")
        print(answer)
        print("-" * 40)

if __name__ == "__main__":
    main()