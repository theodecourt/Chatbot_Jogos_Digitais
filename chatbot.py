import PyPDF2
import openai
import faiss
import numpy as np
import os

import os
from dotenv import load_dotenv

# Carregar variáveis do arquivo .env
load_dotenv()

# Pegar a chave da OpenAI de forma segura
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("Chave da OpenAI não encontrada! Verifique o arquivo .env.")

def extract_text_from_pdf(pdf_path):
    """Extrai todo o texto de um arquivo PDF."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=500, overlap=50):
    """
    Divide o texto em chunks menores.
    - chunk_size: número de palavras por chunk.
    - overlap: número de palavras sobrepostas entre chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start+chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap  # avanço com sobreposição
    return chunks

def get_embeddings(texts):
    """
    Gera embeddings para uma lista de textos utilizando o modelo "text-embedding-ada-002".
    """
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item.embedding for item in response.data]  # Retorna a lista de embeddings
    except Exception as e:
        print(f"Erro ao gerar embeddings: {e}")
        return []

def build_faiss_index(embeddings):
    """
    Cria um índice FAISS a partir da lista de embeddings.
    """
    if not embeddings:
        print("Nenhum embedding foi gerado. Verifique a entrada.")
        return None, None
    
    d = len(embeddings[0])  # Dimensão dos embeddings
    index = faiss.IndexFlatL2(d)
    embedding_array = np.array(embeddings).astype("float32")
    index.add(embedding_array)
    return index, embedding_array

def query_index(query, index, texts, k=3):
    """
    Dado uma query, gera seu embedding e retorna os k chunks mais semelhantes.
    """
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
    """
    Gera a resposta do ChatGPT usando o contexto recuperado e a pergunta do usuário.
    """
    prompt = f"Contexto:\n{context}\n\nPergunta: {query}\n\nResposta:"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Ou "gpt-4" se desejar mais precisão
            messages=[
                {"role": "system", "content": "Você é um assistente que responde perguntas dos jogadores a respeito de um jogo de tabuleiro. Para responder utilize o contexto fornecido."},
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
    text = extract_text_from_pdf(pdf_path)
    
    # Divida o texto em chunks para facilitar a indexação
    chunks = split_text(text, chunk_size=500, overlap=50)
    
    # Gera os embeddings dos chunks
    embeddings = get_embeddings(chunks)
    
    # Cria o índice FAISS
    index, _ = build_faiss_index(embeddings)
    
    if index is None:
        print("Erro ao criar o índice FAISS. Encerrando.")
        return
    
    # Loop de interação com o usuário
    while True:
        query = input("Digite sua pergunta (ou 'sair' para terminar): ")
        if query.lower() == "sair":
            break
        # Recupera os chunks mais relevantes
        retrieved_chunks = query_index(query, index, chunks, k=3)
        context = "\n\n".join(retrieved_chunks)
        
        if not context:
            print("Nenhum contexto relevante encontrado. Tente reformular a pergunta.")
            continue
        
        # Gera a resposta usando o ChatGPT
        answer = generate_answer(query, context)
        print("\nResposta:")
        print(answer)
        print("-" * 40)

if __name__ == "__main__":
    main()