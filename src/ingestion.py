import os
import shutil
import logging
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config import setup_settings, CHROMA_DB_PATH, COLLECTION_NAME_BOOK_1, DATA_FILE_BOOK_1

# Configuração de logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def parse_document(file_path: str):
    """Utiliza o LlamaParse para ler o arquivo PDF e extrair seu conteúdo."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
    logging.info(f"Iniciando o parsing do documento: {file_path}")
    parser = LlamaParse(result_type="markdown")
    documents = parser.load_data(file_path)
    logging.info(f"Parsing concluído. {len(documents)} parte(s) extraída(s).")
    return documents

def split_text(documents, chunk_size=1024, chunk_overlap=200):
    """Realiza o chunking do texto extraído utilizando SentenceSplitter."""
    logging.info("Iniciando o chunking dos documentos...")
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    logging.info(f"Chunking concluído. {len(nodes)} chunks criados.")
    return nodes

def clear_database(db_path=CHROMA_DB_PATH):
    """Remove o diretório do ChromaDB se existir, prevenindo erros de dimensão de embeddings."""
    if os.path.exists(db_path):
        logging.warning(f"Removendo banco de dados antigo em: {db_path}...")
        shutil.rmtree(db_path)
        logging.info("Banco de dados limpo com sucesso.")

def persist_to_chromadb(nodes, db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME_BOOK_1):
    """Instancia o ChromaDB localmente e persiste os embeddings."""
    logging.info(f"Configurando ChromaDB local no caminho: {db_path}...")
    
    # Instancia o cliente Chroma persistente
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    
    # Configura o Vector Store do LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    logging.info("Gerando embeddings e salvando no ChromaDB...")
    # Cria o índice, gerando os embeddings (Gemini) e salvando na base
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    logging.info("Ingestão concluída com sucesso e dados persistidos!")
    return index

def main():
    try:
        # 1. Configurar configurações globais (Variáveis de Ambiente, Gemini e HuggingFace)
        logging.info("Carregando configurações e inicializando modelos...")
        setup_settings()
        
        # 2. Limpar o banco de dados antigo para evitar conflito de vetores
        clear_database()
        
        # 3. Ler PDF com LlamaParse
        file_path = DATA_FILE_BOOK_1
        documents = parse_document(file_path)
        
        # 3. Realizar o Chunking
        nodes = split_text(documents)
        
        # 4. Persistir no ChromaDB
        persist_to_chromadb(nodes)
        
    except Exception as e:
        logging.error(f"Erro no pipeline de ingestão: {e}")

if __name__ == "__main__":
    main()
