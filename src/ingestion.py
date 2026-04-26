import os
import sys
import shutil
import logging

# Corrige o erro de Unicode no Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Força o HuggingFace a usar os modelos em modo offline (já baixados).
# Isso EVITA o SegFault do Windows causado pelo conflito OpenSSL/gRPC.
os.environ["HF_HUB_OFFLINE"] = "1"

# IMPORTANTE: config.py importa sentence_transformers como primeira ação,
# o que "vacina" o processo contra o conflito de DLL do Windows.
# Nenhum outro import de rede/grpc deve acontecer antes disso.
from config import setup_settings, CHROMA_DB_PATH, COLLECTION_NAME_BOOK_1, DATA_FILE_BOOK_1, DOCSTORE_PATH

# Imports seguros (sem dependências gRPC/OpenSSL no nível de módulo)
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore

# IMPORTS TARDIOS (feitos dentro das funções, APÓS setup_settings):
#   - llama_parse        → tem dependências gRPC
#   - chromadb           → tem dependências gRPC/OpenSSL
#   - ChromaVectorStore  → depende do chromadb

# Configuração de logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_document(file_path: str):
    """
    Utiliza o LlamaParse para ler o arquivo PDF e extrair seu conteúdo.
    LlamaParse é importado aqui (import tardio) para evitar SegFault no Windows.
    """
    # Import tardio — deve ocorrer depois do setup_settings()
    from llama_parse import LlamaParse

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    logging.info(f"Iniciando o parsing do documento: {file_path}")
    parser = LlamaParse(result_type="markdown")
    documents = parser.load_data(file_path)
    logging.info(f"Parsing concluído. {len(documents)} parte(s) extraída(s).")
    return documents


def build_hierarchical_nodes(documents):
    """
    Aplica o HierarchicalNodeParser para criar dois níveis de chunks:
      - Nível Pai  : 1024 tokens (contexto rico para o LLM)
      - Nível Filho:  256 tokens (embeddings precisos para busca vetorial)

    Retorna TODOS os nodes (pais + filhos) para persistência no docstore,
    e APENAS os nodes folha para vetorização no ChromaDB.
    """
    logging.info("Iniciando o chunking hierárquico (Parent-Document Retrieval)...")

    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 256]
    )
    all_nodes = parser.get_nodes_from_documents(documents)

    leaf_nodes = get_leaf_nodes(all_nodes)
    root_nodes = get_root_nodes(all_nodes)

    logging.info(
        f"Chunking concluído. "
        f"Total: {len(all_nodes)} nodes | "
        f"Pais (1024 tokens): {len(root_nodes)} | "
        f"Filhos (256 tokens): {len(leaf_nodes)}"
    )
    return all_nodes, leaf_nodes


def persist_docstore(all_nodes, docstore_path: str = DOCSTORE_PATH):
    """
    Persiste TODOS os nodes (pais e filhos) no SimpleDocumentStore (arquivo JSON).
    O RecursiveRetriever usará esse store durante a consulta para fazer o
    lookup: filho -> pai.
    """
    logging.info(f"Persistindo {len(all_nodes)} nodes no SimpleDocumentStore: {docstore_path}")
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)
    docstore.persist(persist_path=docstore_path)
    logging.info("SimpleDocumentStore salvo com sucesso.")


def clear_database(db_path: str = CHROMA_DB_PATH):
    """Remove o diretório do ChromaDB se existir, prevenindo erros de dimensão de embeddings."""
    if os.path.exists(db_path):
        import stat

        def _force_remove(func, path, exc_info):
            """Callback para forçar remoção de arquivos bloqueados no Windows (WinError 5)."""
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                logging.warning(f"Não foi possível remover: {path} — ignorando.")

        logging.warning(f"Removendo banco de dados antigo em: {db_path}...")
        shutil.rmtree(db_path, onexc=_force_remove)
        logging.info("Banco de dados limpo com sucesso.")


def persist_to_chromadb(leaf_nodes, db_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME_BOOK_1):
    """
    Instancia o ChromaDB localmente e persiste os embeddings APENAS dos nodes folha.
    Os nodes folha (256 tokens) serão os alvos da busca vetorial/BM25.
    """
    # Imports tardios — DEVEM ocorrer depois do setup_settings() para evitar SegFault no Windows
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    logging.info(f"Configurando ChromaDB local no caminho: {db_path}...")

    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logging.info(f"Gerando embeddings para {len(leaf_nodes)} nodes filhos e salvando no ChromaDB...")
    VectorStoreIndex(leaf_nodes, storage_context=storage_context)
    logging.info("Ingestão concluída com sucesso e dados persistidos no ChromaDB!")


def main():
    try:
        # 1. Configurar configurações globais (Variáveis de Ambiente, Gemini e HuggingFace)
        logging.info("Carregando configurações e inicializando modelos...")
        setup_settings()

        # 2. Limpar o banco de dados antigo para evitar conflito de vetores
        clear_database()

        # 3. Ler PDF com LlamaParse (import tardio feito dentro da função)
        documents = parse_document(DATA_FILE_BOOK_1)

        # 4. Realizar o Chunking Hierárquico (Small-to-Big)
        all_nodes, leaf_nodes = build_hierarchical_nodes(documents)

        # 5. Persistir TODOS os nodes (pais + filhos) no SimpleDocumentStore (JSON)
        #    Obrigatório: o RecursiveRetriever precisa dos pais para o lookup
        persist_docstore(all_nodes)

        # 6. Persistir APENAS os nodes folha no ChromaDB (vetores de busca)
        persist_to_chromadb(leaf_nodes)

    except Exception as e:
        logging.error(f"Erro no pipeline de ingestão: {e}", exc_info=True)


if __name__ == "__main__":
    main()
