import sentence_transformers  # Fix para evitar SegFault (DLL Hell) no Windows com o ChromaDB
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Constantes globais do projeto
CHROMA_DB_PATH = "./chroma_db_v2"
COLLECTION_NAME_BOOK_1 = "hp_book_1"
DATA_FILE_BOOK_1 = "./data/Harry Potter (1) E A Pedra Filosofal.pdf"
# Caminho do SimpleDocumentStore (nodes pais) persistido pela ingestão
DOCSTORE_PATH = "./docstore.json"

# Controles de exibição (1 = SIM, 0 = NÃO)
HABILITAR_TEXTO_INICIALIZACAO = 0
HABILITAR_TEXTO_DEBUG_RERANK = 0

def setup_settings():
    """Carrega variáveis de ambiente e configura globalmente o Gemini (LLM e Embeddings)."""
    # 1. Carregar variáveis de ambiente
    load_dotenv()
    
    required_keys = ["LLAMA_CLOUD_API_KEY", "GOOGLE_API_KEY"]
    for key in required_keys:
        if not os.getenv(key):
            raise ValueError(f"Chave de API ausente: {key} não encontrada no arquivo .env")
            
    # 2. Configuração de Arquitetura Híbrida
    # Embeddings Locais: Usamos um modelo multilingue (BAAI/bge-m3) que suporta Português, 
    # rodando 100% local e sem custo.
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    
    # IMPORTANTE: Importamos Gemini apenas AQUI para não carregar grpcio antes do 
    # download HTTPS do HuggingFace (causador de SegFault no Windows).
    from llama_index.llms.gemini import Gemini
    
    # LLM Cloud: Gemini para raciocinar e gerar respostas com base nos textos recuperados.
    Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0.0)
