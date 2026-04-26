import os
import sys

# Corrige o erro de Unicode (charmap codec can't encode) no Windows console ao imprimir emojis (🧙‍♂️)
sys.stdout.reconfigure(encoding='utf-8')

# Força o HuggingFace a usar os modelos em modo offline (já que já foram baixados).
# Isso EVITA o "Segmentation Fault" do Windows causado pelo conflito entre o OpenSSL e o gRPC do ChromaDB/Gemini.
os.environ["HF_HUB_OFFLINE"] = "1"

from config import setup_settings, CHROMA_DB_PATH, COLLECTION_NAME_BOOK_1

from llama_index.core import VectorStoreIndex, PromptTemplate, QueryBundle
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from sentence_transformers import CrossEncoder
from typing import List, Optional
from llama_index.core.schema import NodeWithScore

class CustomCrossEncoderReranker(BaseNodePostprocessor):
    model_name: str
    top_n: int = 3
    _model: CrossEncoder = None

    def __init__(self, model_name: str, top_n: int = 3):
        super().__init__(model_name=model_name, top_n=top_n)
        self._model = CrossEncoder(model_name)

    @classmethod
    def class_name(cls) -> str:
        return "CustomCrossEncoderReranker"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not query_bundle or not nodes:
            return nodes
        
        query = query_bundle.query_str
        pairs = [[query, node.get_content()] for node in nodes]
        
        scores = self._model.predict(pairs)
        for node, score in zip(nodes, scores):
            node.score = float(score)
            
        nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return nodes[:self.top_n]

def main():
    # 1. Configurar configurações globais (Variáveis de Ambiente e Gemini)
    print("Carregando configurações e inicializando Gemini...")
    try:
        setup_settings()
    except Exception as e:
        print(f"Erro nas configurações: {e}")
        return

    # IMPORTANTE: Importamos chromadb apenas DEPOIS do setup_settings 
    # para evitar conflito de DLL do OpenSSL com o gRPC no Windows (SegFault).
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    # 2. Conectar ao ChromaDB existente e recriar o índice
    print("Conectando ao banco de dados ChromaDB local...")
    try:
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = db.get_collection(COLLECTION_NAME_BOOK_1)
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Recupera o índice diretamente do Vector Store
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # 3. Busca Híbrida: Preparando o BM25 Retriever
        print("Carregando contexto para o BM25 Retriever (Busca Exata)...")
        all_data = chroma_collection.get()
        nodes = []
        # Reconstrói os TextNodes a partir dos dados do banco para o BM25 processar as palavras-chave
        for doc_id, doc_text, doc_metadata in zip(all_data['ids'], all_data['documents'], all_data['metadatas']):
            nodes.append(TextNode(id_=doc_id, text=doc_text, metadata=doc_metadata or {}))
            
        print("Configurando QueryFusionRetriever (Híbrido) para top 20...")
        vector_retriever = index.as_retriever(similarity_top_k=20)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
        
        # O QueryFusionRetriever mescla os resultados e re-ranqueia (RRF) devolvendo 20
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=20,
            num_queries=1,
            mode="reciprocal_rerank"
        )
        
        print("Configurando Reranker Local (BGE Reranker)...")
        reranker = CustomCrossEncoderReranker(
            top_n=3,
            model_name="BAAI/bge-reranker-v2-m3",
        )    
        
        # 4. Prompt Estrito: Foca APENAS no contexto
        qa_prompt_str = (
            "Você é um assistente especialista no Livro 1 de Harry Potter.\n"
            "Baseie-se ESTRITAMENTE no contexto.\n"
            "Se o nome de um personagem ou criatura não estiver no texto, diga 'Não mencionado'.\n"
            "Não use descrições genéricas de crescimento se elas não constarem no fragmento.\n"
            "NÃO use seu conhecimento prévio.\n\n"
            "Contexto:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Pergunta: {query_str}\n"
            "Resposta: "
        )
        qa_template = PromptTemplate(qa_prompt_str)
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            text_qa_template=qa_template,
            node_postprocessors=[reranker]
        )
    except Exception as e:
        print(f"Erro ao carregar o banco de dados: {e}")
        return

    print("\n" + "="*50)
    print("🧙‍♂️ Bem-vindo à WikiPotterPedia (Livro 1)!")
    print("Faça perguntas sobre 'A Pedra Filosofal' ou digite 'sair' para encerrar.")
    print("="*50 + "\n")

    # 4. Loop de consultas no terminal
    while True:
        pergunta = input("\nSua pergunta: ")
        
        if pergunta.lower().strip() in ['sair', 'exit', 'quit']:
            print("\nMalfeito feito. Até logo! ⚡")
            break
            
        if not pergunta.strip():
            continue
            
        print("Pesquisando no livro e rankeando contexto...")
        try:
            query_bundle = QueryBundle(query_str=pergunta)
            
            # Recupera e faz o Rerank
            nodes_recuperados = query_engine.retrieve(query_bundle)
            
            # Imprime as fontes para depuração (antes da IA)
            print("\n--- Fontes Recuperadas e Rankeadas (Top 3) ---")
            for i, n in enumerate(nodes_recuperados):
                score_str = f"{n.score:.4f}" if n.score is not None else "N/A"
                print(f"[{i+1}] Score: {score_str} | {n.node.text[:200].replace(chr(10), ' ')}...")
            print("----------------------------------------------\n")
            
            # Pede para o LLM gerar a resposta baseada APENAS nesses nodes
            resposta = query_engine.synthesize(query_bundle, nodes_recuperados)
            print(f"🔮 Resposta: {resposta}")
        except Exception as e:
            print(f"\n❌ Erro ao buscar resposta: {e}")

if __name__ == "__main__":
    main()
