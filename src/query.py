import os
import sys

# Corrige o erro de Unicode (charmap codec can't encode) no Windows console ao imprimir emojis (🧙‍♂️)
sys.stdout.reconfigure(encoding='utf-8')

# Força o HuggingFace a usar os modelos em modo offline (já que já foram baixados).
# Isso EVITA o "Segmentation Fault" do Windows causado pelo conflito entre o OpenSSL e o gRPC do ChromaDB/Gemini.
os.environ["HF_HUB_OFFLINE"] = "1"

from config import (
    setup_settings, CHROMA_DB_PATH, COLLECTION_NAME_BOOK_1, DOCSTORE_PATH,
    HABILITAR_TEXTO_INICIALIZACAO, HABILITAR_TEXTO_DEBUG_RERANK
)

from llama_index.core import VectorStoreIndex, PromptTemplate, QueryBundle
from llama_index.core.retrievers import QueryFusionRetriever, RecursiveRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.schema import TextNode, IndexNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from sentence_transformers import CrossEncoder
from typing import List, Optional
from llama_index.core.schema import NodeWithScore


# ---------------------------------------------------------------------------
# Reranker Local (BGE-M3 Cross-Encoder)
# ---------------------------------------------------------------------------

class CustomCrossEncoderReranker(BaseNodePostprocessor):
    model_name: str
    top_n: int = 6
    _model: CrossEncoder = None

    def __init__(self, model_name: str, top_n: int = 6):
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


# ---------------------------------------------------------------------------
# Construção do RecursiveRetriever (Small-to-Big)
# ---------------------------------------------------------------------------

def build_recursive_retriever(index: VectorStoreIndex, chroma_collection, docstore: SimpleDocumentStore):
    """
    Constrói o pipeline de recuperação Parent-Document (Small-to-Big):

    1. QueryFusionRetriever  → busca nos nodes FILHOS (256 tokens) via Vector + BM25
    2. RecursiveRetriever    → faz lookup no docstore e retorna os nodes PAIS (1024 tokens)

    Isso garante: precisão na busca (filhos pequenos) + riqueza de contexto para o LLM (pais grandes).
    """

    # --- Estágio 1: Buscadores nos nodes FILHOS ---
    if HABILITAR_TEXTO_INICIALIZACAO:
        print("Carregando nodes filhos do ChromaDB para o BM25 Retriever...")
    all_data = chroma_collection.get()
    leaf_nodes = [
        TextNode(id_=doc_id, text=doc_text, metadata=doc_metadata or {})
        for doc_id, doc_text, doc_metadata in zip(
            all_data["ids"], all_data["documents"], all_data["metadatas"]
        )
    ]

    if HABILITAR_TEXTO_INICIALIZACAO:
        print(f"  → {len(leaf_nodes)} nodes filhos carregados.")

    vector_retriever = index.as_retriever(similarity_top_k=20)
    bm25_retriever   = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=20)

    # Fusão híbrida (RRF) sobre os filhos
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=20,
        num_queries=1,
        mode="reciprocal_rerank",
    )

    # --- Estágio 2: RecursiveRetriever (filho → pai) ---
    # Mapeia cada node filho para um IndexNode que aponta para seu PAI no docstore.
    # Se o filho não tiver pai (já é raiz), usamos o próprio ID do filho como alvo.
    # --- Estágio 2: RecursiveRetriever (filho → pai) ---
    # Criamos um dicionário que mapeia o ID do node filho para um IndexNode.
    # Quando o RecursiveRetriever encontra esse ID, ele segue o 'index_id' para o PAI.
    if HABILITAR_TEXTO_INICIALIZACAO:
        print("Construindo mapa filho → pai para o RecursiveRetriever...")

    all_nodes_dict = docstore.docs  # Contém todos os nodes (pais e filhos)
    recursive_node_dict = {**all_nodes_dict} # Cópia inicial

    for leaf in leaf_nodes:
        leaf_node_full = all_nodes_dict.get(leaf.node_id)
        if leaf_node_full and leaf_node_full.parent_node:
            parent_id = leaf_node_full.parent_node.node_id
            
            # Substituímos o TextNode folha por um IndexNode no dicionário de busca.
            # O RecursiveRetriever, ao receber este node, verá que é um IndexNode
            # e buscará o 'parent_id' no mesmo dicionário.
            recursive_node_dict[leaf.node_id] = IndexNode(
                text=leaf.text,
                index_id=parent_id,
                id_=leaf.node_id,
                metadata=leaf.metadata,
            )

    if HABILITAR_TEXTO_INICIALIZACAO:
        print(f"  → Mapeamento de {len(leaf_nodes)} nodes filhos para seus respectivos pais concluído.")

    # RecursiveRetriever: busca nos filhos (via hybrid_retriever) e recupera os pais do docstore
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": hybrid_retriever},
        node_dict=recursive_node_dict,
        verbose=False,
    )

    return recursive_retriever, hybrid_retriever


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    # 1. Configurar configurações globais
    if HABILITAR_TEXTO_INICIALIZACAO:
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

    # 2. Validar e carregar o SimpleDocumentStore (gerado pela ingestão)
    if not os.path.exists(DOCSTORE_PATH):
        print(
            f"\n❌ ERRO: SimpleDocumentStore não encontrado em '{DOCSTORE_PATH}'.\n"
            "Execute 'python src/ingestion.py' primeiro para gerar o docstore com os nodes pais."
        )
        return

    if HABILITAR_TEXTO_INICIALIZACAO:
        print(f"Carregando SimpleDocumentStore de: {DOCSTORE_PATH}")
    docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
    total_nodes = len(docstore.docs)
    if HABILITAR_TEXTO_INICIALIZACAO:
        print(f"  → {total_nodes} nodes (pais + filhos) carregados do docstore.")

    # 3. Conectar ao ChromaDB existente e recriar o índice sobre os nodes FILHOS
    if HABILITAR_TEXTO_INICIALIZACAO:
        print("Conectando ao banco de dados ChromaDB local...")
    try:
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = db.get_collection(COLLECTION_NAME_BOOK_1)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
    except Exception as e:
        print(f"Erro ao carregar o banco de dados: {e}")
        return

    # 4. Construir o RecursiveRetriever (Small-to-Big)
    if HABILITAR_TEXTO_INICIALIZACAO:
        print("Configurando RecursiveRetriever (Parent-Document Retrieval)...")
    try:
        recursive_retriever, hybrid_retriever = build_recursive_retriever(
            index, chroma_collection, docstore
        )
    except Exception as e:
        print(f"Erro ao construir o RecursiveRetriever: {e}")
        import traceback; traceback.print_exc()
        return

    # 5. Reranker sobre os NODES PAIS (contexto rico)
    if HABILITAR_TEXTO_INICIALIZACAO:
        print("Configurando Reranker Local (BGE-M3 Cross-Encoder)...")
    reranker = CustomCrossEncoderReranker(
        top_n=6,
        model_name="BAAI/bge-reranker-v2-m3",
    )

    # 6. Prompt estrito — foca APENAS no contexto
    qa_prompt_str = (
        "Você é um assistente especialista e investigador detalhista no Livro 1 de Harry Potter.\n"
        "Sua missão é extrair informações com precisão cirúrgica baseando-se no contexto fornecido.\n\n"
        "DIRETRIZES DE INVESTIGAÇÃO:\n"
        "1. LOCALIZAÇÃO: Ao descrever locais, procure no contexto por andares, alas, salas específicas ou referências geográficas detalhadas.\n"
        "2. DESCRIÇÃO FÍSICA: Se o texto mencionar a cor, o material de um objeto ou o tamanho de algo, inclua na resposta.\n"
        "3. CORRELAÇÃO: Responda com base no contexto, mas sinta-se livre para correlacionar informações de diferentes partes do texto fornecido para formar uma resposta completa.\n\n"
        "RESTRIÇÕES:\n"
        "- Se uma entidade ou detalhe não for encontrado no texto, responda 'Não mencionado'.\n"
        "- NÃO use seu conhecimento prévio sob nenhuma circunstância.\n\n"
        "Contexto:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Pergunta: {query_str}\n"
        "Resposta: "
    )
    qa_template = PromptTemplate(qa_prompt_str)

    # 7. Base Engine (Recuperação + Rerank)
    base_query_engine = RetrieverQueryEngine.from_args(
        retriever=recursive_retriever,
        text_qa_template=qa_template,
        node_postprocessors=[reranker],
    )

    # 8. Transformar o base_engine em uma Ferramenta (Tool)
    query_engine_tool = QueryEngineTool(
        query_engine=base_query_engine,
        metadata=ToolMetadata(
            name="wiki_potter_engine",
            description="Motor de busca especializado no Livro 1 de Harry Potter. Útil para buscar detalhes sobre personagens, locais e eventos específicos.",
        ),
    )

    # 9. Instanciar o SubQuestionQueryEngine para decompor perguntas complexas
    if HABILITAR_TEXTO_INICIALIZACAO:
        print("Configurando Sub-Question Query Engine (Multi-hop Reasoning)...")
    
    # IMPORTANTE: Definimos o gerador de perguntas explicitamente para usar o Gemini (Settings.llm)
    # e evitar o erro que tenta buscar o pacote do OpenAI por padrão.
    from llama_index.core import Settings
    question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[query_engine_tool],
        question_gen=question_gen,
        use_async=False, # Mantemos síncrono para evitar problemas de concorrência no Windows console
    )

    print("\n" + "="*60)
    print("🧙‍♂️ Bem-vindo à WikiPotterPedia (Livro 1) — Small-to-Big Edition!")
    print("Faça perguntas sobre 'A Pedra Filosofal' ou digite 'sair' para encerrar.")
    print("="*60 + "\n")

    # 8. Loop de consultas no terminal
    while True:
        pergunta = input("\nSua pergunta: ")

        if pergunta.lower().strip() in ["sair", "exit", "quit"]:
            print("\nMalfeito feito. Até logo! ⚡")
            break

        if not pergunta.strip():
            continue

        if HABILITAR_TEXTO_DEBUG_RERANK:
            print("Analisando pergunta complexa e decompondo em sub-questões...")
        try:
            # O SubQuestionQueryEngine gerencia internamente a recuperação e síntese
            resposta = query_engine.query(pergunta)
            print(f"🔮 Resposta: {resposta}")

        except Exception as e:
            print(f"\n❌ Erro ao buscar resposta: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
