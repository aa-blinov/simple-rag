import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import DocxReader, PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()


class LlamaIndexRAGSystem:
    def __init__(
        self,
        data_path: str = os.getenv("DATA_PATH", "data"),
        persist_path: str = os.getenv("STORE_PATH", "chroma_store"),
        chunk_size: int = int(os.getenv("CHUNK_SIZE", 384)),
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50)),
        top_k: int = int(os.getenv("TOP_K", 3)),
        lm_studio_base_url: str = os.getenv(
            "LM_STUDIO_API_URL", "http://localhost:1234/v1"
        ),
        embed_model=None,
    ):
        self.data_path = Path(data_path)
        self.persist_path = Path(persist_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embed_model = embed_model
        self.use_hybrid_search = (
            os.getenv("USE_HYBRID_SEARCH", "False").lower() == "true"
        )
        self.verbose_retrieval = (
            os.getenv("VERBOSE_RETRIEVAL", "False").lower() == "true"
        )
        self.actual_search_mode = (
            "Семантический"
            if not self.use_hybrid_search
            else "Гибридный (семантический + BM25)"
        )

        self._setup_logging()
        self._setup_llama_index(lm_studio_base_url)

        self.index = None
        self.query_engine = None

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _setup_llama_index(self, base_url: str) -> None:
        if self.embed_model is not None:
            Settings.embed_model = self.embed_model
        else:
            model_name = os.getenv(
                "EMBED_MODEL_NAME", "Alibaba-NLP/gte-multilingual-base"
            )
            device = os.getenv("EMBED_MODEL_DEVICE", "cpu")
            trust_remote_code = (
                os.getenv("EMBED_MODEL_TRUST_REMOTE_CODE", "True").lower() == "true"
            )

            Settings.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                trust_remote_code=trust_remote_code,
            )

        model = os.getenv("LM_STUDIO_MODEL", "gpt-3.5-turbo")
        api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        temperature = float(os.getenv("LM_STUDIO_TEMPERATURE", 0.1))

        Settings.llm = OpenAI(
            model=model, api_base=base_url, api_key=api_key, temperature=temperature
        )

        Settings.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _get_chroma_client(self):
        return chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def load_documents(self) -> List[Document]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Папка данных {self.data_path} не найдена")

        pdf_files = list(self.data_path.rglob("*.pdf"))
        docx_files = list(self.data_path.rglob("*.docx"))
        doc_files = list(self.data_path.rglob("*.doc"))

        total_files = len(pdf_files) + len(docx_files) + len(doc_files)

        if total_files == 0:
            raise FileNotFoundError(
                f"Ни PDF, ни DOC/DOCX файлы не найдены в {self.data_path}"
            )

        self.logger.info(
            f"Найдено {len(pdf_files)} PDF файлов, {len(docx_files)} DOCX файлов и {len(doc_files)} DOC файлов"
        )

        documents = []
        pdf_reader = PDFReader()
        docx_reader = DocxReader()

        for pdf_file in pdf_files:
            try:
                docs = pdf_reader.load_data(file=pdf_file)
                for doc in docs:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["source"] = str(pdf_file)
                    doc.metadata["filename"] = pdf_file.name
                    doc.metadata["creation_date"] = datetime.now().isoformat()
                    doc.metadata["file_type"] = "pdf"

                documents.extend(docs)
                self.logger.info(
                    f"Загружено {len(docs)} страниц из PDF: {pdf_file.name}"
                )
            except Exception as e:
                self.logger.error(f"Ошибка загрузки PDF {pdf_file}: {e}")
                continue

        for docx_file in docx_files:
            try:
                docs = docx_reader.load_data(file=docx_file)
                for doc in docs:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["source"] = str(docx_file)
                    doc.metadata["filename"] = docx_file.name
                    doc.metadata["creation_date"] = datetime.now().isoformat()
                    doc.metadata["file_type"] = "docx"

                documents.extend(docs)
                self.logger.info(
                    f"Загружено {len(docs)} фрагментов из DOCX: {docx_file.name}"
                )
            except Exception as e:
                self.logger.error(f"Ошибка загрузки DOCX {docx_file}: {e}")
                continue

        for doc_file in doc_files:
            try:
                docs = docx_reader.load_data(file=doc_file)
                for doc in docs:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["source"] = str(doc_file)
                    doc.metadata["filename"] = doc_file.name
                    doc.metadata["creation_date"] = datetime.now().isoformat()
                    doc.metadata["file_type"] = "doc"

                documents.extend(docs)
                self.logger.info(
                    f"Загружено {len(docs)} фрагментов из DOC: {doc_file.name}"
                )
            except Exception as e:
                self.logger.error(f"Ошибка загрузки DOC {doc_file}: {e}")
                continue

        self.logger.info(f"Всего загружено документов: {len(documents)}")
        return documents

    def create_index(self, force_recreate: bool = False) -> VectorStoreIndex:
        """Создание или загрузка индекса"""
        chroma_client = self._get_chroma_client()

        if force_recreate and self.persist_path.exists():
            self.logger.info("Пересоздание индекса...")
            try:
                chroma_client.delete_collection("default")
            except ValueError:
                pass

        try:
            chroma_collection = chroma_client.get_collection("default")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, storage_context=storage_context
            )
            self.logger.info("Загружен существующий индекс")

        except (ValueError, Exception):
            self.logger.info("Создание нового индекса...")
            documents = self.load_documents()

            chroma_collection = chroma_client.create_collection("default")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            for i, doc in enumerate(documents):
                if "source" not in doc.metadata:
                    self.logger.warning(
                        f"Документ #{i} не имеет метаданных 'source', добавляем"
                    )
                    doc.metadata["source"] = f"Документ #{i}"
                self.logger.debug(f"Документ #{i} метаданные: {doc.metadata}")

            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=Settings.embed_model,
            )
            self.logger.info(
                f"Индекс создан успешно. Всего документов: {len(documents)}"
            )

        return self.index

    def setup_query_engine(self) -> RetrieverQueryEngine:
        """Настройка поискового движка (семантический или гибридный)"""
        if self.index is None:
            raise ValueError("Индекс не создан. Вызовите create_index() сначала.")

        vector_retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        retrievers = [vector_retriever]

        hybrid_active = False
        if self.use_hybrid_search:
            try:
                from llama_index.retrievers.bm25 import BM25Retriever

                docs_count = len(self.index.docstore.docs)
                if docs_count > 0:
                    try:
                        bm25_retriever = BM25Retriever.from_defaults(
                            docstore=self.index.docstore, similarity_top_k=self.top_k
                        )
                        retrievers.append(bm25_retriever)
                        self.logger.info(
                            f"Гибридный поиск активирован: семантический + BM25 (найдено {docs_count} документов)"
                        )
                        hybrid_active = True
                    except Exception as e:
                        self.logger.warning(
                            f"Ошибка при инициализации BM25Retriever: {e}. Используется только векторный поиск."
                        )
                else:
                    self.logger.warning(
                        f"BM25Retriever не может быть инициализирован: в индексе нет документов (docs_count={docs_count}). Используется только векторный поиск."
                    )
            except (ImportError, Exception) as e:
                self.logger.warning(
                    f"BM25Retriever недоступен: {e}. Используется только векторный поиск."
                )
        else:
            self.logger.info("Используется только семантический (векторный) поиск")

        if len(retrievers) > 1:
            self.query_engine = RetrieverQueryEngine.from_args(
                QueryFusionRetriever(
                    retrievers,
                    similarity_top_k=self.top_k,
                    num_queries=1,
                    mode=FUSION_MODES.RECIPROCAL_RANK,
                    use_async=False,
                    verbose=self.verbose_retrieval,
                )
            )
        else:
            self.query_engine = RetrieverQueryEngine.from_args(vector_retriever)

        if self.use_hybrid_search and not hybrid_active:
            self.actual_search_mode = "Семантический"

        self.logger.info(
            f"Поисковый движок настроен в режиме: {self.actual_search_mode}"
        )
        return self.query_engine

    def query(self, question: str) -> Dict[str, Any]:
        if self.query_engine is None:
            raise ValueError(
                "Поисковый движок не настроен. Вызовите setup_query_engine() сначала."
            )

        self.logger.info(f"Обработка запроса: {question}")

        try:
            response = self.query_engine.query(question)

            sources = []
            if hasattr(response, "source_nodes"):
                self.logger.info(
                    f"Найдено {len(response.source_nodes)} релевантных узлов"
                )

                for i, node in enumerate(response.source_nodes):
                    self.logger.info(f"Узел #{i + 1} метаданные: {node.metadata}")
                    file_source = node.metadata.get("source")
                    if not file_source:
                        file_source = (
                            node.metadata.get("filename")
                            or node.metadata.get("file")
                            or node.metadata.get("path")
                            or "Неизвестно"
                        )

                    sources.append(
                        {
                            "file": file_source,
                            "score": getattr(node, "score", 0.0),
                            "content": node.text,
                        }
                    )
            else:
                self.logger.warning("Ответ не содержит source_nodes")

            return {"answer": str(response), "sources": sources, "query": question}

        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": f"Ошибка обработки запроса: {e}",
                "sources": [],
                "query": question,
            }

    def get_stats(self) -> Dict[str, Any]:
        if self.index is None:
            return {"status": "Индекс не создан"}

        try:
            chroma_client = self._get_chroma_client()
            collection = chroma_client.get_collection("default")
            count = collection.count()

            return {
                "status": "Готов",
                "total_chunks": count,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "search_mode": self.actual_search_mode,
                "hybrid_search_enabled": self.use_hybrid_search,
                "verbose_retrieval": self.verbose_retrieval,
            }
        except Exception as e:
            return {"status": f"Ошибка получения статистики: {e}"}


def main():
    rag = LlamaIndexRAGSystem()

    print("Создание индекса...")
    rag.create_index()

    print("Настройка поискового движка...")
    rag.setup_query_engine()

    print("RAG система готова!")
    print("Статистика:", rag.get_stats())

    while True:
        question = input("\nЗадайте вопрос (или 'exit' для выхода): ").strip()
        if question.lower() == "exit":
            break

        if question:
            result = rag.query(question)
            print(f"\nОтвет: {result['answer']}")

            if result["sources"]:
                print("\nИсточники:")
                for i, source in enumerate(result["sources"], 1):
                    print(
                        f"{i}. {Path(source['file']).name} (оценка: {source['score']:.3f})"
                    )


if __name__ == "__main__":
    main()
