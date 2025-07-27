import os
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings as ChromaSettings
from rag_system import LlamaIndexRAGSystem
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

load_dotenv()


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    store_path = os.getenv("STORE_PATH", "chroma_store")
    return chromadb.PersistentClient(
        path=store_path, settings=ChromaSettings(anonymized_telemetry=False)
    )


@st.cache_resource(show_spinner="Загрузка модели эмбеддинга...")
def get_embed_model():
    return HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL_NAME", "Alibaba-NLP/gte-multilingual-base"),
        device=os.getenv("EMBED_MODEL_DEVICE", "cpu"),
        trust_remote_code=os.getenv("EMBED_MODEL_TRUST_REMOTE_CODE", "True").lower()
        == "true",
    )


st.set_page_config(page_title="RAG Search", page_icon="🔍", layout="wide")


@st.cache_data(ttl=300)
def get_pdf_files():
    data_dir = Path(os.getenv("DATA_PATH", "data"))
    return sorted(data_dir.rglob("*.pdf")) if data_dir.exists() else []


@st.cache_data(ttl=10)
def get_indexed_files():
    try:
        client = get_chroma_client()
        try:
            col = client.get_collection("default")
            res = col.get(include=["metadatas"])
            metadatas = res.get("metadatas")
            if not metadatas:
                return set()
            return set(m["source"] for m in metadatas if m and "source" in m)
        except Exception:
            return set()
    except Exception:
        return set()


def index_files(files):
    st.info("Индексация...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        embed_model = get_embed_model()

        docs = []
        reader = PDFReader()

        total_files = len(files)
        for idx, f in enumerate(files):
            status_text.text(f"Обработка файла {idx + 1}/{total_files}: {Path(f).name}")
            file_docs = reader.load_data(file=Path(f))
            for doc in file_docs:
                doc.metadata = doc.metadata or {}
                doc.metadata["source"] = str(f)
                doc.metadata["filename"] = Path(f).name
            docs.extend(file_docs)
            progress_bar.progress((idx + 1) / total_files)

        status_text.text("Сохранение в векторную базу данных...")

        client = get_chroma_client()
        try:
            col = client.get_or_create_collection("default")
        except:
            col = client.create_collection("default")

        # Проверяем метаданные
        missing_metadata = 0
        for i, doc in enumerate(docs):
            if "source" not in doc.metadata:
                missing_metadata += 1
                doc.metadata["source"] = f"Документ #{i}"

        if missing_metadata > 0:
            st.warning(
                f"{missing_metadata} документов не имели метаданных 'source', проблема исправлена!"
            )

        store = ChromaVectorStore(chroma_collection=col)
        ctx = StorageContext.from_defaults(vector_store=store)

        status_text.text("Создание векторных эмбеддингов и индексация...")
        VectorStoreIndex.from_documents(
            docs, storage_context=ctx, embed_model=embed_model
        )

        st.success(
            f"Индексация завершена: {len(files)} файлов, всего {len(docs)} фрагментов документов"
        )
    except Exception as e:
        st.error(f"Ошибка индексации: {e}")
    st.session_state.indexing = False
    st.rerun()


def clear_index():
    with st.spinner("Очистка индекса..."):
        try:
            client = get_chroma_client()
            client.delete_collection("default")
            st.success("Индекс очищен")
        except Exception as e:
            st.info(f"Индекс уже пуст или произошла ошибка: {e}")

        get_indexed_files.clear()

        if "rag" in st.session_state:
            st.session_state.rag = None

        st.rerun()


def main():
    for key, default_value in {
        "screen": "index",
        "messages": [],
        "indexing": False,
        "rag": None,
        "last_pdf_count": 0,
        "last_indexed_count": 0,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    st.sidebar.title("Навигация")
    screen = st.sidebar.radio(
        "Экран:",
        ["Индексация", "Чат"],
        index=0 if st.session_state.screen == "index" else 1,
    )
    st.session_state.screen = "index" if screen == "Индексация" else "chat"

    pdfs = get_pdf_files()
    indexed = get_indexed_files()

    st.sidebar.metric(
        "PDF файлов",
        len(pdfs),
        delta=len(pdfs) - st.session_state.last_pdf_count
        if st.session_state.last_pdf_count > 0
        else None,
    )
    st.sidebar.metric(
        "Проиндексировано",
        len(indexed),
        delta=len(indexed) - st.session_state.last_indexed_count
        if st.session_state.last_indexed_count > 0
        else None,
    )
    st.sidebar.metric("Чат-сообщений", len(st.session_state.messages))
    st.session_state.last_pdf_count = len(pdfs)
    st.session_state.last_indexed_count = len(indexed)

    if st.session_state.screen == "index":
        st.header("📚 Индексация документов")
        if not pdfs:
            st.warning("Нет PDF файлов в папке data/")
            return
        table = []
        for f in pdfs:
            table.append(
                {
                    "Файл": f.name,
                    "Путь": str(f),
                    "Статус": "✅" if str(f) in indexed else "❌",
                    "Размер (KB)": f.stat().st_size // 1024,
                    "Изменён": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                }
            )
        df = pd.DataFrame(table)
        st.dataframe(df, use_container_width=True)
        st.markdown("---")
        sel = st.multiselect(
            "Выберите файлы для индексации:",
            [str(f) for f in pdfs if str(f) not in indexed],
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "📥 Индексировать выбранные",
                disabled=not sel or st.session_state.indexing,
            ):
                st.session_state.indexing = True
                index_files(sel)
        with col2:
            if st.button("🗑️ Очистить индекс"):
                clear_index()
    else:
        st.header("💬 Чат по документам")
        if not st.session_state.rag:
            with st.spinner("Инициализация системы вопросов-ответов..."):
                try:
                    embed_model = get_embed_model()

                    rag = LlamaIndexRAGSystem(embed_model=embed_model)
                    rag.create_index()
                    rag.setup_query_engine()
                    st.session_state.rag = rag
                except Exception as e:
                    st.error(f"Ошибка инициализации: {e}")
                    return
        st.markdown("---")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant" and m.get("sources"):
                    with st.expander(f"📚 Источники ({len(m['sources'])})"):
                        for i, s in enumerate(m["sources"], 1):
                            st.markdown(f"**{i}. {Path(s['file']).name}**  ")
                            score = s["score"]
                            if score >= 0.8:
                                score_color = "green"
                                rating = "отличная"
                            elif score >= 0.6:
                                score_color = "blue"
                                rating = "хорошая"
                            elif score >= 0.4:
                                score_color = "orange"
                                rating = "средняя"
                            else:
                                score_color = "red"
                                rating = "низкая"

                            st.markdown(
                                f"Релевантность: :{score_color}[**{rating}**] (`{score:.3f}`)  "
                            )
                            st.markdown("**Фрагмент:**")
                            st.markdown(f"*{s['content']}*")
                            st.divider()
        if q := st.chat_input("Ваш вопрос по документам..."):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Поиск ответа..."):
                    try:
                        res = st.session_state.rag.query(q)
                        st.markdown(res["answer"])
                        if res["sources"]:
                            with st.expander(f"📚 Источники ({len(res['sources'])})"):
                                for i, s in enumerate(res["sources"], 1):
                                    st.markdown(f"**{i}. {Path(s['file']).name}**  ")
                                    score = s["score"]
                                    if score >= 0.8:
                                        score_color = "green"
                                        rating = "отличная"
                                    elif score >= 0.6:
                                        score_color = "blue"
                                        rating = "хорошая"
                                    elif score >= 0.4:
                                        score_color = "orange"
                                        rating = "средняя"
                                    else:
                                        score_color = "red"
                                        rating = "низкая"

                                    st.markdown(
                                        f"Релевантность: :{score_color}[**{rating}**] (`{score:.3f}`)  "
                                    )
                                    st.markdown("**Фрагмент:**")
                                    st.markdown(f"*{s['content']}*")
                                    st.divider()
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": res["answer"],
                                "sources": res["sources"],
                            }
                        )
                    except Exception as e:
                        st.error(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
