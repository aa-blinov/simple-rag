import logging
import os
import time
from datetime import datetime
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import DocxReader, PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_system import LlamaIndexRAGSystem

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_debug.log", encoding="utf-8"),  # Запись в файл
    ],
)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    store_path = os.getenv("STORE_PATH", "chroma_store")
    logger.debug(f"Инициализация ChromaDB клиента с путем: {store_path}")
    return chromadb.PersistentClient(
        path=store_path, settings=ChromaSettings(anonymized_telemetry=False)
    )


@st.cache_resource(show_spinner=False)
def get_embed_model():
    model_name = os.getenv("EMBED_MODEL_NAME", "Alibaba-NLP/gte-multilingual-base")
    device = os.getenv("EMBED_MODEL_DEVICE", "cpu")
    logger.debug(f"Загрузка модели эмбеддинга: {model_name} на устройстве: {device}")
    return HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        trust_remote_code=os.getenv("EMBED_MODEL_TRUST_REMOTE_CODE", "True").lower()
        == "true",
    )


st.set_page_config(page_title="RAG Search", page_icon="🔍", layout="wide")


@st.cache_data(ttl=300)
def get_document_files():
    data_dir = Path(os.getenv("DATA_PATH", "data"))
    logger.debug(f"Поиск документов в директории: {data_dir}")

    if not data_dir.exists():
        logger.warning(f"Директория {data_dir} не существует")
        return []

    pdf_files = list(data_dir.rglob("*.pdf"))
    docx_files = list(data_dir.rglob("*.docx"))
    doc_files = list(data_dir.rglob("*.doc"))

    all_files = sorted(pdf_files + docx_files + doc_files)
    logger.debug(
        f"Найдено файлов: PDF={len(pdf_files)}, DOCX={len(docx_files)}, DOC={len(doc_files)}"
    )

    return [str(f) for f in all_files]


@st.cache_data(ttl=300)
def get_indexed_files():
    try:
        client = get_chroma_client()
        try:
            col = client.get_collection("default")
            res = col.get(include=["metadatas"])
            metadatas = res.get("metadatas")
            if not metadatas:
                return []
            return list({m["source"] for m in metadatas if m and "source" in m})
        except Exception:
            return set()
    except Exception:
        return set()


def index_files(files):
    logger.info(f"Начало индексации {len(files)} файлов")

    # Получаем ссылку на консоль из session_state
    if "console_logs" not in st.session_state:
        st.session_state.console_logs = []

    def add_console_log(message):
        st.session_state.console_logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        )
        if len(st.session_state.console_logs) > 20:
            st.session_state.console_logs = st.session_state.console_logs[-20:]
        logger.debug(message)

    progress_container = st.container()

    with progress_container:
        st.info("🔄 Индексация...")
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        add_console_log("Инициализация модели эмбеддинга...")
        with st.spinner("Загрузка модели эмбеддинга..."):
            embed_model = get_embed_model()
        add_console_log("✅ Модель эмбеддинга загружена")

        docs = []
        pdf_reader = PDFReader()
        docx_reader = DocxReader()
        add_console_log("✅ Ридеры документов инициализированы")

        total_files = len(files)
        add_console_log(f"📋 Начинаем обработку {total_files} файлов...")

        for idx, f in enumerate(files):
            file_path = Path(f)
            file_ext = file_path.suffix.lower()
            current_progress = (idx / total_files) * 0.8  # 80% для обработки файлов

            progress_message = (
                f"📄 Обработка файла {idx + 1}/{total_files}: {file_path.name}"
            )
            status_text.text(progress_message)
            add_console_log(progress_message)

            if file_ext == ".pdf":
                reader = pdf_reader
                file_type = "pdf"
            elif file_ext in [".docx", ".doc"]:
                reader = docx_reader
                file_type = "docx" if file_ext == ".docx" else "doc"
            else:
                skip_message = f"⚠️ Пропуск неподдерживаемого файла: {file_path.name}"
                status_text.text(skip_message)
                add_console_log(skip_message)
                continue

            try:
                add_console_log(f"🔍 Чтение файла {file_path.name}...")
                file_docs = reader.load_data(file=file_path)

                for doc in file_docs:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["source"] = str(f)
                    doc.metadata["file"] = str(f)
                    doc.metadata["filename"] = file_path.name
                    doc.metadata["file_type"] = file_type

                docs.extend(file_docs)
                success_message = (
                    f"✅ Файл {file_path.name}: получено {len(file_docs)} фрагментов"
                )
                status_text.text(success_message)
                add_console_log(success_message)

            except Exception as e:
                error_message = f"❌ Ошибка при обработке {file_path.name}: {e}"
                status_text.text(error_message)
                add_console_log(error_message)
                st.error(error_message)
                logger.error(error_message, exc_info=True)
                time.sleep(2)
                continue

            progress_bar.progress(current_progress)

        add_console_log(f"📊 Всего получено {len(docs)} фрагментов документов")
        status_text.text("💾 Сохранение в векторную базу данных...")
        add_console_log("💾 Подключение к векторной базе данных...")

        client = get_chroma_client()
        try:
            col = client.get_or_create_collection("default")
            add_console_log("✅ Коллекция ChromaDB готова")
        except Exception:
            col = client.create_collection("default")
            add_console_log("✅ Создана новая коллекция ChromaDB")

        files_to_reindex = set(str(f) for f in files)
        add_console_log("🧹 Проверка на дублирующиеся документы...")

        try:
            existing_data = col.get(include=["metadatas"])
            metadatas = existing_data.get("metadatas")
            if metadatas:
                ids_to_delete = []
                for i, metadata in enumerate(metadatas):
                    if metadata and metadata.get("source") in files_to_reindex:
                        ids_to_delete.append(existing_data["ids"][i])

                if ids_to_delete:
                    col.delete(ids=ids_to_delete)
                    delete_message = f"🗑️ Удалено {len(ids_to_delete)} старых чанков для переиндексируемых файлов"
                    add_console_log(delete_message)
                    logger.info(delete_message)
                    get_indexed_files.clear()
                else:
                    add_console_log("✅ Дублирующихся документов не найдено")
        except Exception as e:
            warning_msg = f"⚠️ Ошибка при удалении старых чанков: {e}"
            add_console_log(warning_msg)
            logger.warning(warning_msg)

        missing_metadata = 0
        for i, doc in enumerate(docs):
            if "source" not in doc.metadata:
                missing_metadata += 1
                doc.metadata["source"] = f"Документ #{i}"

        if missing_metadata > 0:
            warning_message = f"⚠️ {missing_metadata} документов не имели метаданных 'source', проблема исправлена!"
            st.warning(warning_message)
            add_console_log(warning_message)

        store = ChromaVectorStore(chroma_collection=col)
        ctx = StorageContext.from_defaults(vector_store=store)

        progress_bar.progress(0.9)
        embedding_message = "🧠 Создание векторных эмбеддингов и индексация..."
        status_text.text(embedding_message)
        add_console_log(embedding_message)

        VectorStoreIndex.from_documents(
            docs, storage_context=ctx, embed_model=embed_model
        )

        progress_bar.progress(1.0)
        final_message = f"🎉 Индексация завершена: {len(files)} файлов, всего {len(docs)} фрагментов документов"
        st.success(final_message)
        add_console_log(final_message)
        logger.info(final_message)

        get_indexed_files.clear()

    except Exception as e:
        error_message = f"💥 Критическая ошибка индексации: {e}"
        st.error(error_message)
        add_console_log(error_message)
        logger.error(error_message, exc_info=True)

    st.session_state.indexing = False
    add_console_log("🏁 Процесс индексации завершен")


def clear_index():
    logger.info("Начало очистки индекса")

    if "console_logs" not in st.session_state:
        st.session_state.console_logs = []

    def add_console_log(message):
        st.session_state.console_logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        )
        if len(st.session_state.console_logs) > 20:
            st.session_state.console_logs = st.session_state.console_logs[-20:]
        logger.debug(message)

    add_console_log("🗑️ Начинаем очистку индекса...")

    clear_container = st.container()
    with clear_container:
        with st.spinner("Очистка индекса..."):
            try:
                client = get_chroma_client()
                client.delete_collection("default")
                success_msg = "Индекс очищен"
                st.success(success_msg)
                add_console_log("✅ " + success_msg)
                logger.info(success_msg)
            except Exception as e:
                info_msg = f"Индекс уже пуст или произошла ошибка: {e}"
                st.info(info_msg)
                add_console_log("ℹ️ " + info_msg)
                logger.warning(info_msg)

            get_indexed_files.clear()
            add_console_log("🔄 Кэш проиндексированных файлов очищен")
            logger.debug("Кэш проиндексированных файлов очищен")

            if "rag" in st.session_state:
                st.session_state.rag = None
                add_console_log("🔄 RAG система сброшена")
                logger.debug("RAG система сброшена")

        st.rerun()


def main():
    logger.debug("Запуск главной функции приложения")

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

    docs = [Path(p) for p in get_document_files()]
    indexed = get_indexed_files()

    logger.debug(f"Статистика: документов={len(docs)}, проиндексировано={len(indexed)}")

    st.sidebar.title("📊 Статистика")
    st.sidebar.metric(
        "Документов",
        len(docs),
        delta=len(docs) - st.session_state.last_pdf_count
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

    st.session_state.last_pdf_count = len(docs)
    st.session_state.last_indexed_count = len(indexed)

    tab1, tab2 = st.tabs(["📚 Индексация документов", "💬 Чат по документам"])

    with tab1:
        st.header("📚 Индексация документов")
        if st.session_state.indexing:
            st.info("🔄 Идёт процесс индексации... Пожалуйста, подождите.")
            return

        if not docs:
            st.warning("Нет PDF, DOC или DOCX файлов в папке data/")
            return

        if "selected_files" not in st.session_state:
            st.session_state.selected_files = []

        data_dir = Path(os.getenv("DATA_PATH", "data"))

        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

        st.subheader("Выбор файлов для индексации")
        records = []
        for f in docs:
            file_type = f.suffix.lower()
            icon = "📄"
            if file_type == ".pdf":
                icon = "📕"
            elif file_type in [".doc", ".docx"]:
                icon = "📘"
            try:
                relp = f.relative_to(data_dir)
                dirp = str(relp.parent) if str(relp.parent) != "." else ""
            except Exception:
                dirp = str(f.parent)
            records.append(
                {
                    "Директория": dirp,
                    "Имя файла": f"{icon} {f.stem}",
                    "Тип": file_type[1:].upper(),
                    "Проиндексировано": str(f) in indexed,
                }
            )
        df_grid = pd.DataFrame(records)

        gb = GridOptionsBuilder.from_dataframe(df_grid)

        gb.configure_column("file", hide=True)
        gb.configure_column("indexed", headerName="Проиндексировано")
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        grid_options = gb.build()
        grid_response = AgGrid(
            df_grid,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            enable_enterprise_modules=False,
        )
        # Safely handle no selection (None) case
        raw_selection = grid_response.get("selected_rows")
        if raw_selection is None:
            selected = []
        elif isinstance(raw_selection, pd.DataFrame):
            selected = raw_selection.to_dict(orient="records")
        else:
            selected = raw_selection
        st.session_state.selected_files = [row["file"] for row in selected]
        st.info(
            f"Выбрано файлов для индексации: {len(st.session_state.selected_files)}"
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "📥 Индексировать выбранные",
                disabled=not st.session_state.selected_files
                or st.session_state.indexing,
            ):
                st.session_state.indexing = True
                index_files(st.session_state.selected_files)
        with col2:
            if st.button("🗑️ Очистить индекс"):
                clear_index()

        # Постоянная консоль логирования под таблицей (всегда открыта)
        st.markdown("---")
        st.subheader("📊 Консоль логирования")

        # Инициализируем логи, если их еще нет
        if "console_logs" not in st.session_state:
            st.session_state.console_logs = [
                f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Система готова к работе"
            ]

        # Создаем контейнер для консоли
        console_container = st.container()

        # Отображаем консоль
        with console_container:
            if st.session_state.console_logs:
                # Показываем последние 15 записей
                console_text = "\n".join(st.session_state.console_logs[-15:])
                st.code(console_text, language="text")

        # Кнопка очистки логов
        if st.button("🧹 Очистить логи", help="Очистить консоль логирования"):
            st.session_state.console_logs = [
                f"[{datetime.now().strftime('%H:%M:%S')}] 🧹 Логи очищены"
            ]
            st.rerun()

    with tab2:
        logger.debug("Открыт экран чата")

        # Проверяем, есть ли проиндексированные документы
        if len(indexed) == 0:
            st.warning(
                "📝 Для начала работы с чатом необходимо проиндексировать документы"
            )
            st.info(
                "👈 Перейдите на вкладку **'📚 Индексация документов'** и выберите файлы для индексации"
            )
            return

        if not st.session_state.rag:
            logger.debug("Инициализация RAG системы")
            init_container = st.container()
            with init_container:
                with st.spinner("Инициализация системы вопросов-ответов..."):
                    try:
                        embed_model = get_embed_model()

                        rag = LlamaIndexRAGSystem(embed_model=embed_model)
                        rag.create_index()
                        rag.setup_query_engine()
                        st.session_state.rag = rag
                        logger.info("RAG система успешно инициализирована")
                    except Exception as e:
                        error_msg = f"Ошибка инициализации: {e}"
                        st.error(error_msg)
                        logger.error(error_msg, exc_info=True)
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
            logger.info(f"Получен вопрос от пользователя: {q}")
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Поиск ответа..."):
                    try:
                        logger.debug("Начало обработки запроса через RAG систему")
                        res = st.session_state.rag.query(q)
                        logger.debug(
                            f"Получен ответ, количество источников: {len(res.get('sources', []))}"
                        )

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
                        logger.info(
                            "Ответ успешно сформирован и добавлен в историю чата"
                        )
                    except Exception as e:
                        error_msg = f"Ошибка: {e}"
                        st.error(error_msg)
                        logger.error(
                            f"Ошибка при обработке запроса: {e}", exc_info=True
                        )


if __name__ == "__main__":
    main()
