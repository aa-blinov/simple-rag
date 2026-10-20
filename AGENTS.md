# AGENTS.md — agent context for `simple-rag`

A local question-and-answer system over a folder of PDF and Word documents.
It embeds the documents with a HuggingFace model, keeps the vectors in a
ChromaDB folder on the disk and answers from any OpenAI-compatible endpoint.
There are two entry points: a Streamlit interface and a terminal loop.

This file is what you need to work on the project without reading all of it.

## Run / build / test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # the 14 pinned dependencies
cp .env.example .env              # optional; every value has a default
streamlit run streamlit_rag.py    # the browser interface, on port 8501
python rag_system.py              # the terminal loop
```

There is no test suite, no linter configuration, no CI workflow and no
packaging metadata. `python -m compileall rag_system.py streamlit_rag.py` is
the closest thing to a check. Python 3.10 is the floor.

Two things have to be ready before the first answer: the embedding model,
which is downloaded from HuggingFace on the first use, and an
OpenAI-compatible server answering on `http://localhost:1234/v1`.

## Where things live

| Path | What it holds |
| --- | --- |
| `rag_system.py` | `LlamaIndexRAGSystem`: the readers, the chunker, the embedding model, the Chroma collection, the query engine and the terminal `main()`. |
| `streamlit_rag.py` | The Streamlit app: the file grid, `index_files`, `clear_index`, the log console and the chat tab. |
| `requirements.txt` | 14 dependencies, every one of them pinned with `==`. |
| `.env.example` | 15 variables and their defaults, one of which the code ignores. |
| `data/` | The documents. Read with `rglob`, so subfolders count. |
| `chroma_store/` | The ChromaDB folder. One collection in it, named `default`. |
| `rag_debug.log` | Written by the Streamlit app at DEBUG level. Gitignored. |

The app imports the library by module name (`from rag_system import
LlamaIndexRAGSystem`), so it has to be started from the project root.

## Traps

- **The grid has no `file` column, and the selection is read as
  `row["file"]`.** The records behind the grid carry the keys "Директория",
  "Имя файла", "Тип" and "Проиндексировано", and the two
  `configure_column` calls name "file" and "indexed", neither of which
  exists. `st.session_state.selected_files = [row["file"] for row in
  selected]` raises `KeyError: 'file'` on the first selected row, so the
  indexing button cannot index anything until the path column or the key
  comes back.
- **Hybrid search only works in the run that builds the index.**
  `setup_query_engine` adds the BM25 retriever only when
  `len(self.index.docstore.docs) > 0`. The docstore is filled when the index
  is built from documents and empty when the index is loaded from the
  Chroma collection with `VectorStoreIndex.from_vector_store`, which is what
  every run after the first does. The mode then reports "Семантический" and
  the engine keeps the vector retriever alone.
- **The chunks in the store are not always 384 tokens.**
  `CHUNK_SIZE` and `CHUNK_OVERLAP` are installed into the global
  `Settings.node_parser` by `LlamaIndexRAGSystem.__init__`. The Streamlit
  path calls `VectorStoreIndex.from_documents` without a node parser, so it
  takes the global one, which is the LlamaIndex default until a
  `LlamaIndexRAGSystem` exists in the same process.
- **A `.doc` file is read with the DOCX reader.** Both entry points route
  `.doc` to `DocxReader`, which parses the OOXML format. A legacy binary
  `.doc` fails to parse, the file is skipped with an error, and in the
  browser every failure costs a `time.sleep(2)`.
- **The old chunks are deleted before the new ones exist.** `index_files`
  removes the chunks of the selected files from the collection before it
  embeds anything. A failure after that point leaves the collection without
  those files, and nothing puts them back.
- **`create_index` catches everything.** `except (ValueError, Exception)`
  treats a broken collection the same as a missing one, and the fallback
  path calls `create_collection("default")`, which raises when the
  collection is already there. A load failure therefore surfaces as a
  collection error rather than as its cause.
- **The document list is cached for 300 seconds.** `get_document_files` and
  `get_indexed_files` are `@st.cache_data(ttl=300)`, so a file dropped into
  `data/` does not appear in the grid until the TTL runs out.
  `get_indexed_files.clear()` clears the whole data cache, not just that one
  function.
- **The log level is decided in the source.** `streamlit_rag.py` calls
  `logging.basicConfig(level=logging.DEBUG, ...)` with a file handler on
  `rag_debug.log`, and `rag_system.py` calls
  `logging.basicConfig(level=logging.INFO, ...)` from `_setup_logging`,
  which does nothing once the root logger has handlers.
  `.env.example` carries `LOG_LEVEL=INFO`, and no code reads it.
- **The constructor defaults are read at import time.**
  `data_path: str = os.getenv("DATA_PATH", "data")` is a default argument,
  so the value is frozen when `rag_system` is imported. Changing the
  environment afterwards has no effect on it.
- **A failed question becomes the answer.** `query` catches every exception
  and returns the error message as the answer text with an empty source
  list, so the browser shows a failed request in the same place it shows a
  reply from the model.
- **`pandas` is imported and not declared.** `streamlit_rag.py` imports
  pandas at the top and `requirements.txt` does not list it. It arrives
  through streamlit (`pandas<3,>=1.4.0`) and through
  llama-index-readers-file (`pandas<2.3.0`). Three pins are never imported
  at all: `docx2txt`, `llama-index-embeddings-openai` and `pillow`.

## Style / safety rails

- **The interface is in Russian.** The tab titles, the buttons, the metrics,
  the log lines and the error messages are Russian, and so are the
  comments and the commit messages. Keep them that way. The documents in
  this repository are English.
- **Every dependency is pinned with `==`.** Do not float a version, and do
  not add one without a reason next to it.
- **The model is reached over an OpenAI-compatible API.** The default is
  LM Studio on port 1234 and the key is the literal `lm-studio`. Do not
  hardcode a hosted provider.
- **The corpus stays on the machine.** The documents, the embeddings and
  the store are local files, and only the question and the retrieved
  fragments leave the machine, to the configured endpoint.
- **Every answer carries its sources.** The file name, the score and the
  fragment travel with the answer. Keep that when the query path changes.
- **The LlamaIndex `Settings` are global.** The embedding model, the LLM
  and the node parser are process-wide, and they are set from the
  constructor of one class. Two systems in one process share them.
- **The Chroma collection is shared by both entry points.** The browser
  writes into `default` directly and the terminal reads it. Clearing the
  index in the browser removes what the terminal sees.

## Where the product truth lives

`PRODUCT.md` holds it. Read it before changing the behaviour of the system,
and write the change there when the behaviour changes.
