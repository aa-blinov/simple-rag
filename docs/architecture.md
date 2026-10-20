# Architecture

How `simple-rag` is put together.

## The shape

Two entry points, one class, one Chroma collection and one process-wide
settings object.

`rag_system.py` holds `LlamaIndexRAGSystem`, the class both entry points
use. It reads the documents, installs the LlamaIndex settings, opens the
Chroma collection and builds the query engine. `streamlit_rag.py` is the
browser interface on top of it: it lists the files, runs the indexing job,
shows the log and asks the questions. Neither file imports the other except
in one direction, and the class knows nothing about the browser.

```
data/ (PDF, DOCX, DOC)
  |
  |  rglob, three suffixes
  v
PDFReader / DocxReader                 (both entry points)
  |
  |  SimpleNodeParser: 384 tokens, 50 tokens of overlap
  v
HuggingFaceEmbedding                   (gte-multilingual-base, on the CPU)
  |
  v
ChromaDB, collection "default"          (chroma_store/)

question
  |
  v
VectorStoreIndex.as_retriever(top_k=3)
  |   + BM25Retriever, only when the docstore is populated
  v
QueryFusionRetriever                    (reciprocal rank fusion)
  |
  v
RetrieverQueryEngine
  |
  v
OpenAI-compatible endpoint              (LM Studio, http://localhost:1234/v1)
  |
  v
answer, and the sources behind it: file, score, fragment
```

| Module | What it holds |
| --- | --- |
| `rag_system.py` | `LlamaIndexRAGSystem`: the readers, the chunker, the embedding model, the Chroma collection, the query engine, and the terminal loop in `main()`. |
| `streamlit_rag.py` | The Streamlit app: the file grid, `index_files`, `clear_index`, the log console and the chat tab. |
| `requirements.txt` | 14 dependencies, every one of them pinned with `==`. |
| `.env.example` | 15 variables and their defaults. One of them, `LOG_LEVEL`, is read by nothing. |
| `data/` | The documents. Read with `rglob`, so a subfolder counts. |
| `chroma_store/` | The ChromaDB folder. One collection in it, named `default`. |
| `rag_debug.log` | Written by the Streamlit app at DEBUG level. Gitignored. |

The two Python files are the whole program. `src/`, `scripts/` and `tests/`
are empty directories with a `.gitkeep` in each, and there is no packaging
metadata.

## The data flow

### Indexing, from the browser

```
the grid
  |  the paths of the selected rows
  v
index_files(files)
  |  reads every file with PDFReader or DocxReader
  |  writes metadata: source, file, filename, file_type
  v
VectorStoreIndex.from_documents(docs, storage_context, embed_model)
  |
  v
the collection "default"
```

Before it embeds anything, `index_files` reads the metadata of the
collection and deletes the chunks whose `source` matches one of the selected
files. The embedding model is loaded first, through the cached
`get_embed_model`.

### Indexing, from the terminal

```
main()
  |
  v
rag.create_index()
  |  get_collection("default")
  |  on any failure: read the documents and create the collection
  v
rag.setup_query_engine()
  |
  v
a question, then another, until the word "exit"
```

The two paths reach the collection by different routes. The browser writes
with `VectorStoreIndex.from_documents`; the terminal loads it with
`VectorStoreIndex.from_vector_store` and only builds one from documents when
the collection is missing.

### Asking

```
question
  |
  v
query_engine.query(question)
  |
  v
response.source_nodes
  |  metadata["source"] for the file, node.score for the score
  v
answer text + a list of sources
```

The answer text is `str(response)`. The sources are built from
`response.source_nodes`, and the file name falls back through `source`,
`filename`, `file` and `path` before it gives up and writes "Неизвестно".

## Decisions worth knowing about

### The settings are process-wide

`Settings.embed_model`, `Settings.llm` and `Settings.node_parser` are
global, and `LlamaIndexRAGSystem.__init__` installs all three. Two systems
in one process share them.

The cost lands on the chunker. The node parser is installed only when a
`LlamaIndexRAGSystem` exists, and the Streamlit path calls
`VectorStoreIndex.from_documents` without a parser of its own, so it takes
the global one. Before that constructor runs, the chunks are the LlamaIndex
default rather than 384 tokens with 50 of overlap. The browser avoids the
second copy of the model by handing its cached embedding model into the
constructor.

### Both entry points write into one collection

The name is the literal `default` in both files. The browser writes chunks
and deletes the old ones by hand; the terminal loads what is there.

The cost is that the two writers do not agree about the shape of a chunk.
The browser writes `source`, `file`, `filename` and `file_type`; the
terminal writes `source`, `filename`, `creation_date` and `file_type`. Only
`source` is common, and it is the one the reader of the collection uses.

### The old chunks go before the new ones exist

`index_files` deletes the chunks of the selected files, then reads the
documents, then embeds them. A failure between the deletion and the
embedding leaves the collection without those files, and nothing puts them
back.

Deleting after the embedding would keep the store correct in that case, at
the cost of holding two copies of the selected files for the length of the
run.

### The hybrid retriever needs a populated docstore

`setup_query_engine` adds `BM25Retriever` when `USE_HYBRID_SEARCH` is on
and `len(self.index.docstore.docs) > 0`. The docstore is filled when the
index is built from documents and empty when the index is loaded from the
Chroma collection, which is what every run after the first does. The mode
line then reads "Семантический", and the engine keeps the vector retriever
alone.

So the switch in `.env` does what it says on the run that builds the index
and nothing on the runs after it.

### A `.doc` file is read with the DOCX reader

Both entry points route `.doc` to `DocxReader`, which parses the OOXML
format. A legacy binary `.doc` fails to parse, the file is skipped with an
error, and in the browser every failure costs a `time.sleep(2)`.

### A failed question becomes the answer

`query` catches every exception and returns the error message as the answer
text, with an empty source list. The browser then shows a failed request in
the same place it shows a reply from the model.

### The console keeps 20 lines and shows 15

`add_console_log` truncates the list to the last 20 entries, and the panel
renders the last 15 of them. The older five are kept and never shown, which
is what makes the console survive a rerun of the page.

### The grid's selection is read with a key the grid does not have

The records behind the grid carry "Директория", "Имя файла", "Тип" and
"Проиндексировано". The two `configure_column` calls name "file" and
"indexed", and the selection is read as `row["file"]`. The first selected
row raises `KeyError: 'file'`, so the indexing button cannot index
anything, and the index is built from the terminal.

### The environment is read at two different moments

`rag_system.py` reads the environment at import time, because the
constructor's default arguments call `os.getenv` when the module loads.
The browser reads it at call time, inside its cached functions. Changing
`DATA_PATH` after the import has no effect on the class.

## Boundaries

- **`streamlit_rag.py` may import `rag_system.py`.** The other direction does
  not exist: `rag_system.py` knows nothing about Streamlit and runs in a
  terminal with no browser in sight.
- **Only `streamlit_rag.py` may call `st.*`.** The class stays free of the
  browser, which is what lets the same class serve both entry points.
- **The Chroma client is opened in two places and nowhere else.**
  `rag_system.py` opens it in `_get_chroma_client`, and the browser opens it
  in `get_chroma_client`, which is a `@st.cache_resource`.
- **The embedding model is built in two places.** `get_embed_model` in the
  browser, and the constructor in `rag_system.py` when it is handed no
  model. Everything else receives it.
- **The collection name is a literal in two files.** There is no shared
  constant for `default`, and no namespace on top of it.
- **The LlamaIndex `Settings` are global to the process.** A second system in
  the same process inherits the embedding model, the LLM and the node parser
  of the first one.
