# simple-rag

Ask questions about a folder of your own PDF and Word documents. The files
are read, cut into chunks, embedded with a model that runs on your machine
and kept in a ChromaDB store on your disk, and a chat in the browser answers
from them, with the source file and the relevance score for every fragment
it used. The same index answers questions in a terminal, and the language
model behind it is any OpenAI-compatible server, LM Studio by default.

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://github.com/aa-blinov/simple-rag)

---

## Quick start

```bash
git clone https://github.com/aa-blinov/simple-rag.git
cd simple-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run streamlit_rag.py
```

The app serves on port 8501 by default. Put the documents in `data/` and
open the first tab, where the files are listed in a grid.

The files are picked in that grid, and the indexing run is started from
there. Reading the selection raises `KeyError: 'file'` in the current code,
so the index is built from the terminal for now:

```bash
python rag_system.py
```

It reads everything under `data/`, then waits for a question until you type
`exit`.

The model that writes the answers has to be running before you ask anything:
start LM Studio, load a model and leave its server on port 1234, or point
`LM_STUDIO_API_URL` at another OpenAI-compatible endpoint.

The interface, the log lines and the error messages are in Russian. The
documents are in English.

## What it does

- **Index a folder.** PDF, DOCX and DOC files under `data/` are read, cut
  into chunks of 384 tokens with 50 tokens of overlap and embedded with
  `Alibaba-NLP/gte-multilingual-base` on the CPU by default.
- **Keep the vectors next to the documents.** ChromaDB writes to
  `chroma_store/`, in a collection named `default`. The browser and the
  terminal use the same collection.
- **Ask in the browser.** The chat tab answers from the indexed chunks and
  shows the file, the score and the fragment for every source behind it. A
  score of 0.8 or more is rated "отличная", 0.6 "хорошая", 0.4 "средная",
  and anything below that "низкая".
- **Ask in the terminal.** `python rag_system.py` builds the index and then
  takes questions until you type `exit`.
- **Search by meaning, and by word.** With `USE_HYBRID_SEARCH=True` a BM25
  retriever joins the vector retriever, and the two rankings are fused with
  reciprocal rank fusion.
- **Watch the run.** The indexing run reports which file is being read, how
  many fragments came out of it and where the embedding step stands, in a
  progress bar, a status line and a console under the grid.
- **Clear the store.** One button deletes the `default` collection and
  starts over.

## How it works

```
data/*.pdf | *.docx | *.doc
  -> PDFReader | DocxReader            (rag_system.py)
  -> SimpleNodeParser                  (chunk_size 384, overlap 50)
  -> HuggingFaceEmbedding              (gte-multilingual-base, cpu)
  -> ChromaDB "default"                (chroma_store/)

question
  -> VectorStoreIndex.as_retriever(top_k=3)   [ + BM25Retriever ]
  -> QueryFusionRetriever               (reciprocal rank fusion)
  -> RetrieverQueryEngine
  -> OpenAI-compatible endpoint         (LM Studio, http://localhost:1234/v1)
```

`rag_system.py` holds the class both entry points use,
`LlamaIndexRAGSystem`. Its constructor installs the LlamaIndex settings:
the embedding model, the LLM and the node parser. `create_index()` opens
the Chroma collection and, when there is no collection yet, reads the
documents and embeds them. `setup_query_engine()` builds the retriever, and
`query()` returns the answer text together with the sources behind it.

`streamlit_rag.py` is the browser interface. It writes into the same
collection through `VectorStoreIndex.from_documents`, and for the chat it
builds its own `LlamaIndexRAGSystem`, handing it the embedding model it has
already loaded and cached.

Two details are worth knowing before changing the code. A `.doc` file is
read with `DocxReader`, the reader for the OOXML format, so a legacy binary
`.doc` fails to parse. And the LlamaIndex settings are process-wide, so the
node parser, the embedding model and the LLM are shared by everything that
runs in the same process.

## Configuration

The values come from a `.env` file in the project root, which python-dotenv
reads at import, or from the environment. Every one of them has a default in
the code, so the file is optional.

| Variable | Default | What it does |
| --- | --- | --- |
| `LM_STUDIO_API_URL` | `http://localhost:1234/v1` | The base URL of the OpenAI-compatible endpoint that writes the answers. |
| `LM_STUDIO_API_KEY` | `lm-studio` | The key sent with the request. LM Studio ignores it. |
| `LM_STUDIO_MODEL` | `gpt-3.5-turbo` | The model name in the request. LM Studio answers with the model it has loaded. |
| `LM_STUDIO_TEMPERATURE` | `0.1` | The sampling temperature. |
| `EMBED_MODEL_NAME` | `Alibaba-NLP/gte-multilingual-base` | The HuggingFace model that produces the embeddings. |
| `EMBED_MODEL_DEVICE` | `cpu` | Where the model runs. `cuda` is the other option. |
| `EMBED_MODEL_TRUST_REMOTE_CODE` | `True` | Whether the code in the model repository is allowed to run. |
| `CHUNK_SIZE` | `384` | Tokens per chunk. |
| `CHUNK_OVERLAP` | `50` | Tokens that two neighbouring chunks share. |
| `TOP_K` | `3` | How many chunks the retriever returns. |
| `USE_HYBRID_SEARCH` | `False` | Adds the BM25 retriever to the vector one. |
| `VERBOSE_RETRIEVAL` | `False` | Logs what the fusion step did. |
| `DATA_PATH` | `data` | The folder the documents are read from. |
| `STORE_PATH` | `chroma_store` | The folder ChromaDB writes to. |
| `LOG_LEVEL` | `INFO` | No code reads it. The level is set in the source: `INFO` in `rag_system.py`, `DEBUG` in `streamlit_rag.py`. |

## Development

```bash
pip install -r requirements.txt   # the 14 pinned dependencies
streamlit run streamlit_rag.py    # the browser interface
python rag_system.py              # the terminal loop
```

The program is the two Python files. There is no test suite, no linter
configuration, no CI workflow and no packaging metadata, so
`pip install -r requirements.txt` is the whole setup.

| Path | What it holds |
| --- | --- |
| `rag_system.py` | `LlamaIndexRAGSystem`: the readers, the chunker, the Chroma collection, the query engine and the terminal loop. |
| `streamlit_rag.py` | The Streamlit app: the file grid, the indexing run, the log console and the chat. |
| `requirements.txt` | 14 dependencies, each pinned to an exact version. |
| `.env.example` | The variables and their defaults. |
| `data/` | The documents. You create it, and `.gitignore` keeps it out of the repository. |
| `chroma_store/` | The vector store. Created on the first run. |

Python 3.10 is the floor, and `streamlit-aggrid==1.1.7` is what sets it.

## License

MIT, © Alexander Blinov. See [LICENSE](LICENSE).
