# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

The interface is a Streamlit application (streamlit 1.47.1) that runs on the
operator's own machine. The second entry point is a terminal loop in the
same repository. Everything else is local: the documents, the embeddings and
the vector store. The one network dependency the answer path has is the
OpenAI-compatible endpoint, LM Studio by default.

## Users

The repository carries no description on GitHub and had no README before
this one, so the audience is written down nowhere. What the code says about
the person at the keyboard: they have a folder of PDF and Word documents,
they want to ask questions about them, and they run the language model
themselves through LM Studio. The interface language is Russian, which
points at a Russian-speaking operator.

Nobody else is confirmed. There is no account, no team, no shared store and
no record of who uses it.

## Product Purpose

The product does one thing: it answers questions about a folder of the
operator's own documents.

The documents are read, split into chunks, embedded and stored, and the
answer is written from the retrieved fragments. Every answer carries the
file it came from, the relevance score and the fragment text, because the
person asking has to check the answer against the document.

Success for the user is an answer that names its source. Success for the
product is that the corpus never leaves the machine and that the vector
store is a folder the operator can delete.

## Positioning

The neighbour is a hosted chat-with-your-PDF service, where the documents
are uploaded to somebody else's machine and the model is somebody else's
too. This one keeps the corpus, the embeddings and the store on the
operator's disk, and points the LLM at whatever OpenAI-compatible server
they already run.

The second difference is the source trail: the fragments are shown with the
file name and the score, next to the answer.

The third is the corpus format: a folder of PDF, DOCX and DOC files that the
operator already has, indexed from the interface file by file, without
uploading anything anywhere.

Nothing here is claimed as better. The trade is the one the operator makes
on purpose: a local model and a local store, and the work of running them.

## Operating Context

The operator starts LM Studio, loads a model and leaves its server on port
1234. They drop the documents into `data/`, start the app with
`streamlit run streamlit_rag.py`, and pick the files in the grid on the
indexing tab. The run takes a progress bar, a status line and a log console
that keeps the last 20 lines and shows the last 15 of them.

The first run downloads the embedding model from HuggingFace. The first
index of a folder runs the embedding model over every chunk on the CPU,
which is the slow part.

Asking is the second tab. A question goes out, and the answer comes back
with an expandable list of sources, each rated from "отличная" (0.8 and
above) to "низкая" (below 0.4). The chat history lives in the browser
session and is written nowhere.

The command-line path does the same thing without a browser:
`python rag_system.py` indexes everything under `data/` and then waits for a
question until the word `exit`.

## Capabilities and Constraints

Confirmed by the code:

- PDF, DOCX and DOC files under `DATA_PATH` are read. `.doc` goes through
  the DOCX reader, so a legacy binary `.doc` fails to parse.
- Chunks are 384 tokens with 50 tokens of overlap, and the top 3 chunks are
  retrieved.
- The embedding model is `Alibaba-NLP/gte-multilingual-base` on the CPU.
- The vectors live in a ChromaDB collection named `default` under
  `chroma_store/`.
- The answer comes from an OpenAI-compatible endpoint, LM Studio on
  `http://localhost:1234/v1` by default, with temperature 0.1.
- Hybrid search adds BM25 and fuses the two rankings with reciprocal rank
  fusion. The keyword retriever is built only when the index holds its
  documents in memory.
- Every answer carries the source file, the score and the fragment.
- The browser can index a selection of files, clear the index and show the
  run log.
- A question that fails is answered with the error message and no sources.

Deliberately not true, and worth knowing before a change:

- There are no tests, no CI workflow and no packaging metadata.
- There is no authentication, and the app is served by the Streamlit
  server.
- The chat history is not persisted.
- The corpus is one folder. There is no second collection, no namespace and
  no per-user store.
- The first indexing run needs a network connection for the model download.

Open questions, for the owner to answer:

- Whether `.doc` support is meant to work. The reader cannot parse the
  format it claims to support.
- Whether the file selection in the grid is meant to hand the paths to the
  indexer. It cannot at the moment.
- Whether hybrid search is meant to survive a restart.
- Whether the repository is MIT licensed. GitHub reports no licence for it,
  and the `LICENSE` file here is the house text.

## Brand Commitments

**Name:** simple-rag. **Voice:** the interface is in Russian and speaks in
short sentences and noun phrases: "Индексация документов", "Очистить индекс",
"Ваш вопрос по документам...". The answer panel rates every source with a
word as well as a number: excellent, good, medium, low.

**The interface states what it is doing.** The sidebar carries three
counters (documents found, documents indexed, chat messages), the run has a
progress bar and a log console, and a failure shows its error in the
interface rather than only in the log file.

The name and the marks are the repository's own. There is no logo and no
palette: the interface is the Streamlit default theme with one page icon.

## Evidence on Hand

- `rag_system.py` holds the readers, the chunker, the Chroma collection and
  the query engine, with the defaults in the constructor.
- `streamlit_rag.py` holds the interface: the grid, the indexing run, the
  clear action, the log console and the chat.
- `.env.example` lists 15 variables, and 14 of them are read by the code.
- `requirements.txt` pins 14 packages. `docx2txt`,
  `llama-index-embeddings-openai` and `pillow` are never imported.
- The repository has one commit, `cd2c203`, dated 2025-07-29. Its subject
  line names logging, file indexing, document selection and the feedback the
  interface gives.

Absent, and not to be invented: no usage numbers, no count of indexed
documents, no testimonials, no comparison with other products, no
screenshots of the interface, and no licence file of the repository's own.
The repository carried no description and no README before this one.

## Product Principles

1. **The corpus stays on the machine.** The documents, the embeddings and
   the vector store are local files. A feature that needs a hosted corpus
   changes the product.
2. **Every answer names its source.** The file, the score and the fragment
   travel with the answer, so the person asking can check the reply against
   the document it came from.
3. **The model is the operator's.** The endpoint is OpenAI-compatible and
   the default is LM Studio on the operator's own machine. No hosted model
   is assumed.
4. **The state is visible.** Counts, progress, the run log and the error
   text are in the interface, not only in the log file.
5. **One folder is the corpus.** The scope of the product is a directory of
   PDF and Word files and the questions asked about it.

## Accessibility & Inclusion

- **The interface language is Russian.** A reader who does not read Russian
  cannot use it. The documents in this repository are in English, so the two
  do not have the same audience.
- **Every source is shown as text.** The fragment, the file name and the
  score are in the panel, and the ratings are words as well as numbers.
- **The relevance rating has no standard behind it.** The four thresholds
  (0.8, 0.6, 0.4) are hard-coded in the chat panel, and nothing measures
  them.
- **The grid is mouse-driven.** The file table uses checkboxes in an AG
  Grid, and the keyboard path through it has not been checked.
- **A contrast or screen-reader standard has not been set.** The interface
  uses the Streamlit default theme, and the accessibility behaviour is
  whatever that theme does.
