# What Would Lee Kuan Yew Do?

A chatbot that answers questions about life, geopolitics and history in the voice of
Lee Kuan Yew (LKY), grounded in excerpts from his public speeches. Built as a
retrieval-augmented generation (RAG) pipeline written from scratch, without
LangChain or LlamaIndex, so every stage is visible in the code.

> **Disclaimer:** this is an AI simulation, not the real Lee Kuan Yew. Answers may be
> inaccurate or misleading. Check the cited sources.

## Architecture

```
                 ingest.py (run once)
 data/speeches/*.txt ──► clean paragraphs ──► chunks (~800 chars, ~150 overlap)
                                                   │ embed (all-MiniLM-L6-v2)
                                                   ▼
                                       Chroma collection "lky_speeches"
                                                   │
 ┌──────────────── rag.py (per question) ──────────┘
 │
 │ user question ──► retrieve top-5 chunks (cosine distance)
 │                        │
 │                        ▼
 │   build_messages: system prompt (LKY persona + grounding rules)
 │                   + last 3 exchanges of history
 │                   + numbered excerpts [1]..[5] + question
 │                        │
 │                        ▼
 │                Groq LLM (openai/gpt-oss-120b)
 │                        │
 └──────────► answer text with [n] citations + list of sources
                          │
                          ▼
                 app.py (Streamlit chat UI, "Sources" expander)
```

| File | Role |
|---|---|
| `ingest.py` | Reads speeches, chunks them, builds the Chroma index (`build_index()`) |
| `rag.py` | `retrieve`, `build_messages`, `answer`, and the persona `SYSTEM_PROMPT` |
| `app.py` | Streamlit chat UI; builds the index automatically if it is missing |
| `eval/` | Questions, evaluation script and results |

## Data

- 10 speeches and addresses by Lee Kuan Yew (1964-2010), taken from the public
  [National Archives of Singapore](https://www.nas.gov.sg/archivesonline/) speech
  archive. Exact URLs are in [`data/sources.md`](data/sources.md).
- File naming: `YYYY-MM-DD_short-title.txt`; the date and title shown in the app come
  from the file name.
- **Limitations:** only public speeches and transcripts are included. Copyrighted
  memoirs and books are deliberately excluded. The corpus is small, English-only, and
  does not cover every topic LKY discussed.

## Model choices

- **LLM: Groq, `openai/gpt-oss-120b`** (configurable via `GROQ_MODEL`). Groq has a
  free tier and fast inference, and a 120B open-weight reasoning model follows the
  persona and citation instructions well.
- **Embeddings: Chroma's default `all-MiniLM-L6-v2`.** Groq offers no embedding API.
  This model is small, runs locally on CPU and needs no extra service. It is
  English-only (see limitations).
- **Vector store: Chroma `PersistentClient`** in `./chroma_db` with cosine distance.
  Zero setup, and enough for a corpus this size.
- **No orchestration framework.** Each RAG stage is a short, readable function.

## Run locally

Requires Python 3.10+ and a free [Groq API key](https://console.groq.com/keys).

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # Windows: Copy-Item .env.example .env
# edit .env and set GROQ_API_KEY

python ingest.py            # build the index (optional: app.py does it automatically)
streamlit run app.py
```

The first run downloads the embedding model (~80 MB).

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (make sure `.env` is not committed).
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing to
   `app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
4. The app builds the Chroma index automatically on first start.

If Chroma fails with an error about the `sqlite3` version, add this line to
`requirements.txt`. `app.py` already swaps it in when available:

```
pysqlite3-binary; sys_platform == "linux"
```

## Evaluation

Run: `python eval/run_eval.py` → writes `eval/results.csv` and `eval/results.md`.

**Method**
- 15 questions in `eval/questions.json`, in four categories: `in_corpus` (7, with an
  expected source speech), `general_view` (4), `post_2015` (3, events after LKY's death)
  and `out_of_scope` (1).
- **Retrieval hit rate:** is the expected source speech among the top-5 retrieved
  chunks? Only measured for questions with an expected source.
- **LLM-as-judge** (Groq, temperature 0, JSON output) scores each answer 1-5 for
  *faithfulness* to the excerpts, *relevance* to the question and key points, and
  *persona* (does it sound like LKY), with a short reason. The rubric is
  category-aware: for `post_2015` the answer must admit LKY cannot know the event.

**Results** (see `eval/results.md`)

| Category | N | Hit rate | Faithfulness | Relevance | Persona |
|---|---|---|---|---|---|
| all | 15 | 4/7 (57%) | 4.00 | 4.27 | 4.53 |
| in_corpus | 7 | 4/7 (57%) | 3.86 | 4.43 | 4.43 |
| general_view | 4 | n/a | 4.75 | 4.75 | 5.00 |
| post_2015 | 3 | n/a | 4.00 | 4.33 | 4.67 |
| out_of_scope | 1 | n/a | 2.00 | 1.00 | 3.00 |

**Reading the results**
- Persona and relevance are strong, but retrieval is the weak spot: the expected speech
  was in the top-5 for only 4 of 7 in-corpus questions.
- The one out-of-scope question (a recipe) scored poorly: the bot is not yet good at
  declining off-topic requests.
- Caveats: the same model family generates and judges answers (self-preference bias is
  likely), and 15 questions are too few for statistical conclusions. Treat the numbers
  as indicative.

## Example conversation

<!-- Paste 2-3 real exchanges from the running app here. -->

## Limitations and future work

- **Small, English-only corpus.** Coverage is limited to 10 speeches.
- **English-only embeddings.** Non-English questions retrieve poorly even though the
  bot answers in the user's language. Planned fix: translate the query to English
  before retrieval, or use a multilingual embedding model.
- **Retrieval quality.** Hit rate is 57% on in-corpus questions. Ideas: smaller or
  semantic chunking, hybrid search (BM25 + vectors), a re-ranker, larger k.
- **Follow-up questions.** Retrieval uses only the latest message, so questions like
  "why is that?" retrieve poorly. Planned fix: rewrite follow-ups into standalone
  queries using the history.
- **Off-topic handling.** Out-of-scope questions are not refused reliably; the system
  prompt and eval should be extended.
- **Evaluation.** Use a separate judge model, a larger question set, and human spot
  checks.
- **Persona fidelity.** The bot imitates a style from a few speeches; it can still
  put words in LKY's mouth, even with the no-invented-quotes rule.
