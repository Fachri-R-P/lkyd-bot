import os

import chromadb
from dotenv import load_dotenv
from groq import Groq

from ingest import COLLECTION_NAME, DB_DIR

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_HISTORY_MESSAGES = 6  # 3 putaran tanya-jawab terakhir

SYSTEM_PROMPT = """\
You are Lee Kuan Yew (1923-2015), the founding Prime Minister of Singapore, \
answering questions in the first person.

Style:
- Speak bluntly and pragmatically: firm, analytical, no sugarcoating.
- Draw on Singapore's experience and on history to illustrate your points.
- Keep answers focused; do not ramble.

Grounding rules:
- You will be given numbered excerpts from your speeches and writings, like [1], [2].
- Base your answer on these excerpts and cite them inline as [1], [2], matching \
the excerpt numbers. Only cite excerpts you actually used.
- Never invent direct quotes. Only put text in quotation marks if it appears \
verbatim in the excerpts.
- If the excerpts are not enough, answer from your well-known principles and views, \
and say clearly that this is reasoning from your known views, not a quotation.
- You died on 23 March 2015. For events after that date, admit you do not know \
about them, then give a view based on your principles.

Language: reply in the same language the user writes in.
"""


def get_client():
    # dibuat saat dipanggil, bukan saat import
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_collection(COLLECTION_NAME)

def to_english_query(question: str) -> str:
    response = get_client().chat.completions.create(
        model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        messages=[
            {
                "role": "system",
                "content": "Translate the user's text to English. If it is already "
                "English, return it unchanged. Output only the text, nothing else.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def retrieve(query, k=5):
    result = get_collection().query(query_texts=[query], n_results=k)
    chunks = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append(
            {
                "text": text,
                "title": meta["title"],
                "date": meta["date"],
                "source_file": meta["source_file"],
                "distance": distance,
            }
        )
    return chunks


def format_context(chunks):
    # kasih nomor [1], [2], ... biar model bisa sitasi
    parts = [
        f"[{i}] {c['title']} ({c['date']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(parts)


def build_messages(question, chunks, history):
    # kutipan cuma ditempel di pertanyaan terbaru, bukan di history
    user_content = (
        f"Excerpts from my speeches and writings:\n\n{format_context(chunks)}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-MAX_HISTORY_MESSAGES:],
        {"role": "user", "content": user_content},
    ]


def answer(question, history=None, k=5):
    history = history or []
    chunks = retrieve(question, k=k)
    messages = build_messages(question, chunks, history)

    response = get_client().chat.completions.create(
        model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        messages=messages,
        temperature=0.4,
    )
    return {"answer": response.choices[0].message.content, "sources": chunks}


if __name__ == "__main__":
    result = answer("Why did Singapore invest so much in training its workers?")
    print(result["answer"])
    print("\nSources:")
    for i, s in enumerate(result["sources"], start=1):
        print(f"[{i}] {s['title']} ({s['date']}) distance={s['distance']:.3f}")
