import re
from pathlib import Path

import chromadb

BASE_DIR = Path(__file__).resolve().parent
SPEECH_DIR = BASE_DIR / "data" / "speeches" #txt pidato
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "lky_speeches"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def parse_filename(path):
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_(.+)$", path.stem)
    if match:
        date, slug = match.groups()
    else:
        date, slug = "unknown", path.stem
    return {
        "date": date,
        "title": slug.replace("-", " ").title(),
        "source_file": path.name,
    }

def split_long_paragraph(paragraph, size):
    pieces = []
    while len(paragraph) > size:
        cut = paragraph.rfind(" ", 0, size)
        if cut <= 0:
            cut = size
        pieces.append(paragraph[:cut])
        paragraph = paragraph[cut:].lstrip()
    pieces.append(paragraph)
    return pieces

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = [piece for p in paragraphs for piece in split_long_paragraph(p, size)]

    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            #overlap diambil dari chunk sebelumnya
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:]
            current = f"{tail}\n\n{piece}"
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks

def build_index():
    files = sorted(SPEECH_DIR.glob("*.txt"))

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0
    for path in files:
        meta = parse_filename(path)
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        if not chunks:
            continue
        collection.add(
            ids=[f"{path.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{**meta, "chunk_index": i} for i in range(len(chunks))],
        )
        total_chunks += len(chunks)

    return len(files), total_chunks

if __name__ == "__main__":
    n_files, n_chunks = build_index()
    print(f"{n_files} file, {n_chunks} chunk masuk index")