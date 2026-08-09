"""
STEP 2: Turn each text chunk into a list of numbers (an "embedding"),
then store all of them in a FAISS index so we can search fast.

Analogy: imagine every paragraph gets a GPS coordinate based on what
it MEANS (not what words it uses). Two paragraphs about "batteries"
will land near each other on this "meaning map," even if one says
"battery" and the other says "energy storage cell."
"""

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ingest import load_and_chunk_all

MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, fast, runs on your laptop
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.pkl"


def build_index(data_folder="data"):
    print("Loading and chunking your documents...")
    chunks = load_and_chunk_all(data_folder)
    print(f"Got {len(chunks)} chunks.")

    print(f"Loading embedding model ({MODEL_NAME})... this downloads once.")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print("Turning chunks into numbers (embeddings)...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("Building the FAISS search index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # simplest kind: straight-line distance
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done! Saved '{INDEX_FILE}' and '{CHUNKS_FILE}'.")
    return index, chunks


if __name__ == "__main__":
    build_index()
