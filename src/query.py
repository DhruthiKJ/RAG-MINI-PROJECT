"""
STEP 3: Ask a question, and get back the paragraphs from YOUR OWN
documents that are most likely to contain the answer.

This is "retrieval" — we're not making anything up, we're finding
the real chunks that match your question's meaning.
"""

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from embed_store import MODEL_NAME, INDEX_FILE, CHUNKS_FILE


def load_everything():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, chunks, model


def ask(question, top_k=3):
    index, chunks, model = load_everything()

    # Turn the question into the same kind of numbers as our chunks
    question_vector = model.encode([question])
    question_vector = np.array(question_vector, dtype="float32")

    # Find the top_k closest chunks on the "meaning map"
    distances, indices = index.search(question_vector, top_k)

    print(f"\nQuestion: {question}\n")
    print("Most relevant chunks from your documents:\n")
    for rank, idx in enumerate(indices[0], start=1):
        chunk = chunks[idx]
        print(f"--- Match {rank} (from {chunk['source']}) ---")
        print(chunk["text"])
        print()


if __name__ == "__main__":
    while True:
        q = input("Ask a question (or type 'quit'): ")
        if q.lower() == "quit":
            break
        ask(q)
