"""
STEP 1: Read files from the data/ folder and chop them into small chunks.

Think of this like tearing a big book into index cards, a few sentences
on each card, so we can search through the cards later.
"""

import os
from pypdf import PdfReader


def read_pdf(filepath):
    """Read all text out of a PDF file."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def read_txt(filepath):
    """Read all text out of a plain .txt file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Cut a big blob of text into overlapping chunks.

    chunk_size: how many characters per chunk (roughly one paragraph)
    overlap: how many characters repeat between chunks, so we don't
             accidentally cut a sentence in half and lose meaning
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_and_chunk_all(data_folder="data"):
    """
    Look at every file in data_folder, read it, chop it up,
    and return one big list of chunks with their source filename.
    """
    all_chunks = []
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)

        if filename.endswith(".pdf"):
            text = read_pdf(filepath)
        elif filename.endswith(".txt"):
            text = read_txt(filepath)
        else:
            continue  # skip files we don't know how to read

        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append({"text": c, "source": filename})

    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    print(f"Made {len(chunks)} chunks from your files.")
    if chunks:
        print("\nExample chunk:")
        print(chunks[0])
