# Ask My Notes 🧠 — a mini RAG project

A tiny chatbot that answers questions using **your own PDF/text notes** —
not the internet, not made-up facts, just your files.

## How it works (explained like you're 4)

1. We take your notes and tear them into small paragraphs ("chunks"),
   like index cards.
2. We give every card a "meaning GPS coordinate" (called an embedding)
   using an AI model.
3. When you ask a question, we find the question's GPS coordinate too,
   and show you the cards parked closest to it.

That's the whole trick. No magic, just smart searching.

## Project structure

```
rag-mini-project/
├── data/              <- put your PDFs or .txt notes here
├── src/
│   ├── ingest.py       <- Step 1: reads & chops your files
│   ├── embed_store.py  <- Step 2: turns chunks into numbers, builds search index
│   └── query.py        <- Step 3: ask questions, get answers
├── requirements.txt
└── README.md
```

## How to run it (do these in order)

### 1. Install the tools
```bash
pip install -r requirements.txt
```

### 2. Add your notes
Drop any `.pdf` or `.txt` files into the `data/` folder.

### 3. Build the search index (do this once, or whenever you add new notes)
```bash
cd src
python embed_store.py
```
This reads your notes, turns them into numbers, and saves an `index.faiss`
file — think of it as your searchable "meaning map."

### 4. Ask questions!
```bash
python query.py
```
Type a question, hit enter, get back the matching paragraphs from your
own notes.

## What's next (once this works)

- **Add real generated answers**: instead of just showing chunks, feed
  them into an LLM (like Claude) with a prompt like *"answer using only
  this context: ..."* — this turns "search" into a true chatbot.
- **Add a web interface**: wrap `query.py` in a Streamlit app so you can
  ask questions in a browser instead of a terminal.
- **Try different chunk sizes**: smaller chunks = more precise but less
  context; bigger chunks = more context but less precise.

## Why this project matters

This is called **RAG (Retrieval-Augmented Generation)** — it's one of
the most in-demand skills in applied ML right now, because it's how
real companies build chatbots that answer from *their own* data instead
of hallucinating.
