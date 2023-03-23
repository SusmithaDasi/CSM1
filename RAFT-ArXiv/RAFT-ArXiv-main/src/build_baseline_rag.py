import json
import chromadb
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_DIR  = Path("data/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION  = "arxiv_papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 4

def build_index():
    print("[index] Loading chunks...")
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    print(f"[index] Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    embedder   = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids   = [c["chunk_id"] for c in batch]
        metas = [{"title": c["title"], "paper_id": c["paper_id"]} for c in batch]
        embs  = embedder.encode(texts).tolist()
        collection.add(documents=texts, embeddings=embs, ids=ids, metadatas=metas)

    print(f"[index] Indexed {len(chunks)} chunks → {CHROMA_DIR}")
    return embedder, collection

class BaseRAG:
    def __init__(self):
        self.embedder   = SentenceTransformer(EMBED_MODEL)
        client          = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION)

    def retrieve(self, query, top_k=TOP_K):
        query_emb = self.embedder.encode([query]).tolist()
        results   = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        docs = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            docs.append({"rank": i + 1, "text": doc, "title": meta["title"]})
        return docs

    def answer(self, query):
        import ollama
        docs    = self.retrieve(query)
        context = "\n\n".join([f"[Doc {d['rank']}] {d['title']}\n{d['text']}" for d in docs])
        prompt  = f"""Answer the question using only the provided documents.
If the answer is not in the documents, say "I don't know based on the provided context."

Question: {query}

Documents:
{context}

Answer:"""
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        return {
            "query":     query,
            "answer":    response["message"]["content"],
            "retrieved": docs,
            "model":     "base_rag_llama3",
        }

def main():
    print("=" * 60)
    print("RAFT ArXiv — Building Base RAG index")
    print("=" * 60)
    build_index()

    print("\n[test]  Running smoke test...")
    rag    = BaseRAG()
    result = rag.answer("What is retrieval augmented generation?")
    print(f"\nQuery:  {result['query']}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"\n[done]  Base RAG ready.")
    print("Next step: python eval/evaluate.py")

if __name__ == "__main__":
    main()
