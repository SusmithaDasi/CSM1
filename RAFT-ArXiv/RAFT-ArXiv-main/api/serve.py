import json
import sys
import torch
import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import chromadb

sys.path.append(".")

app = FastAPI(title="RAFT ArXiv API", version="1.0")

CHROMA_DIR   = Path("data/chroma_db")
ADAPTER_PATH = Path("data/raft_model/lora_adapter")
MODEL_NAME   = "microsoft/phi-2"
EMBED_MODEL  = "all-MiniLM-L6-v2"
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K        = 4

# Global model holders
embedder      = None
collection    = None
raft_model    = None
raft_tokenizer = None

@app.on_event("startup")
def load_models():
    global embedder, collection, raft_model, raft_tokenizer

    print("[startup] Loading embedder...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("[startup] Loading ChromaDB...")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("arxiv_papers")

    print("[startup] Loading RAFT model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    raft_model     = PeftModel.from_pretrained(base, ADAPTER_PATH).to(DEVICE)
    raft_model.eval()
    raft_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    raft_tokenizer.pad_token = raft_tokenizer.eos_token
    print("[startup] All models ready.")

class QueryRequest(BaseModel):
    question: str
    top_k:    int = TOP_K

class AnswerResponse(BaseModel):
    question:  str
    answer:    str
    model:     str
    retrieved: list[dict] = []

def retrieve(query, top_k=TOP_K):
    query_emb = embedder.encode([query]).tolist()
    results   = collection.query(query_embeddings=query_emb, n_results=top_k)
    docs = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        docs.append({"rank": i+1, "text": doc, "title": meta["title"]})
    return docs

@app.get("/health")
def health():
    return {"status": "ok", "models": ["base_rag", "raft"]}

@app.post("/rag/base", response_model=AnswerResponse)
def base_rag(req: QueryRequest):
    docs    = retrieve(req.question, req.top_k)
    context = "\n\n".join([f"[Doc {d['rank']}] {d['title']}\n{d['text']}" for d in docs])
    prompt  = f"""Answer the question using only the provided documents.
If the answer is not in the documents, say "I don't know based on the provided context."

Question: {req.question}

Documents:
{context}

Answer:"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    return AnswerResponse(
        question=req.question,
        answer=response["message"]["content"],
        model="base_rag_llama3",
        retrieved=docs,
    )

@app.post("/rag/raft", response_model=AnswerResponse)
def raft_endpoint(req: QueryRequest):
    docs    = retrieve(req.question, req.top_k)
    context = "\n\n".join([f"[Doc {d['rank']}] {d['title']}\n{d['text']}" for d in docs])
    prompt  = f"""Instruct: Answer the question using only the provided documents.
Question: {req.question}

Documents:
{context[:600]}

Output:"""
    inputs = raft_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = raft_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=raft_tokenizer.eos_token_id,
        )
    answer = raft_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Output:")[-1].strip()
    return AnswerResponse(
        question=req.question,
        answer=answer,
        model="raft_phi2",
        retrieved=docs,
    )

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
def root():
    return FileResponse("api/static/index.html")
