import json
import random
import time
import re
from pathlib import Path
from tqdm import tqdm
import ollama

CHUNKS_PATH      = Path("data/processed/chunks.json")
RAFT_OUTPUT_PATH = Path("data/raft_dataset/raft_train.json")
RAFT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

NUM_EXAMPLES    = 200
NUM_DISTRACTORS = 3
OLLAMA_MODEL    = "llama3"
ORACLE_RATIO    = 0.8

QA_PROMPT = """Read this text and generate a question and answer.

Text: {chunk}

Rules:
- Question must be answerable from the text only
- Answer must be 1-2 sentences max
- Return ONLY the JSON below, nothing else

{{"question": "write question here", "answer": "write answer here"}}"""

def extract_json(text):
    # Strategy 1: direct parse
    try:
        return json.loads(text.strip())
    except:
        pass
    # Strategy 2: find JSON block
    try:
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    # Strategy 3: extract fields manually
    try:
        q = re.search(r'"question"\s*:\s*"([^"]+)"', text)
        a = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
        if q and a:
            return {"question": q.group(1), "answer": a.group(1)}
    except:
        pass
    return None

def generate_qa(chunk_text):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": QA_PROMPT.format(chunk=chunk_text[:600])}],
            options={"temperature": 0.3}
        )
        raw = response["message"]["content"].strip()
        return extract_json(raw)
    except Exception as e:
        return None

def format_documents(oracle, distractors, include_oracle):
    docs = distractors.copy()
    if include_oracle:
        docs.append(oracle)
    random.shuffle(docs)
    return "\n\n".join([
        f"[Document {i+1}]\nTitle: {d['title']}\n{d['text']}"
        for i, d in enumerate(docs)
    ])

def main():
    print("=" * 60)
    print("RAFT ArXiv — Building dataset (200 examples)")
    print("=" * 60)

    with open(CHUNKS_PATH) as f:
        all_chunks = json.load(f)
    print(f"\n[load]  {len(all_chunks)} chunks available")

    # Shuffle and try more chunks than needed to compensate for failures
    candidates = random.sample(all_chunks, min(350, len(all_chunks)))

    raft_dataset = []
    failed       = 0

    print(f"\n[build] Targeting {NUM_EXAMPLES} examples (trying up to {len(candidates)} chunks)...")
    print(f"        Model: {OLLAMA_MODEL} | Distractors: {NUM_DISTRACTORS}\n")

    pbar = tqdm(candidates)
    for oracle in pbar:
        if len(raft_dataset) >= NUM_EXAMPLES:
            break

        qa = generate_qa(oracle["text"])
        if not qa or "question" not in qa or "answer" not in qa:
            failed += 1
            pbar.set_postfix({"saved": len(raft_dataset), "failed": failed})
            continue

        distractors = [
            c for c in all_chunks
            if c["paper_id"] != oracle["paper_id"]
        ]
        distractor_sample = random.sample(distractors, min(NUM_DISTRACTORS, len(distractors)))
        include_oracle    = random.random() < ORACLE_RATIO
        documents_text    = format_documents(oracle, distractor_sample, include_oracle)

        raft_dataset.append({
            "question":        qa["question"],
            "oracle_answer":   qa["answer"],
            "documents":       documents_text,
            "include_oracle":  include_oracle,
            "oracle_chunk_id": oracle["chunk_id"],
            "oracle_title":    oracle["title"],
            "instruction":     "Answer the question using only the provided documents. Cite the relevant document in your answer.",
            "input":           f"Question: {qa['question']}\n\nDocuments:\n{documents_text}",
            "output":          qa["answer"] if include_oracle else "I don't know based on the provided context.",
        })
        pbar.set_postfix({"saved": len(raft_dataset), "failed": failed})

    with open(RAFT_OUTPUT_PATH, "w") as f:
        json.dump(raft_dataset, f, indent=2)

    print(f"\n[done]  {len(raft_dataset)} examples saved → {RAFT_OUTPUT_PATH}")
    print(f"        Failed: {failed}")
    print(f"        With oracle:    {sum(1 for x in raft_dataset if x['include_oracle'])}")
    print(f"        Without oracle: {sum(1 for x in raft_dataset if not x['include_oracle'])}")
    print("\nNext step: python src/finetune_qlora.py")

if __name__ == "__main__":
    main()
