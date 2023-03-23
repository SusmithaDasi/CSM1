import arxiv
import os
import json
import time
from tqdm import tqdm
from pathlib import Path

SEARCH_QUERIES = [
    "large language models",
    "retrieval augmented generation",
    "fine-tuning transformers",
    "attention mechanism deep learning",
    "prompt engineering",
]
PAPERS_PER_QUERY = 40
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
RAW_DIR          = Path("data/raw")
PROCESSED_DIR    = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def fetch_papers(query, max_results):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "id":       result.entry_id,
            "title":    result.title,
            "abstract": result.summary,
            "authors":  [str(a) for a in result.authors],
            "url":      result.pdf_url,
            "query":    query,
        })
    return papers

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 100]

def parse_abstract_into_chunks(paper):
    text = f"Title: {paper['title']}\n\n{paper['abstract']}"
    chunks = chunk_text(text)
    return [
        {
            "paper_id": paper["id"],
            "title":    paper["title"],
            "chunk_id": f"{paper['id']}_{i}",
            "text":     chunk,
            "source":   "abstract",
        }
        for i, chunk in enumerate(chunks)
    ]

def main():
    print("=" * 60)
    print("RAFT ArXiv — Day 1: Fetching and chunking papers")
    print("=" * 60)

    all_papers = []
    seen_ids   = set()

    for query in SEARCH_QUERIES:
        print(f"\n[fetch] Query: '{query}'")
        papers = fetch_papers(query, PAPERS_PER_QUERY)
        for p in papers:
            if p["id"] not in seen_ids:
                all_papers.append(p)
                seen_ids.add(p["id"])
        print(f"        Got {len(papers)} papers | Total unique: {len(all_papers)}")
        time.sleep(1)

    meta_path = RAW_DIR / "papers_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[save]  Raw metadata → {meta_path} ({len(all_papers)} papers)")

    all_chunks = []
    print("\n[chunk] Splitting papers into text chunks...")
    for paper in tqdm(all_papers):
        chunks = parse_abstract_into_chunks(paper)
        all_chunks.extend(chunks)

    chunks_path = PROCESSED_DIR / "chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n[done]  {len(all_chunks)} chunks saved → {chunks_path}")
    print(f"        Average chunk length: {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} chars")
    print("\nNext step: python src/build_raft_dataset.py")

if __name__ == "__main__":
    main()
