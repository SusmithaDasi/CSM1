import json
import re
import ollama
from pathlib import Path
from tqdm import tqdm

RESULTS_PATH = Path("eval/results/evaluation_results.json")
JUDGE_PATH   = Path("eval/results/judge_results.json")

JUDGE_PROMPT = """Compare two answers to a question. Score each 1-5.

Question: {question}
Expected answer: {expected}

Answer A (Base RAG): {base_answer}
Answer B (RAFT): {raft_answer}

Score rules:
- faithfulness: does the answer stick to the documents? (1=makes things up, 5=fully grounded)
- correctness: does it match the expected answer? (1=wrong, 5=correct)
- conciseness: is it focused and direct? (1=rambling, 5=precise)

Respond with ONLY valid JSON, no explanation:
{{"bf": X, "bc": X, "bn": X, "rf": X, "rc": X, "rn": X, "winner": "base or raft or tie"}}

Where X is your actual score (not X, a real number 1-5)."""

def extract_json(text):
    # Clean newlines before parsing
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    try:
        # Remove newlines inside JSON
        cleaned = re.sub(r'\n\s*', ' ', text)
        match = re.search(r'\{[^{}]+\}', cleaned)
        if match:
            return json.loads(match.group())
    except:
        pass
    try:
        result = {}
        for key in ["bf", "bc", "bn", "rf", "rc", "rn"]:
            m = re.search(rf'"{key}"\s*:\s*(\d)', text)
            if m:
                result[key] = int(m.group(1))
        m = re.search(r'"winner"\s*:\s*"(base|raft|tie)"', text, re.IGNORECASE)
        if m:
            result["winner"] = m.group(1).lower()
        if len(result) == 7:
            return result
    except:
        pass
    return None

def judge(example):
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=example["question"][:150],
                expected=example["expected"][:150],
                base_answer=example["base_answer"][:200],
                raft_answer=example["raft_answer"][:200],
            )}],
            options={"temperature": 0.0}
        )
        raw = response["message"]["content"].strip()
        return extract_json(raw)
    except Exception as e:
        print(f"\n[warn] {e}")
        return None

def main():
    print("=" * 60)
    print("RAFT ArXiv — LLM-as-Judge Evaluation")
    print("=" * 60)

    with open(RESULTS_PATH) as f:
        results = json.load(f)
    print(f"\n[judge] Evaluating {len(results)} examples...")

    judge_results = []
    wins   = {"base": 0, "raft": 0, "tie": 0}
    failed = 0

    for ex in tqdm(results):
        verdict = judge(ex)
        if not verdict or "bf" not in verdict:
            failed += 1
            continue
        judge_results.append({**ex, "verdict": verdict})
        winner = verdict.get("winner", "tie").lower().strip()
        if winner in wins:
            wins[winner] += 1
        else:
            wins["tie"] += 1

    with open(JUDGE_PATH, "w") as f:
        json.dump(judge_results, f, indent=2)

    if not judge_results:
        print("\n[error] All judgements failed.")
        return

    metrics = [
        ("faithfulness", "bf", "rf"),
        ("correctness",  "bc", "rc"),
        ("conciseness",  "bn", "rn"),
    ]

    print("\n" + "=" * 52)
    print("LLM JUDGE RESULTS (scored 1-5)")
    print("=" * 52)
    print(f"{'Metric':<18} {'Base RAG':>10} {'RAFT':>10} {'Winner':>10}")
    print("-" * 52)

    for label, bkey, rkey in metrics:
        b = round(sum(r["verdict"][bkey] for r in judge_results) / len(judge_results), 2)
        r = round(sum(r["verdict"][rkey] for r in judge_results) / len(judge_results), 2)
        winner = "RAFT" if r > b else ("Base RAG" if b > r else "Tie")
        print(f"{label:<18} {b:>10} {r:>10} {winner:>10}")

    total = len(judge_results)
    print(f"\nHead-to-head wins (out of {total}):")
    print(f"  Base RAG : {wins['base']}")
    print(f"  RAFT     : {wins['raft']}")
    print(f"  Tie      : {wins['tie']}")
    print(f"  Failed   : {failed}")
    print(f"\n[done]  Results → {JUDGE_PATH}")

if __name__ == "__main__":
    main()
