import json
import sys
import torch
import ollama
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append(".")
from src.build_baseline_rag import BaseRAG

RAFT_DATA_PATH = Path("data/raft_dataset/raft_train.json")
ADAPTER_PATH   = Path("data/raft_model/lora_adapter")
RESULTS_DIR    = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME     = "microsoft/phi-2"
NUM_TEST       = 15
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"

def load_raft_model():
    print("[raft]  Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH).to(DEVICE)
    model.eval()
    return model, tokenizer

def raft_answer(query, documents, model, tokenizer):
    prompt = f"""Instruct: Answer the question using only the provided documents.
Question: {query}

Documents:
{documents[:600]}

Output:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Output:")[-1].strip()

def score(predicted, expected):
    pred_tokens = set(predicted.lower().split())
    exp_tokens  = set(expected.lower().split())
    if not exp_tokens:
        return {"precision": 0, "recall": 0, "f1": 0}
    precision = len(pred_tokens & exp_tokens) / len(pred_tokens) if pred_tokens else 0
    recall    = len(pred_tokens & exp_tokens) / len(exp_tokens)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {
        "precision":    round(precision, 3),
        "recall":       round(recall, 3),
        "f1":           round(f1, 3),
    }

def main():
    print("=" * 60)
    print("RAFT ArXiv — Head-to-head Evaluation")
    print("=" * 60)

    with open(RAFT_DATA_PATH) as f:
        all_data = json.load(f)
    test_data = all_data[-NUM_TEST:]
    print(f"\n[eval]  Testing on {len(test_data)} examples")

    base_rag            = BaseRAG()
    raft_model, tokenizer = load_raft_model()
    results             = []

    for ex in tqdm(test_data):
        query    = ex["question"]
        expected = ex["oracle_answer"]
        docs     = ex["documents"]

        base_ans = base_rag.answer(query)["answer"]
        raft_ans = raft_answer(query, docs, raft_model, tokenizer)

        results.append({
            "question":    query,
            "expected":    expected,
            "base_answer": base_ans,
            "raft_answer": raft_ans,
            "base_scores": score(base_ans, expected),
            "raft_scores": score(raft_ans, expected),
        })

    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics  = ["precision", "recall", "f1"]
    base_avg = {m: round(sum(r["base_scores"][m] for r in results) / len(results), 3) for m in metrics}
    raft_avg = {m: round(sum(r["raft_scores"][m] for r in results) / len(results), 3) for m in metrics}

    print("\n" + "=" * 45)
    print(f"{'Metric':<15} {'Base RAG':>10} {'RAFT':>10} {'Delta':>10}")
    print("-" * 45)
    for m in metrics:
        delta = round(raft_avg[m] - base_avg[m], 3)
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"{m:<15} {base_avg[m]:>10} {raft_avg[m]:>10}   {arrow} {abs(delta)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RAFT vs Base RAG — Benchmark", fontsize=14, fontweight="bold")

    x, w = range(len(metrics)), 0.35
    ax1  = axes[0]
    ax1.bar([i - w/2 for i in x], [base_avg[m] for m in metrics], w, label="Base RAG", color="#5DCAA5", alpha=0.85)
    ax1.bar([i + w/2 for i in x], [raft_avg[m] for m in metrics], w, label="RAFT",     color="#7F77DD", alpha=0.85)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1); ax1.set_title("Score Comparison")
    ax1.legend(); ax1.set_ylabel("Score"); ax1.grid(axis="y", alpha=0.3)

    deltas = [round(raft_avg[m] - base_avg[m], 3) for m in metrics]
    colors = ["#1D9E75" if d >= 0 else "#D85A30" for d in deltas]
    ax2    = axes[1]
    ax2.bar(metrics, deltas, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("RAFT Improvement over Base RAG")
    ax2.set_ylabel("Delta"); ax2.grid(axis="y", alpha=0.3)
    for i, d in enumerate(deltas):
        ax2.text(i, d + 0.005, f"{'+' if d >= 0 else ''}{d}", ha="center", fontsize=10)

    plt.tight_layout()
    chart_path = RESULTS_DIR / "benchmark_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot]  Chart saved → {chart_path}")
    print(f"[done]  Results → {results_path}")

if __name__ == "__main__":
    main()
