import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

RAFT_DATA_PATH = Path("data/raft_dataset/raft_train.json")
OUTPUT_DIR     = Path("data/raft_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME     = "microsoft/phi-2"
LORA_R         = 4
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
MAX_LENGTH     = 256
BATCH_SIZE     = 1
GRAD_ACCUM     = 2
LEARNING_RATE  = 2e-4
NUM_EPOCHS     = 1

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"[device] Using: {DEVICE}")

PROMPT_TEMPLATE = """Instruct: {instruction}

{input}

Output: {output}"""

def format_example(example):
    return {"text": PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"][:600],
        output=example["output"],
    )}

def main():
    print("=" * 60)
    print("RAFT ArXiv — QLoRA Fine-tuning (phi-2)")
    print("=" * 60)

    with open(RAFT_DATA_PATH) as f:
        raw_data = json.load(f)
    print(f"\n[load]  {len(raw_data)} training examples")

    formatted  = [format_example(ex) for ex in raw_data]
    split_idx  = int(len(formatted) * 0.9)
    train_data = Dataset.from_list(formatted[:split_idx])
    val_data   = Dataset.from_list(formatted[split_idx:])
    print(f"[split] Train: {len(train_data)} | Val: {len(val_data)}")

    print(f"\n[model] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    train_tok = train_data.map(tokenize, remove_columns=["text"])
    val_tok   = val_data.map(tokenize,   remove_columns=["text"])

    print(f"[model] Loading base model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    if DEVICE == "mps":
        model = model.to("mps")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n[train] Starting fine-tuning...")
    trainer.train()

    adapter_path = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n[done]  LoRA adapter saved → {adapter_path}")
    print("\nNext step: python src/build_baseline_rag.py")

if __name__ == "__main__":
    main()
