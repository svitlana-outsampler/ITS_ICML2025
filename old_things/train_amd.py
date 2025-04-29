from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import torch
import json
import re
from sentence_transformers import SentenceTransformer, util

# === CONFIGURATION ===
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "test_jsonl.jsonl"
OUTPUT_DIR = "./qwen2.5-lora-output"

# === CHARGEMENT DU DATASET JSONL ===
def load_jsonl_dataset(path):
    with open(path, 'r') as f:
        lines = [json.loads(l) for l in f]
    return Dataset.from_list(lines)

raw_dataset = load_jsonl_dataset(DATASET_PATH)
split_dataset = raw_dataset.train_test_split(test_size=0.10)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === PREPROCESSING ===
def tokenize(example):
    prompt = example["input"]
    response = example["output"]
    full_prompt = f"{prompt}\n\n{response}"
    tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# === CHARGEMENT DU MODÈLE SANS quantization ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # ← ou torch.float16 si ton ROCm ne supporte pas bf16
    trust_remote_code=True
).to("cuda")  # ou "hip" si nécessaire selon ta version de torch

# === CONFIGURATION LORA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adapter selon Qwen si besoin
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# === FONCTIONS DE NETTOYAGE POUR ÉVALUATION ===
def remove_time_series(text):
    series_pattern = r"Series:\s*\[.*?\]"
    return re.sub(series_pattern, "", text)

def remove_prompt(text):
    prompt_pattern = r"Describe the time series in three sentences\. First sentence: describe increasing/decreasing/flat pattern\. Second sentence: describe the overall trend and the noise\. Third sentence: describe local and globe extrema\. Series:"
    return re.sub(prompt_pattern, "", text)

model_st = SentenceTransformer("all-MiniLM-L6-v2")

def compute_semantic_similarity(model, tokenizer, dataset, output_file=None):
    dataset = dataset.select(range(5))
    examples = dataset.to_list()
    inputs = [ex["input"] for ex in examples]
    gold_outputs = [ex["output"] for ex in examples]
    
    generated_outputs = []
    for inp in inputs:
        inputs_tokenized = tokenizer(inp, return_tensors="pt").to("cuda")
        prompt_len = inputs_tokenized.input_ids.shape[1]
        with torch.no_grad():
            output_ids = model.generate(**inputs_tokenized, max_new_tokens=128)
        generated_only_ids = output_ids[0][prompt_len:]
        output_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
        generated_outputs.append(output_text)

    emb_generated = model_st.encode(generated_outputs, convert_to_tensor=True)
    emb_gold = model_st.encode(gold_outputs, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_generated, emb_gold)
    avg_score = float(cosine_scores.diag().cpu().numpy().mean())

    if output_file:
        output_data = [{"input": inp, "generated_output": gen_out, "gold_output": gold_out} 
                       for inp, gen_out, gold_out in zip(inputs, generated_outputs, gold_outputs)]
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

    return avg_score

# === ENTRAÎNEMENT ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=100,
    learning_rate=2e-4,
    bf16=True,  # ou fp16=True selon ce qui est supporté
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10, 
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

print("\nÉvaluation avant entraînement (similarité sémantique)...")
score_before = compute_semantic_similarity(model, tokenizer, dataset["test"], output_file="evaluation_avant.json")
print(f"Score moyen avant entraînement : {score_before:.4f}")

trainer.train()

print("\nÉvaluation après entraînement (similarité sémantique)...")
score_after = compute_semantic_similarity(model, tokenizer, dataset["test"], output_file="evaluation_apres.json")
print(f"Score moyen après entraînement : {score_after:.4f}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
