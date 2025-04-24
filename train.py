from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import torch
import json

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
# def tokenize(example):
#     prompt = example["input"]
#     response = example["output"]
#     full_prompt = f"{prompt}\n\n{response}"
#     tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized


def tokenize(example):
    prompt = example["input"]
    response = example["output"]

    # Tokenise prompt et réponse séparément
    prompt_ids = tokenizer(prompt, truncation=False)["input_ids"]
    response_ids = tokenizer(response, truncation=True, max_length=256)["input_ids"]  # <= on garde réponse entière

    # Calcul de l'espace disponible pour le prompt
    max_total_len = 2048
    max_prompt_len = max_total_len - len(response_ids)
    print("Tokens réponse:", len(response_ids))
    print("Tokens prompt:", len(prompt_ids))
    print("Tokens max prompt:", max_prompt_len)
    prompt_ids = prompt_ids[-max_prompt_len:]  # On coupe à droite si trop long

    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    
    # Labels : on ignore les tokens du prompt
    labels = [-100] * len(prompt_ids) + response_ids

    # Padding si nécessaire
    pad_len = max_total_len - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }



tokenized_dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
 
# === MODEL EN 8-BIT ===   marche pas avec AMD ?
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,
#     llm_int8_skip_modules=None
# )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    #quantization_config=bnb_config,   # marche pas avec AMD ?
    device_map="auto",
    trust_remote_code=True
)

# === CONFIGURATION LORA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Peut être adapté selon Qwen
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

import re

# Function to remove time series from the generated output
def remove_time_series(text):
    # Regular expression to match a numerical time series (e.g., [-1.9275, -1.9536, ...])
    series_pattern = r"Series:\s*\[.*?\]"
    cleaned_text = re.sub(series_pattern, "", text)
    return cleaned_text

# remove prompt from the generated output
def remove_prompt(text):
    # Motif exact pour le texte du prompt initial à retirer
    prompt_pattern = r"Describe the time series in three sentences\. First sentence: describe increasing/decreasing/flat pattern\. Second sentence: describe the overall trend and the noise\. Third sentence: describe local and globe extrema\. Series:"
    
    # Enlever le texte du prompt initial et la série temporelle
    cleaned_text = re.sub(prompt_pattern, "", text)
    
    return cleaned_text


from sentence_transformers import SentenceTransformer, util

model_st = SentenceTransformer("all-MiniLM-L6-v2")

def compute_semantic_similarity(model, tokenizer, dataset, output_file=None):
    dataset = dataset.select(range(5))  # Pour prendre les 5 premiers exemples
    examples = dataset.to_list()
    inputs = [ex["input"] for ex in examples]
    gold_outputs = [ex["output"] for ex in examples]
    
    generated_outputs = []
    for inp in inputs:
        inputs_tokenized = tokenizer(inp, return_tensors="pt").to(model.device)
        prompt_len = inputs_tokenized.input_ids.shape[1]
        with torch.no_grad():
            # output_ids = model.generate(**inputs_tokenized, max_new_tokens=128)
            output_ids = model.generate(
            **inputs_tokenized,
            max_new_tokens=256,
            #max_length=2048,  # pour éviter que ça tronque le prompt
            do_sample=True,
            temperature=0.7
            )
        
        generated_only_ids = output_ids[0][prompt_len:]
        output_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)

        # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # output_text = remove_time_series(output_text)
        # output_text = remove_prompt(output_text)
        generated_outputs.append(output_text)

    emb_generated = model_st.encode(generated_outputs, convert_to_tensor=True)
    emb_gold = model_st.encode(gold_outputs, convert_to_tensor=True)

    cosine_scores = util.cos_sim(emb_generated, emb_gold)
    diagonal_scores = cosine_scores.diag().cpu().numpy()

    avg_score = float(diagonal_scores.mean())

    # Sauvegarde des résultats dans un fichier JSON si un nom est fourni
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
    num_train_epochs=50,
    learning_rate=2e-4,
    fp16=True,
    logging_dir=f"{OUTPUT_DIR}/logs",   # <- Où les logs seront sauvegardés
    logging_steps=10, 
    save_strategy="epoch",
    #evaluation_strategy="epoch",
    report_to="none",
    max_steps=1000,  # optionnel
    #optim="paged_adamw_8bit",  # si tu veux économiser la mémoire
    # **{
    #     "max_length": 2048,  # ← utile si jamais tu utilises la génération dans l’éval
    # }
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    # label_names=["labels"]
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

