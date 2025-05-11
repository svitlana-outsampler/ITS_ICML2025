from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import torch
import json
import os
import re
import logging

from sentence_transformers import SentenceTransformer, util

# === CONFIGURATION ===
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "test_jsonl.jsonl"

# Extraire le nom court du modèle
model_name_parts = MODEL_NAME.split('/')
model_short_name = model_name_parts[-1] if model_name_parts else "unknown_model"
OUTPUT_DIR = f"./{model_short_name.lower()}-lora-output"

print("Training outputs saved in:", OUTPUT_DIR)

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

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
####### xxxxx tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === PREPROCESSING ===
def tokenize(example):
    prompt = example["input"]
    response = example["output"]

    # Tokenise prompt et réponse séparément
    prompt_ids = tokenizer(prompt, truncation=False)["input_ids"]
    response_ids = tokenizer(response, truncation=True, max_length=256)["input_ids"]

    # Calcul de l'espace disponible pour le prompt
    max_total_len = 1024
    max_prompt_len = max_total_len - len(response_ids)
    prompt_ids = prompt_ids[-max_prompt_len:]  # Coupe si trop long

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

print("Tokenisation des données...")
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
print("Tokenisation terminée.")

# === MODÈLE ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)

# === CONFIGURATION LORA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# === MODÈLE DE SIMILARITÉ SÉMANTIQUE ===
model_st = SentenceTransformer("all-MiniLM-L6-v2")

# === FONCTION D'ÉVALUATION SÉMANTIQUE ===
def compute_semantic_similarity(model, tokenizer, dataset, output_file=None):
    model.eval()
    examples = dataset.to_list()
    inputs = [ex["input"] for ex in examples]
    gold_outputs = [ex["output"] for ex in examples]
    
    generated_outputs = []
    for inp in inputs:
        inputs_tokenized = tokenizer(inp, return_tensors="pt").to(model.device)
        prompt_len = inputs_tokenized.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs_tokenized,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )

        generated_only_ids = output_ids[0][prompt_len:]
        output_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
        generated_outputs.append(output_text)

    emb_generated = model_st.encode(generated_outputs, convert_to_tensor=True)
    emb_gold = model_st.encode(gold_outputs, convert_to_tensor=True)
    scores = torch.nn.functional.cosine_similarity(emb_generated, emb_gold)
    avg_score = float(scores.mean())

    if output_file:
        output_data = [{
            "input": inp, 
            "generated_output": gen_out, 
            "gold_output": gold_out, 
            "score": float(score)
        } for inp, gen_out, gold_out, score in zip(inputs, generated_outputs, gold_outputs, scores)]
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
    
    model.train()
    return avg_score

# === CALLBACK POUR ÉVALUATION À CHAQUE CHECKPOINT ===
class SemanticSimilarityCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_dataset, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        print(f"🔄 Étape terminée : {state.global_step}")
        
    def on_evaluate(self, args, state, control, **kwargs):
        print("\n✨ Évaluation de la similarité sémantique à la sauvegarde du checkpoint ✨\n")
        if not hasattr(self, "model") or self.model is None:
            print("Erreur : self.model est None.")
            return
        
        #trainer.model.eval()
        step = state.global_step
        print(f"création du fichier d'évaluation pour le checkpoint {step}")
        output_file = os.path.join(self.output_dir, f"evaluation_checkpoint-{step}.json")
        print(f"fichier d'évaluation : {output_file}")
        
        print(f"calcul de la similarité sémantique pour le checkpoint {step}")
        score = compute_semantic_similarity(
            self.model, 
            self.tokenizer, 
            self.test_dataset, 
            output_file=output_file
        )
        
        #trainer.model.train()
        print(f"\n✅ Checkpoint {step}: Similarité sémantique = {score:.4f}\n")

# === ENTRAÎNEMENT ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=7,
    learning_rate=2e-4,
    bf16=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=10,
    report_to="none",
    max_steps=160,
    disable_tqdm=False
)

# Configurer le logging
logging.basicConfig(level=logging.INFO, force=True)

# Créer le répertoire de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    callbacks=[SemanticSimilarityCallback(model, tokenizer, dataset["test"], OUTPUT_DIR)],
)

# Ajouter le callback d'évaluation sémantique
# semantic_callback = SemanticSimilarityCallback(tokenizer, dataset["test"], OUTPUT_DIR)
# trainer.add_callback(semantic_callback)

# Évaluation initiale avant entraînement
print("\n🔍 Évaluation avant entraînement (similarité sémantique)...")
score_before = compute_semantic_similarity(model, tokenizer, dataset["test"], output_file=os.path.join(OUTPUT_DIR, "evaluation_avant.json"))
print(f"Score moyen avant entraînement : {score_before:.4f}")

# Entraînement
trainer.train()

# Évaluation finale après entraînement
print("\n📊 Évaluation après entraînement (similarité sémantique)...")
score_after = compute_semantic_similarity(model, tokenizer, dataset["test"], output_file=os.path.join(OUTPUT_DIR, "evaluation_apres.json"))
print(f"Score moyen après entraînement : {score_after:.4f}")

# Sauvegarder le modèle final
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)