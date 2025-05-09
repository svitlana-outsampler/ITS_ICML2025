#! python
#the previous line makes the script executable
import torch
import numpy as np
import json
import logging
import os
import argparse
import math
from typing import Dict, List, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import SentenceTransformer, util
import evaluate # For perplexity calculation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger(__name__)

# === HELPER FUNCTIONS ===

def load_jsonl_dataset(num_samples: int, path: str) -> Dataset:
    """Loads a dataset from a JSON Lines file."""
    try:
        # take only the num_samples fisrst lines of the file f
        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for i, l in enumerate(f) if i < num_samples]
        if not lines:
            raise ValueError(f"Dataset file is empty: {path}")
        # Check if required columns exist
        if not all(key in lines[0] for key in ["input", "output"]):
             raise ValueError("Dataset must contain 'input' and 'output' columns.")
        return Dataset.from_list(lines)
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset {path}: {e}")
        raise

# def create_tokenizer(model_name_or_path: str) -> AutoTokenizer:
#     """Creates and configures the tokenizer."""
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
#         tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer
def create_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """Creates and configures the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    logger.info(f"Tokenizer for {model_name_or_path} created")
    logger.info(f"Tokenizer EOS token: {tokenizer.eos_token}")
    logger.info(f"Tokenizer PAD token: {tokenizer.pad_token}")
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure padding is applied on the right side
    tokenizer.padding_side = 'right'  # Add this line
    # test: tokenize a string to see if the tokenizer is working
    test_string = "01"
    test_tokens = tokenizer(test_string)
    logger.info(f"{test_string=} Test tokens: {test_tokens}")
    test_string = "1555"
    test_tokens = tokenizer(test_string)
    logger.info(f"{test_string=} Test tokens: {test_tokens}")
    test_string = "15"
    test_tokens = tokenizer(test_string)
    logger.info(f"{test_string=} Test tokens: {test_tokens}")
    test_string = "zorro is a fox"
    test_tokens = tokenizer(test_string)
    logger.info(f"{test_string=} Test tokens: {test_tokens}")
    test_string = "fifteen is 15"
    test_tokens = tokenizer(test_string)
    logger.info(f"{test_string=} Test tokens: {test_tokens}")
    #exit()
    return tokenizer

def preprocess_data(examples: Dict[str, List[Any]], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, List[int]]:
    """
    Tokenizes input and output pairs, preparing them for causal LM training.
    Labels are set to -100 for prompt tokens.
    Handles potential truncation by prioritizing the response and recent prompt context.
    """
    processed_batch = {"input_ids": [], "attention_mask": [], "labels": []}
    
    prompts = examples["input"]
    responses = examples["output"]

    for prompt, response in zip(prompts, responses):
        # Tokenize prompt and response separately (without padding initially)
        # Using add_special_tokens=False might be needed depending on the model if you manually add EOS later
        prompt_ids = tokenizer(prompt, truncation=False, add_special_tokens=False)["input_ids"]
        # Add EOS token to response if model expects it
        response_with_eos = response + tokenizer.eos_token
        response_ids = tokenizer(response_with_eos, truncation=False, add_special_tokens=True)["input_ids"]

        # Calculate available space for the prompt
        max_prompt_len = max_length - len(response_ids)

        if max_prompt_len <= 0:
            # Response itself is too long, truncate response (might lose info)
            logger.warning(f"Response is longer than max_length ({max_length}). Truncating response.")
            response_ids = response_ids[:max_length - 1] # Leave space for potential EOS if needed elsewhere
            prompt_ids = [] # No space left for prompt
        elif len(prompt_ids) > max_prompt_len:
            # Prompt needs truncation, keep the end (more recent context)
            logger.warning(f"Prompt is longer than available space ({max_prompt_len}). Truncating prompt.")
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + response_ids
        # check that the last token is the EOS token
        if input_ids[-1] != tokenizer.eos_token_id:
            exit()
        attention_mask = [1] * len(input_ids)
        # Labels: -100 for prompt tokens, actual token IDs for response tokens
        labels = [-100] * len(prompt_ids) + response_ids

        # Padding
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len # Padding tokens should also be ignored in loss

        # Ensure lengths match max_length after processing
        if len(input_ids) != max_length or len(attention_mask) != max_length or len(labels) != max_length:
             logger.error(f"Length mismatch after processing! Expected {max_length}, Got: input_ids={len(input_ids)}, attention_mask={len(attention_mask)}, labels={len(labels)}")
             # Handle error appropriately, e.g., skip this example or raise an exception
             # For now, let's pad/truncate aggressively to fit, though this might hide issues
             input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(input_ids))
             attention_mask = attention_mask[:max_length] + [0] * (max_length - len(attention_mask))
             labels = labels[:max_length] + [-100] * (max_length - len(labels))


        processed_batch["input_ids"].append(input_ids)
        processed_batch["attention_mask"].append(attention_mask)
        processed_batch["labels"].append(labels)

    return processed_batch

def create_model(model_name_or_path: str, quantization: str, lora_config: LoraConfig) -> AutoModelForCausalLM:
    """Loads the base model, applies quantization, and wraps with PEFT."""
    bnb_config = None
    load_in_4bit = False
    load_in_8bit = False
    torch_dtype = torch.float16 # Default to float16 for efficiency

    if quantization == "8bit":
        logger.info("Applying 8-bit quantization.")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        load_in_8bit = True
         # Note: 8-bit might have compatibility issues on some hardware (like older AMD GPUs)
    elif quantization == "4bit":
        logger.info("Applying 4-bit quantization (NF4).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # Or float16 depending on hardware
        )
        load_in_4bit = True
        torch_dtype = torch.bfloat16 # Often recommended with 4-bit

    else:
        logger.info("No quantization applied. Loading in float16.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distributes model layers across available devices
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        # use_flash_attention_2=True # Optional: Requires flash-attn library and compatible hardware
    )

    if load_in_4bit or load_in_8bit:
        logger.info("Preparing model for k-bit training.")
        model = prepare_model_for_kbit_training(model)

    logger.info("Applying LoRA configuration.")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model

# === Evaluation Metrics ===
perplexity_metric = evaluate.load("perplexity", module_type="metric")
sbert_model = None # Global variable for Sentence Transformer model

def compute_metrics(eval_preds):
    """Computes perplexity during evaluation."""
    logits, labels = eval_preds
    
    # Convert numpy arrays to PyTorch tensors if needed
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # Shift logits and labels for autoregressive models
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten tokens and filter out padding (-100)
    valid_labels_mask = shift_labels != -100
    active_logits = shift_logits[valid_labels_mask]
    active_labels = shift_labels[valid_labels_mask]

    # Calculate perplexity only if there are valid labels
    if active_labels.numel() > 0:
        try:
            # Convert back to numpy for the perplexity metric if needed
            results = perplexity_metric.compute(
                predictions=active_logits.numpy() if isinstance(active_logits, torch.Tensor) else active_logits,
                model_id='gpt2'  # model_id is just a placeholder here
            )
            return {"perplexity": results["mean_perplexity"]}
        except Exception as e:
            logger.warning(f"Could not compute perplexity: {e}")
            return {"perplexity": float('inf')}
    else:
        logger.warning("No valid labels found in batch for perplexity calculation.")
        return {"perplexity": float('inf')}

# === Custom Evaluation Callback ===
class SemanticSimilarityCallback(TrainerCallback):
    """Callback to compute semantic similarity using Sentence Transformers."""
    def __init__(self, eval_dataset: Dataset, tokenizer: AutoTokenizer, output_dir: str, sbert_model_name: str = "all-MiniLM-L6-v2", generation_kwargs: Dict = None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.sbert_model_name = sbert_model_name
        self.sbert_model = None # Lazy load
        self.generation_kwargs = generation_kwargs or {}
        os.makedirs(os.path.join(self.output_dir, "custom_eval"), exist_ok=True)

    def _load_sbert_model(self):
        if self.sbert_model is None:
            logger.info(f"Loading Sentence Transformer model: {self.sbert_model_name}")
            self.sbert_model = SentenceTransformer(self.sbert_model_name)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Evaluate before training starts."""
        logger.info("Performing initial semantic similarity evaluation...")
        output_file = os.path.join(self.output_dir, "custom_eval", "evaluation_before_training.json")
        score = self.compute_similarity(model, output_file)
        logger.info(f"Initial semantic similarity score: {score:.4f}")
        # Log to trainer state if possible (requires integration, e.g., adding to state.log_history)
        state.log_history.append({"semantic_similarity_before": score, "step": 0})


    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Evaluate after each evaluation phase triggered by evaluation_strategy."""
        logger.info("Performing semantic similarity evaluation during training...")
        output_file = os.path.join(self.output_dir, "custom_eval", f"evaluation_step_{state.global_step}.json")
        score = self.compute_similarity(model, output_file)
        logger.info(f"Semantic similarity score at step {state.global_step}: {score:.4f}")
        
        # The Trainer's log method automatically handles logging to configured reporters
        if state.is_world_process_zero:  # Ensure logging only happens once in distributed settings
            # Get the trainer instance from kwargs if available
            trainer = kwargs.get('trainer', None)
            if trainer is not None and hasattr(trainer, 'log'):
                trainer.log({"semantic_similarity": score})
            else:  # Fallback for direct logging if trainer instance isn't available
                state.log_history.append({"semantic_similarity": score, "step": state.global_step})

    def compute_similarity(self, model, output_file=None) -> float:
        """Computes semantic similarity and optionally saves results."""
        self._load_sbert_model() # Ensure SBERT model is loaded
        if not self.sbert_model:
             logger.error("SBERT model not loaded, cannot compute similarity.")
             return 0.0

        logger.info(f"Evaluating on {len(self.eval_dataset)} examples for semantic similarity.")
        model.eval() # Ensure model is in eval mode

        inputs = [ex["input"] for ex in self.eval_dataset]
        gold_outputs = [ex["output"] for ex in self.eval_dataset]
        generated_outputs = []

        # --- Generation Loop ---
        # Consider batching if performance is critical and VRAM allows
        # generation_batch_size = 4 # Example batch size
        # for i in range(0, len(inputs), generation_batch_size):
        #     batch_inputs = inputs[i:i+generation_batch_size]
        #     inputs_tokenized = self.tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device) # Adjust max_length as needed
        #     prompt_len = inputs_tokenized.input_ids.shape[1] # Be careful with padding - need mask or individual lengths if not left-padded

        for inp in inputs: # Simpler per-example generation
            try:
                # Ensure input fits within model context, potentially truncate
                inputs_tokenized = self.tokenizer(inp, return_tensors="pt").to(model.device) # Max length for input only
                prompt_len = inputs_tokenized.input_ids.shape[1]

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs_tokenized,
                        max_new_tokens=self.generation_kwargs.get('max_new_tokens', 256),
                        do_sample=self.generation_kwargs.get('do_sample', True),
                        temperature=self.generation_kwargs.get('temperature', 0.7),
                        pad_token_id=self.tokenizer.pad_token_id, # Important for generation
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                # Decode only the newly generated tokens
                generated_only_ids = output_ids[0][prompt_len:]
                output_text = self.tokenizer.decode(generated_only_ids, skip_special_tokens=True)
                generated_outputs.append(output_text.strip())

            except Exception as e:
                logger.error(f"Error during generation for input: {inp[:100]}... Error: {e}")
                generated_outputs.append("[GENERATION ERROR]") # Placeholder on error

        # --- Similarity Calculation ---
        if not generated_outputs:
             logger.warning("No outputs were generated.")
             return 0.0

        try:
            emb_generated = self.sbert_model.encode(generated_outputs, convert_to_tensor=True, show_progress_bar=True)
            emb_gold = self.sbert_model.encode(gold_outputs, convert_to_tensor=True, show_progress_bar=True)

            # Ensure embeddings are on the same device for cosine similarity
            emb_generated = emb_generated.to(emb_gold.device)

            if emb_generated.shape[0] != emb_gold.shape[0]:
                 logger.error(f"Mismatch in number of generated ({emb_generated.shape[0]}) and gold ({emb_gold.shape[0]}) embeddings.")
                 # Handle mismatch, e.g., only score pairs that exist
                 min_len = min(emb_generated.shape[0], emb_gold.shape[0])
                 emb_generated = emb_generated[:min_len]
                 emb_gold = emb_gold[:min_len]
                 # Adjust inputs/outputs lists if saving results
                 inputs = inputs[:min_len]
                 generated_outputs = generated_outputs[:min_len]
                 gold_outputs = gold_outputs[:min_len]


            if emb_generated.shape[0] == 0:
                logger.warning("No valid embeddings generated for scoring.")
                return 0.0

            diagonal_scores = torch.nn.functional.cosine_similarity(emb_generated, emb_gold).cpu().numpy()
            avg_score = float(diagonal_scores.mean()) if diagonal_scores.size > 0 else 0.0

        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            avg_score = 0.0 # Indicate failure
            diagonal_scores = [] # Ensure it exists for saving

        # --- Save Results ---
        if output_file:
            logger.info(f"Saving custom evaluation results to: {output_file}")
            output_data = [{"input": inp, "generated_output": gen_out, "gold_output": gold_out, "score": float(diagonal_scores[i]) if i < len(diagonal_scores) else None}
                           for i, (inp, gen_out, gold_out) in enumerate(zip(inputs, generated_outputs, gold_outputs))]
            try:
                with open(output_file, "w", encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save custom evaluation results: {e}")

        model.train() # Switch back to train mode
        return avg_score


# === MAIN SCRIPT LOGIC ===
def main(args):
    logger.info("Starting fine-tuning script...")
    logger.info(f"Script arguments: {args}")

    # --- 1. Load and Prepare Dataset ---
    logger.info(f"Loading dataset from: {args.dataset_path}")
    raw_dataset = load_jsonl_dataset(1000,args.dataset_path)

    logger.info("Splitting dataset into train and test sets...")
    split_dataset = raw_dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    dataset = DatasetDict({
        "train": split_dataset["train"],
        "test": split_dataset["test"]
    })
    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Test size: {len(dataset['test'])}")

    # --- 2. Create Tokenizer ---
    logger.info(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = create_tokenizer(args.model_name)

    # --- 3. Tokenize Dataset ---
    logger.info("Tokenizing dataset...")
    # Use functools.partial to pass static arguments to the map function
    from functools import partial
    tokenize_fn = partial(preprocess_data, tokenizer=tokenizer, max_length=args.max_seq_length)

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=args.preprocessing_num_workers, # Use multiple processes for speed
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=not args.overwrite_cache, # Use cache if available
        desc="Running tokenizer on dataset",
    )
    logger.info("Dataset tokenization complete.")
    logger.info(f"Example tokenized input keys: {tokenized_dataset['train'].features.keys()}")
    # Log an example to check lengths and padding/labels
    logger.info(f"Example input_ids[0]: {tokenized_dataset['train'][0]['input_ids'][:50]}...") # Show start
    logger.info(f"Example labels[0]: {tokenized_dataset['train'][0]['labels'][:50]}...")      # Show start
    logger.info(f"Example input_ids length: {len(tokenized_dataset['train'][0]['input_ids'])}")
    logger.info(f"Example labels length: {len(tokenized_dataset['train'][0]['labels'])}")


    # --- 4. Configure LoRA ---
    logger.info("Configuring LoRA...")
    # Adapt target_modules based on the model architecture if needed.
    # Common targets for Qwen-like models might include 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
    # Inspect the model architecture (print(model)) to find Linear layers if defaults don't work well.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules or ["q_proj", "v_proj"], # Use default or provided list
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # --- 5. Load Model ---
    logger.info(f"Loading base model ({args.model_name}) with quantization: {args.quantization}")
    model = create_model(args.model_name, args.quantization, lora_config)


    # --- 6. Configure Training Arguments ---
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    logger.info(f"Setting up Training Arguments (Output Dir: {args.output_dir})")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        # fp16=True if args.quantization == 'none' else False, # FP16 recommended if not quantizing heavily
        # bf16=True if torch.cuda.is_bf16_supported() and args.quantization != '8bit' else False, # BF16 often better if supported
        fp16 = not (args.quantization in ['4bit', '8bit']), # Enable FP16 only if not using bitsandbytes quantization which handles its own types
        bf16 = args.quantization in ['4bit', '8bit'] and torch.cuda.is_bf16_supported(), # Use BF16 with quantization if supported

        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if args.eval_steps is None else "steps",
        eval_steps=args.eval_steps if args.eval_steps is not None else None, # Evaluate every N steps if specified, else per epoch
        save_strategy="epoch" if args.save_steps is None else "steps",
        save_steps=args.save_steps if args.save_steps is not None else None, # Save checkpoint every N steps if specified, else per epoch
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True, # Load the best checkpoint (based on eval loss/metric) at the end
        metric_for_best_model="loss", # Or "perplexity" if computed, or custom metric name
        greater_is_better=False, # For loss/perplexity, lower is better
        report_to=args.report_to.split(',') if args.report_to else "tensorboard", # Report to wandb, tensorboard, or none
        seed=args.seed,
        # Optimizations
        optim="paged_adamw_8bit" if args.quantization != 'none' else "adamw_torch", # Paged AdamW good for quantized models
        gradient_checkpointing=args.gradient_checkpointing, # Saves memory at cost of compute
        # ddp_find_unused_parameters=False, # Set to False if encountering issues with DDP and LoRA
        per_device_eval_batch_size=1,  # Reduced from default (same as train batch size)
        fp16_full_eval=True,  # Use FP16 for evaluation even if training in BF16
        eval_accumulation_steps=2,  # Accumulate evaluation results to save memory
    )

    # --- 7. Initialize Trainer ---
    logger.info("Initializing Trainer...")

    # Setup custom evaluation callback
    generation_params = {
        'max_new_tokens': args.eval_max_new_tokens,
        'do_sample': args.eval_do_sample,
        'temperature': args.eval_temperature,
    }
    custom_eval_callback = SemanticSimilarityCallback(
        eval_dataset=dataset["test"], # Use original test dataset for generation
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        sbert_model_name=args.sbert_model_name,
        generation_kwargs=generation_params
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        #compute_metrics=compute_metrics, # Add perplexity calculation
        callbacks=[custom_eval_callback], # Add custom callback
        # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), # Optional: if specific collation needed
    )

    # --- 8. Resume Logic ---
    resume_from_checkpoint = None
    if args.resume:
        last_checkpoint = TrainingArguments.get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
            resume_from_checkpoint = last_checkpoint
        else:
            logger.warning(f"--resume flag was passed, but no checkpoint found in {args.output_dir}. Starting training from scratch.")

    # --- 9. Start Training ---
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # --- 10. Save Final Model & Metrics ---
    logger.info("Training finished. Saving final model and tokenizer...")
    # Option 1: Save the final adapter weights
    # model.save_pretrained(args.output_dir) # Saves only the LoRA adapter

    # Option 2: Merge weights and save the full model (requires more disk space/memory)
    # logger.info("Merging LoRA weights with base model...")
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(os.path.join(args.output_dir, "final_merged_model"))
    # tokenizer.save_pretrained(os.path.join(args.output_dir, "final_merged_model"))
    
    # Default: Save adapter and tokenizer separately (most common for LoRA)
    trainer.save_model(args.output_dir) # Saves adapter config + weights using trainer method
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Final model adapters and tokenizer saved to: {args.output_dir}")

    # Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # Saves trainer state (optimizer, scheduler, rng)

    # --- 11. Final Evaluation ---
    logger.info("Performing final evaluation on the test set...")
    eval_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    # Add prefix to distinguish final eval metrics
    final_eval_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
    trainer.log_metrics("eval", final_eval_metrics)
    trainer.save_metrics("eval", final_eval_metrics)

    # Run final custom semantic similarity evaluation
    logger.info("Performing final semantic similarity evaluation...")
    final_score = custom_eval_callback.compute_similarity(
        trainer.model, # Use the trained model from the trainer
        output_file=os.path.join(args.output_dir, "custom_eval", "evaluation_final.json")
    )
    logger.info(f"Final semantic similarity score: {final_score:.4f}")
    trainer.log({"final_semantic_similarity": final_score})


    logger.info("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model using LoRA")

    # Model & Data Arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model identifier")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save checkpoints and final model. If None, derived from model name.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of dataset to use for testing")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for tokenization")

    # Training Arguments
    parser.add_argument("--num_train_epochs", type=float, default=4.0, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type (e.g., 'linear', 'cosine')")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log training information every N steps")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every N steps (overrides save_strategy='epoch' if set)")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluate every N steps (overrides evaluation_strategy='epoch' if set)")
    parser.add_argument("--save_total_limit", type=int, default=10, help="Maximum number of checkpoints to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory")

    # LoRA Arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (rank)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--lora_target_modules", nargs='+', default=None, help="List of module names to apply LoRA to (e.g., 'q_proj' 'v_proj'). Default depends on model type.")

    # Quantization Arguments
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "8bit", "4bit"], help="Quantization type (none, 8bit, 4bit)")

    # Monitoring & Restart Arguments
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Integrations to report results to (e.g., 'wandb', 'tensorboard', 'none', or comma-separated)")
    parser.add_argument("--resume", action='store_true', help="Resume training from the last checkpoint in output_dir")

    # Evaluation Arguments
    parser.add_argument("--sbert_model_name", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model for semantic similarity evaluation")
    parser.add_argument("--eval_max_new_tokens", type=int, default=256, help="Max new tokens for generation during custom evaluation")
    parser.add_argument("--eval_do_sample", type=bool, default=True, help="Whether to use sampling during custom evaluation generation")
    parser.add_argument("--eval_temperature", type=float, default=0.7, help="Temperature for sampling during custom evaluation generation")

    # Preprocessing Arguments
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="Number of processes for dataset preprocessing (default uses dataset library's heuristic)")
    parser.add_argument("--overwrite_cache", action='store_true', help="Overwrite the cached preprocessed datasets")


    args = parser.parse_args()

    # Derive output_dir if not provided
    if args.output_dir is None:
        model_name_parts = args.model_name.split('/')
        model_short_name = model_name_parts[-1] if model_name_parts else "unknown_model"
        args.output_dir = f"./{model_short_name.lower()}-lora-output"
        logger.info(f"--output_dir not specified, derived as: {args.output_dir}")

    main(args)