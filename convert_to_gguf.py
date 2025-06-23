#!/usr/bin/env python3
"""
hf2gguf.py – Swiss-army knife for exporting Hugging Face models (optionally with LoRA) to GGUF.

Prerequisites
-------------
git clone https://github.com/ggerganov/llama.cpp
pip install "torch>=2.2" transformers peft safetensors accelerate huggingface-hub

Usage examples
--------------
# 1) Plain model, Q4_K_M
python hf2gguf.py --base mistralai/Mistral-7B-Instruct-v0.3 --quant q4_K_M

# 2) Merge LoRA, then quantise to Q6_K
python hf2gguf.py --base meta-llama/Llama-3-8b \
                  --lora TheBloke/Samantha-8B-LoRA \
                  --merge-lora \
                  --quant q6_K \
                  --outfile llama3-samantha-q6k.gguf

# 3) Keep LoRA separate (two GGUF files)
python hf2gguf.py --base meta-llama/Llama-3-8b \
                  --lora ./my_adapter \
                  --strategy separate \
                  --quant q8_0
"""
import argparse, os, subprocess, tempfile, shutil, sys
from pathlib import Path

LLAMA_CPP = Path(os.getenv("LLAMA_CPP", "llama.cpp")).resolve()
CONVERT_HF = LLAMA_CPP / "convert-hf-to-gguf.py"
CONVERT_LORA = LLAMA_CPP / "convert-lora-to-gguf.py"

def run(cmd, **kw):
    print(">", *cmd); subprocess.check_call(cmd, **kw)

def convert_base(model, outpath, outtype):
    run([sys.executable, CONVERT_HF, str(model),
         "--outfile", str(outpath), "--outtype", outtype])

def convert_lora(adapter, outpath, outtype):
    run([sys.executable, CONVERT_LORA, str(adapter),
         "--outfile", str(outpath), "--outtype", outtype])

def merge_lora(base, adapter, merged_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch, safetensors.torch as st

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base)
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF repo or local path")
    ap.add_argument("--lora", help="HF repo or local path of LoRA adapter")
    ap.add_argument("--merge-lora", action="store_true",
                    help="merge adapter into base before conversion")
    ap.add_argument("--strategy", choices=["merge", "separate"], default="merge",
                    help="‘merge’ = single GGUF, ‘separate’ = base GGUF + adapter GGUF")
    ap.add_argument("--quant", default="f16", help="GGUF outtype (q8_0, q6_K, ...)")
    ap.add_argument("--outfile", help="desired output file name")
    args = ap.parse_args()

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)

    if not args.lora:
        out = Path(args.outfile or f"{Path(args.base).name}.{args.quant}.gguf")
        convert_base(args.base, out, args.quant)

    else:
       if args.strategy == "separate" and not args.merge_lora:
           base_out = Path(args.outfile or f"{Path(args.base).name}.{args.quant}.gguf")
           adapter_out = base_out.with_suffix(f"-adapter.{args.quant}.gguf")
           convert_base(args.base, base_out, args.quant)
           convert_lora(args.lora, adapter_out, args.quant)

       else:  # merge strategy
           merged = tmp_path / "merged"
           merge_lora(args.base, args.lora, merged)
           out = Path(args.outfile or f"{Path(args.base).stem}-merged.{args.quant}.gguf")
           convert_base(merged, out, args.quant)

    tmp.cleanup()

if __name__ == "__main__":
    main()
