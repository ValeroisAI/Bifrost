"""
================================================================================
 🐍 PYTHON 4K SPECIALIZED CODE TOKENIZER TRAINER
================================================================================
Trains a 4,096 Byte-Pair Encoding (BPE) Tokenizer strictly optimized for Python:
  - Preserves exact 4-space ("    "), 8-space ("        "), and newline indents.
  - Includes atomic tokens for Python keywords: def, return, class, import, etc.
  - Generates valerois_code_tokenizer_4k.json
================================================================================
"""

import os
import sys
import json
import glob
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def collect_python_training_corpus(out_txt="python_code_corpus.txt"):
    print("[*] Collecting Python source code training corpus...")
    files_to_scan = []
    
    # 1. Workspace Python files
    files_to_scan.extend(glob.glob("*.py"))
    
    # 2. Existing JSONL code datasets
    for jsonl in ["titan_algorithmic_code_master.jsonl", "titan_master_docstring_sft.jsonl", "expert_1_coder.jsonl"]:
        if os.path.exists(jsonl):
            files_to_scan.append(jsonl)
            
    # 3. Python standard library / venv lib files
    venv_lib = os.path.join(".venv", "Lib")
    if os.path.exists(venv_lib):
        for root, _, files in os.walk(venv_lib):
            for f in files:
                if f.endswith(".py"):
                    files_to_scan.append(os.path.join(root, f))
                    if len(files_to_scan) >= 500:
                        break
            if len(files_to_scan) >= 500:
                break

    print(f"[*] Found {len(files_to_scan)} Python source files to extract...")

    total_lines = 0
    with open(out_txt, "w", encoding="utf-8", errors="ignore") as out_f:
        for fpath in files_to_scan:
            try:
                if fpath.endswith(".jsonl"):
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                p = data.get("prompt", "")
                                c = data.get("completion", "")
                                out_f.write(p + "\n" + c + "\n")
                                total_lines += 2
                else:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        out_f.write(content + "\n")
                        total_lines += content.count("\n")
            except Exception:
                pass

    print(f"  [+] Compiled {total_lines:,} lines of pure Python code into {out_txt} ({os.path.getsize(out_txt)/(1024*1024):.2f} MB)")
    return out_txt

def train_code_tokenizer(corpus_file: str, vocab_size: int = 4096, out_json: str = "valerois_code_tokenizer_4k.json"):
    print(f"\n[*] Training {vocab_size} Vocab Python Code BPE Tokenizer...")

    # BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenizer that respects whitespace & indentation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [
        "<pad>", "<eos>", "<unk>", "<bos>",
        "    ", "        ", "            ", # 4, 8, 12 spaces indents
        "\n    ", "\n        ", "\n            ", # Newline + indents
        "def ", "return ", "class ", "import ", "from ", "self.", "lambda ",
        "assert ", "yield ", "async ", "await ", "try:", "except:", "finally:",
        "if ", "elif ", "else:", "for ", "while ", "in ", "not in ", "is ",
        "True", "False", "None", " -> ", " == ", " != ", " <= ", " >= ",
        '"""', "'''", "f\"", "f'", "r\"", "r'", "b\"", "b'"
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train([corpus_file], trainer)
    tokenizer.save(out_json)

    print(f"  [🏆 SUCCESS] Trained and saved Python Code Tokenizer: {out_json}")
    print(f"  [*] Total Vocab Size: {tokenizer.get_vocab_size():,}")

    # Test tokenization on sample python code
    sample_code = 'def fibonacci(n: int) -> list:\n    """Return list of fib numbers."""\n    if n <= 0:\n        return []\n    fibs = [0, 1]\n    while len(fibs) < n:\n        fibs.append(fibs[-1] + fibs[-2])\n    return fibs\n'
    encoded = tokenizer.encode(sample_code)
    print(f"\n[Test Tokenization]:\n  Original Text Length : {len(sample_code)} characters")
    print(f"  Encoded Token Count  : {len(encoded.ids)} tokens (vs ~110 in old generic tokenizer!)")
    print(f"  First 15 Token IDs   : {encoded.ids[:15]}")
    print(f"  First 15 Decoded     : {[tokenizer.decode([t]) for t in encoded.ids[:15]]}")

def main():
    print("=" * 80)
    print(" 🚀 4K PYTHON SPECIALIZED CODE TOKENIZER BUILDER")
    print("=" * 80)
    corpus = collect_python_training_corpus()
    train_code_tokenizer(corpus, vocab_size=4096, out_json="valerois_code_tokenizer_4k.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
