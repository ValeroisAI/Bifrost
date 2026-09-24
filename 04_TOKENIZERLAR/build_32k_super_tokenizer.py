"""
================================================================================
 ⚡ 32,768 VOCABULARY SUPER-TOKENIZER BUILDER FOR 14B TITAN
================================================================================
Builds a high-capacity 32K BPE tokenizer covering:
  - Python / Code AST constructs, indentation, keywords, operators
  - Turkish & English syllables and words
  - GSM8K / Math / Science LaTeX & Unicode symbols
================================================================================
"""

import os
import sys
import glob
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def build_32k_tokenizer():
    print("=" * 80)
    print(" ⚡ TRAINING 32,768 VOCABULARY SUPER-TOKENIZER FOR 14B TITAN")
    print("=" * 80)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = [
        "<pad>", "<eos>", "<unk>", "<bos>",
        "<|im_start|>", "<|im_end|>",
        "<|thought|>", "<|thought_end|>",
        "<|code|>", "<|code_end|>",
        "<|python|>", "<|json|>"
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=32768,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )

    # Gather all text sources
    files = glob.glob("*.jsonl") + glob.glob("*.md")
    print(f"[*] Training on {len(files)} local source files...")

    tokenizer.train(files=files, trainer=trainer)
    tokenizer.save("valerois_tokenizer_32k.json")
    tokenizer.save("ForRocm/valerois_tokenizer_32k.json")

    print(f" ⭐ 32K Super-Tokenizer Created: Vocab Size = {tokenizer.get_vocab_size()}")
    print("=" * 80)

if __name__ == "__main__":
    build_32k_tokenizer()
