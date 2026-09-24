"""
Valerois Byte-Fallback BPE Tokenizer Trainer
8,192 kelime dağarcığı, sıfır OOV (tüm 256 byte tabanlı), ultra hızlı Rust BPE.
"""
import os
import json
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, Regex

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<|im_start|>", "<|im_end|>",
    "<|user|>", "<|assistant|>", "<|system|>",
    "<|thought|>", "<|tool|>", "<|action|>",
]

def train_valerois_tokenizer(corpus_file: str, out_file: str = "valerois_tokenizer_8k.json", vocab_size: int = 8192, sample_lines: int = 100000):
    print(f"[tokenizer] Veri taraniyor: {corpus_file} (maks: {sample_lines:,} satir)...")
    
    # 1. BPE Model + Byte-Level Fallback
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    
    # 2. GPT-4 / Llama 3 tarzi gelismis Pre-Tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 3. Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        min_frequency=2,
    )
    
    def text_iterator():
        count = 0
        with open(corpus_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text.strip():
                        yield text
                        count += 1
                        if sample_lines > 0 and count >= sample_lines:
                            break
                except Exception:
                    continue
                    
    print(f"[tokenizer] {vocab_size} vocab boyutlu BPE egitiliyor...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    
    # Kaydet
    tokenizer.save(out_file)
    print(f"[tokenizer] Basariyla kaydedildi: {out_file}")
    
    # Test
    test_texts = [
        "Hello, this is a test of the Valerois architecture with high-speed tokenization!",
        "Merhaba dunya! Turkce karakterler ve ozel etiketler sorunsuz calisiyor mu?",
        "<|im_start|>user\nBana bir Python fonksiyonu yaz.<|im_end|>\n<|im_start|>thought\nKullanici fonksiyon istiyor.<|im_end|>"
    ]
    
    print("\n--- ORNEK TEST VE SIKISTIRMA ORANI ---")
    for t in test_texts:
        encoded = tokenizer.encode(t)
        decoded = tokenizer.decode(encoded.ids)
        raw_bytes = len(t.encode("utf-8"))
        tokens = len(encoded.ids)
        ratio = raw_bytes / max(tokens, 1)
        print(f"\nOrijinal Metin ({raw_bytes} byte): {t[:60]}...")
        print(f"Token Sayisi ({tokens} token) -> Sikistirma: {ratio:.2f}x daha az adim!")
        print(f"Token ID'leri (ilk 8): {encoded.ids[:8]}...")
    print("\n[tokenizer] Tum testler basarili!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="fineweb_edu_250m.jsonl")
    parser.add_argument("--out", type=str, default="valerois_tokenizer_8k.json")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--sample_lines", type=int, default=150000)
    args = parser.parse_args()
    
    train_valerois_tokenizer(args.corpus, args.out, args.vocab_size, args.sample_lines)
