# Valerois Architecture Development Guide
Active project: `Heimdall/` (see `Heimdall/README.md`). Old work (Bifrost CSL, Kuzgun, Valkir, ...) lives in `ESKI/`; `ESKI/AGENT_CONTEXT_BRIEFING.md` has the history.

## Core Rules:
1. Check `ESKI/AGENT_CONTEXT_BRIEFING.md` only for historical background.
2. Architecture Focus: Bifrost CSL + Sliding Window Cache + Attention/Retrieval Layer.
3. Keep memory footprint O(1) or lightweight O(N), no full-sequence quadratic explosion without chunking/sliding window.
4. Tokenizer: `04_TOKENIZERLAR/valerois_tokenizer_8k.json` (vocab: 8192).
5. Dataset: `stream_coder_100k.bin` is available in root (copy also in `ESKI/02_TURNUVALAR_VE_TESTLER/`).
