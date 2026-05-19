# Results - RoPE vs Learned PE on RF/Telecom Abstracts

| Metric | Baseline (learned PE) | RoPE | Delta |
|---|---:|---:|---:|
| Parameters | 13,797,120 | 13,698,816 | -98,304 |
| Best-val step | 4999 | 4999 | - |
| Val loss | 3.7051 | 3.5757 | -0.1295 |
| **Val perplexity** | **40.66** | **35.72** | **+12.1%** |

**Setup.** ~13.8M-param decoder-only transformer (6 layers, 6 heads, n_embd=384, block_size=256). Byte-level BPE, vocab 8000, trained on arXiv `eess.SP` abstracts (9.45M training tokens). Same seed, same recipe (5000 steps, batch 32, AdamW lr=3e-4 with 200-step warmup + cosine decay, fp16+GradScaler, dropout 0.1, grad-norm clip 1.0). The **only** difference between runs: `pos_encoding`.

Comparison reported at each model's **best-val** checkpoint (early-stop discipline, not fixed-step).
