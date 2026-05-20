# nanoGPT-rf-rope-ablation

**A controlled architectural ablation: Rotary Position Embedding (RoPE) vs. a learned positional embedding on a from-scratch decoder-only transformer trained on RF/telecom domain text.**

At matched training compute, identical seed, identical recipe, with `pos_encoding` as the *only* differing variable, **RoPE achieves 35.72 validation perplexity vs. 40.66 for the learned-positional-embedding baseline — a 12.1% improvement on this corpus, with 98,304 fewer parameters** (RoPE has no learned position table).

---

## TL;DR

- ~14M-parameter GPT trained from scratch on 9.45M tokens of arXiv `eess.SP` (signal-processing) abstracts.
- **Clean controlled ablation:** the `pos_encoding` config field is the *only* difference between the two training runs. Every other knob — seed, data order, optimizer, schedule, dropout, weight init — is byte-identical.
- **Result:** RoPE wins by **+12.1% validation perplexity**, with **−98,304 parameters** (the entire learned position table, removed).
- **Reproducible end-to-end** on a single free Kaggle T4 in ~30 minutes from scratch, or ~2 minutes when reusing cached checkpoints from a prior Commit.

## Motivation

RoPE (Su et al., 2021) is the positional encoding behind LLaMA, GPT-NeoX, Qwen, and most modern open-weight LLMs. Its empirical advantage over learned positional embeddings is well-documented at scale. **What's less clear is how cleanly that advantage transfers to small, from-scratch models on narrow domain text** — the regime hobbyists, applied researchers, and small-team engineers actually operate in.

This project tests the question concretely on **arXiv `eess.SP` (signal-processing) abstracts**: under a strictly matched-compute, single-variable ablation, does RoPE outperform a learned positional embedding? The answer here is *yes, by a measurable margin, with strictly fewer parameters* — and the result is constructed so that **"RoPE caused this"** is structurally defensible rather than statistically argued, because exactly one independent variable changes between runs.

A side virtue of RoPE is its natural interpretation in signal-processing terms: position becomes a phase ramp, the attention dot product is a correlator, the absolute carrier phase cancels in coherent correlation, and only the relative phase (= relative position) survives. The implementation runs on exactly that property — making the ~98k learned positional parameters of the baseline literally zero in the RoPE arm.

## Method

### Corpus
- **Source:** Cornell-University/arxiv Kaggle dataset (the full arXiv metadata snapshot — ~5.3 GB JSON-lines, all categories).
- **Filter:** papers whose `categories` field contains `eess.SP` as a token — split-and-test, not substring (substring matching would silently over-match future sibling categories like a hypothetical `eess.SPA`).
- **Cleaning:** collapse all whitespace runs to a single space (arXiv abstracts ship with hard line-wraps that would otherwise become learned position artifacts in the tokenizer); preserve math notation and case.
- **Output:** 39,818 cleaned abstracts, **48.7 MB / 48.74M characters** → `eess_sp_corpus.txt`.

### Tokenizer
- **Custom byte-level BPE** (GPT-2 style), HuggingFace `tokenizers`, vocab 8000.
- Byte-level base alphabet → **no `<unk>` token ever** — math symbols and Greek letters degrade to byte sequences.
- One special token: `<|endoftext|>` (document boundary in the packed stream).
- **Measured compression: 5.16 chars/token** on the corpus → 9.45M training tokens.
- **Domain payoff (verified):** `OFDM` and `MIMO` each encode to **a single token**; off-domain words like `photosynthesis`, `chlorophyll` shatter into 5–6. The tokenizer learned *this corpus's* statistics, not GPT-2's.

### Model architecture
A nanoGPT-style decoder-only transformer:

```
Token IDs (B, T)                                      e.g. (32, 256)
   │
   ▼
Embeddings:
   • token table    (8000 × 384)
   • position table  (256 × 384)        ← removed when pos_encoding="rope"
   │
   ▼
6 × Block:
   • LayerNorm
   • Causal self-attention (6 heads, head_dim 64)
       └─ if pos_encoding="rope": rotate Q, K with a fixed cos/sin table
   • residual
   • LayerNorm
   • MLP (Linear 384→1536, GELU, Linear 1536→384)
   • residual
   │
   ▼
Final LayerNorm
   │
   ▼
lm_head (tied with token embedding)  →  logits (B, T, 8000)
```

- Pre-LayerNorm + residuals; dropout 0.1 in both attention and MLP.
- Tied input/output embeddings; weight init `N(0, 0.02)` with GPT-2 scaled init (`std = 0.02/√(2·n_layer)`) on residual output projections.
- **The positional encoding is a single config switch:**
  - `pos_encoding="learned"` → `nn.Embedding(256, 384)` adds a learned vector per position → **98,304 learned parameters.**
  - `pos_encoding="rope"` → no positional table; a fixed cos/sin rotation, registered as a **buffer** (not a parameter), applied to Q and K inside attention → **0 learned positional parameters.**
- Total: **13,797,120** (learned PE) / **13,698,816** (RoPE). The delta is exactly 256 × 384 = 98,304 — the entire positional table.

### Training recipe (byte-identical for both runs)
- 5000 optimizer steps · batch 32 · context 256 → 8,192 tokens/step → 40.96M tokens seen (~4.3 passes over the 9.45M-token corpus).
- AdamW, peak LR 3e-4 with 200-step linear warmup → cosine decay to 3e-5; `betas=(0.9, 0.95)`, `weight_decay=0.1`.
- **Mixed precision: fp16 + `GradScaler`.** Tesla T4 is Turing (SM 7.5) — FP16 tensor cores, no bf16 acceleration (bf16 requires Ampere SM 8.0+).
- Gradient clipping at global norm 1.0.
- Deterministic 95% / 5% train/val split, seed 1337.
- **Checkpoint on best val loss** (not final step) — eval discipline against potentially-different overfitting schedules between the two arms.
- Logged live to Weights & Biases.
- Wall-clock: ~10 min per training run on a single Kaggle T4; the full Commit (corpus + tokenizer + data + both training runs + evaluation) is ~30 min end-to-end.

### Evaluation
- **Quantitative:** validation perplexity, averaged over 50 batches of held-out tokens per estimate. Compared at *each model's* best-val checkpoint — not at a fixed step. If two arms overfit at different rates, a fixed-step comparison measures overfitting dynamics rather than positional-encoding quality.
- **Qualitative:** 5 RF/telecom prompts, temperature 0.8, nucleus (top-p) sampling at 0.9, 100 generated tokens each.

## Results

### Quantitative

| Metric | Baseline (learned PE) | RoPE | Δ |
|---|---:|---:|---:|
| Parameters | 13,797,120 | 13,698,816 | **−98,304** |
| Best-val step | 4999 | 4999 | — |
| Val loss | 3.7051 | 3.5757 | −0.1295 |
| **Val perplexity** | **40.66** | **35.72** | **+12.1%** |

The parameter delta of exactly −98,304 = 256 × 384 = the full learned positional embedding table, removed. **RoPE wins with strictly fewer parameters** — the comparison is biased *against* RoPE on a pure-capacity basis, and it still wins.

### Qualitative

Both models produce fluent, domain-correct RF/telecom prose: real `eess.SP` vocabulary (OFDM, MIMO, beamforming, RIS, mmWave, channel estimation, hybrid precoding, NOMA) in correct arXiv-abstract register and structure. Both also exhibit the expected small-model failure modes — n-gram looping, semantic looseness on longer generations, occasional contradictions. The qualitative gap between the two is much smaller than the perplexity gap suggests: at this scale, a 4-PPL gap doesn't manifest as a dramatic difference in a 100-token continuation. Five paired prompt outputs are produced by Cell 26 of the notebook.

### Training curves

Logged live to Weights & Biases, public project: **[wandb.ai/rjrajpoot01-namal-university/rf-rope-ablation](https://wandb.ai/rjrajpoot01-namal-university/rf-rope-ablation)**. Both runs (`baseline-learned-pe` and `rope`) show monotonically-decreasing val loss; RoPE's val curve sits below baseline's for essentially the entire run.

## Limitations (read these before citing the +12.1%)

These are real and they matter for how strongly the result should be read.

1. **Single seed per arm.** The +12.1% gap is a single point estimate; population mean is unknown. A complete v2 would run 3–5 seeds per arm and report mean ± std.
2. **Small corpus, small model.** 9.45M training tokens against ~13.8M parameters is roughly 50× below Chinchilla compute-optimal. Training is multi-epoch (~4.3 passes), so both arms overfit modestly — fortunately *symmetrically*, which preserves the ablation conclusion but caps absolute quality.
3. **Domain-specific tokenizer.** Byte-level BPE on `eess.SP` excels on RF/telecom text and degrades sharply off-domain (biology words shatter into 5–6 tokens). Perplexity numbers here are *not* comparable across corpora — a deliberate scope choice, not a hidden flaw.
4. **256-token context only.** RoPE's much-cited length-extrapolation advantages are not exercised by this setup.
5. **No statistical-significance test.** With one run per arm, the appropriate framing is "directionally consistent with the published literature for RoPE on small-scale tasks" — *not* "statistically significant 12.1% improvement."

## Reproducibility

Two paths.

**Fast (reuse the trained checkpoints, ~2 min):**

1. Open the notebook on Kaggle (or upload `notebooks/rf_rope_ablation.ipynb`).
2. Attach the notebook's most recent successful Version output as an Input:
   *Add Input → Notebook Output Files → notebook → latest Version*.
3. Attach a `WANDB_API_KEY` Kaggle Secret (key from [wandb.ai/authorize](https://wandb.ai/authorize)).
4. *Save Version → Save & Run All (Commit)*. The bootstrap cell copies cached `.bin` and `.pt` artifacts into `/kaggle/working/`; every training cell sees its checkpoint and prints `[cached] … skipped`. Total runtime ~2 minutes.

**Full from-scratch run (~30 min on a T4):**

1. Same setup, but *do not* attach the prior Notebook Output as an Input.
2. *Save Version → Save & Run All (Commit)*.
3. The notebook rebuilds the corpus, trains the BPE tokenizer, encodes the token bins, runs a 100-step smoke test, trains the 5000-step baseline, trains the 5000-step RoPE, and produces the head-to-head comparison.

The notebook is a single file with no imports beyond the Kaggle base image, `wandb`, and `einops`. The arXiv dataset is fetched via `kagglehub` so the path is never hardcoded.

Trained checkpoints will additionally be published to HuggingFace Hub:

- `MuhammadHanzalaIqbal/rf-gpt-baseline-14m` (learned PE)
- `MuhammadHanzalaIqbal/rf-gpt-rope-14m` (RoPE)

## References

- Su et al., 2021. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).
- Karpathy, A. *nanoGPT* — code style and structural reference.
- Cornell University, *arXiv metadata snapshot* on Kaggle: [Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).

## License

MIT — see [`LICENSE`](LICENSE).

---

*Author: Muhammad Hanzala Iqbal · Pull requests and issues welcome.*

## Related work by the same author

Wireless / signal-processing portfolio:

- [`mimo-spatial-filtering`](https://github.com/MuhammadHanzalaIqbal/mimo-spatial-filtering) — SVD precoding, beam-domain analysis, CSI-error impact.
- [`mimo-channel-estimation`](https://github.com/MuhammadHanzalaIqbal/mimo-channel-estimation) — SRS-based LS channel estimation under TDD reciprocity.
- [`mimo-channel-denoising`](https://github.com/MuhammadHanzalaIqbal/mimo-channel-denoising) — adaptive delay-window + soft singular-value shrinkage, strictly improves on the hard-window baseline.
