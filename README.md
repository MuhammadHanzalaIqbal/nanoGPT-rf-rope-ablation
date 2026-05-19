# nanoGPT-rf-rope-ablation

**RoPE vs learned positional embedding on a ~14M-parameter decoder-only transformer trained from scratch on arXiv RF/telecom abstracts.**

At matched 5000-step compute, identical seed, identical recipe, with the *only* difference being the positional encoding, **RoPE achieves 35.72 validation perplexity vs 40.66 for the learned-positional-embedding baseline — a 12.1% lower perplexity** on this corpus, *with 98,304 fewer parameters* (RoPE has no learned position table).

---

## TL;DR

- ~14M-parameter GPT (6 layers, 6 heads, `n_embd=384`, `block_size=256`, vocab 8000 byte-level BPE).
- Trained from scratch on **9.45M tokens** of arXiv `eess.SP` (signal-processing) abstracts.
- **Clean ablation:** a single `pos_encoding` config field is flipped between the two runs; every other knob — seed, data order, optimizer, schedule, dropout, init — is byte-identical.
- **RoPE wins by +12.1% val perplexity** at matched compute, with **zero** learned positional parameters.
- Reproducible end-to-end on a single free Kaggle T4 in **~30 minutes** for a fresh full run, **~2 minutes** when reusing the cached checkpoints from a prior Commit.

## Why this project

I'm a wireless/RF engineer pivoting into LLM engineering. This is the Week-1 deliverable of a self-imposed 6-week sprint that combines my RF/telecom domain knowledge with modern transformer engineering and aims at a publishable, defensible, reproducible artifact.

I chose RoPE-vs-learned-PE because:

1. It's the single most consequential modern-LLM architectural choice that wasn't covered in the foundational tutorials I came from (Karpathy's *Let's build GPT* / *reproduce GPT-2*).
2. The ablation is structurally crisp: RoPE has **zero** learned positional parameters — it injects relative position via a fixed rotation of Q and K inside attention. The "what changed" is unambiguous and the param-count delta is exactly the removed positional embedding table.
3. It maps cleanly onto my home domain. RoPE is *position-as-phase*: each token's vector is rotated by an angle proportional to its position; the attention dot product is a correlation; an absolute carrier phase cancels in coherent correlation and only the relative phase survives. RoPE rides on exactly that property to inject *relative*-position awareness with no learned parameters.

## Method

### Corpus
- **Source:** Cornell-University/arxiv Kaggle dataset (the full arXiv metadata snapshot — ~5.3 GB JSON-lines, all categories).
- **Filter:** papers whose categories field contains `eess.SP` (Signal Processing) as a token — `categories.split()`, not substring matching (substring would silently over-match future sibling categories like `eess.SPA`).
- **Cleaning:** collapse all whitespace runs to single spaces (arXiv abstracts ship with hard line-wraps that would otherwise become learned-position artifacts in the tokenizer); preserve math notation and case (these encode domain signal).
- **Output:** 39,818 cleaned abstracts, **48.7 MB / 48.74M characters** in `eess_sp_corpus.txt`.

### Tokenizer
- **Custom byte-level BPE** (GPT-2 style), trained with HuggingFace `tokenizers`, vocab 8000.
- Byte-level base alphabet means **no `<unk>` token ever** — math symbols and Greek letters degrade gracefully to byte sequences.
- One special token: `<|endoftext|>` (used as the document boundary in the packed token stream).
- **Measured compression: 5.16 chars/token** on the corpus → 9.45M training tokens.
- Domain payoff (verified empirically): `OFDM`, `MIMO` each encode to a **single token**; off-domain words like `photosynthesis`, `chlorophyll` shatter into 5–6 tokens. The tokenizer learned *this corpus's* statistics, not GPT-2's.

### Model architecture
A nanoGPT-style decoder-only transformer, built from scratch in PyTorch:

```
Token IDs (B, T)                                      e.g. (32, 256)
   │
   ▼
Embeddings:
   • token table   (8000 × 384)
   • position table (256 × 384)        ← removed when pos_encoding="rope"
   │
   ▼
6 × Block:
   • LayerNorm
   • Causal self-attention (6 heads, head_dim 64)
       └─ if pos_encoding="rope":  rotate Q, K with fixed cos/sin table
   • residual
   • LayerNorm
   • MLP (Linear 384→1536, GELU, Linear 1536→384)
   • residual
   │
   ▼
Final LayerNorm
   │
   ▼
lm_head (tied with token embedding) → logits (B, T, 8000)
```

- Pre-LayerNorm + residual connections; dropout 0.1 inside both attention and MLP.
- Tied input/output embeddings; GPT-2 weight init: `N(0, 0.02)`, with `std = 0.02 / √(2·n_layer)` on the residual output projections.
- **Position encoding is the single configurable switch:**
  - `pos_encoding="learned"` → `nn.Embedding(256, 384)` adds a learned vector at each position (**98,304 learned parameters**).
  - `pos_encoding="rope"` → no position table; a fixed cos/sin rotation, registered as a **buffer** (not a parameter), applied to Q and K inside attention (**0 learned positional parameters**).
- Total parameters: **13,797,120** (learned-PE) / **13,698,816** (RoPE). The delta is exactly the 256 × 384 = 98,304 positional table.

### Training recipe (byte-identical for both runs)
- 5000 optimizer steps · batch 32 · context length 256 → 8,192 tokens/step → 40.96M tokens seen ≈ ~4.3 passes over the 9.45M-token corpus.
- AdamW, peak LR 3e-4 with 200-step linear warmup → cosine decay to 3e-5; `betas=(0.9, 0.95)`, `weight_decay=0.1`.
- **Mixed precision: fp16 + `GradScaler`.** Tesla T4 is Turing (SM 7.5) — it has FP16 tensor cores but no bf16 acceleration (bf16 needs Ampere SM 8.0+). The original brief targeted bf16; this is a hardware-correctness fix.
- Gradient clipping at global norm 1.0.
- Train/val split: deterministic 95% / 5% of the packed stream (seeded 1337).
- **Best-val checkpoint** saved (not final-step — eval-discipline against potentially-different overfitting schedules between the two arms).
- Logged live to Weights & Biases.
- Wall-clock: each training run ~10 min on a single Kaggle T4; a full Commit (corpus rebuild + BPE training + data bins + smoke test + both 5000-step runs + evaluation) takes ~30 min end-to-end.

### Evaluation
- **Quantitative:** validation perplexity, averaged over 50 batches of held-out tokens per estimate. Compared at *each model's* **best-val checkpoint** — not at a fixed step. (If the two arms overfit at different rates, a fixed-step comparison would measure overfitting dynamics rather than positional-encoding quality.)
- **Qualitative:** 5 RF/telecom prompts, temperature 0.8, nucleus (top-p) sampling at 0.9, 100 generated tokens each.

## Results

### Quantitative

| Metric | Baseline (learned PE) | RoPE | Δ |
|---|---:|---:|---:|
| Parameters | 13,797,120 | 13,698,816 | **−98,304** |
| Best-val step | 4999 | 4999 | — |
| Val loss | 3.7051 | 3.5757 | −0.1295 |
| **Val perplexity** | **40.66** | **35.72** | **+12.1%** |

The parameter delta of exactly −98,304 = 256 × 384 = the entire learned positional embedding table, removed. So RoPE wins **with strictly fewer parameters** — the comparison is biased *against* RoPE on a pure-capacity basis, and it still wins.

### Qualitative

Both models produce fluent, domain-correct RF/telecom prose: real eess.SP vocabulary (OFDM, MIMO, beamforming, RIS, mmWave, channel estimation, hybrid precoding, NOMA) used in mostly-right contexts, and recognizable arXiv-abstract register and structure. Both also suffer the expected small-model failure modes — n-gram looping (`"the transmit power, and the transmit power"`), semantic looseness on long generations, occasional contradictions, and mid-stream "new abstract" restarts (the `<|endoftext|>` boundary in the packed training stream showing through). The qualitative gap between the two is much smaller than the perplexity gap suggests — a model at PPL 36 vs 40 will not read dramatically differently in a 100-token continuation. Five paired prompt outputs are produced by Cell 26 of the notebook; they are best evaluated blind for win/loss/tie.

### Training curves

Logged live to Weights & Biases, public project: **[wandb.ai/rjrajpoot01-namal-university/rf-rope-ablation](https://wandb.ai/rjrajpoot01-namal-university/rf-rope-ablation)**. Both runs (`baseline-learned-pe` and `rope`) show monotonically-decreasing val loss; RoPE's val curve sits below baseline's for essentially the entire run.

## Limitations and honest caveats

These are real and they matter for how strongly the result should be read.

1. **Single seed per arm.** The +12% gap is a single point estimate. I do not claim it's the population mean. A reviewer would reasonably ask for 3–5 seeds per arm to bound the variance — that is the obvious next experiment.
2. **Small corpus, small model.** ~9.45M training tokens against a ~13.8M-param model is roughly **50× below Chinchilla-compute-optimal**. The model trains for ~4.3 epochs, so both arms overfit modestly (the train–val gap widens from ~0 to ~0.2 nats over 5000 steps). Crucially this happens *symmetrically*, so the ablation conclusion is preserved — but absolute perplexity is mediocre and the model is, in absolute terms, not very good.
3. **Below the brief's "~25M" target.** I deliberately kept the clean architectural numbers (6 layers, dim 384, vocab 8000) and reported the true ~13.8M parameter count rather than tuning a knob to hit a round headline. The ablation result is unaffected by this scale choice at this regime.
4. **Domain-specific tokenizer.** Byte-level BPE on `eess.SP` is excellent on RF/telecom text and *terrible off-domain*. Perplexity numbers from this model are not comparable to those of any model trained with a different tokenizer or on different text. This is a deliberate, defensible scope choice — not a hidden flaw.
5. **256-token context window only.** RoPE's much-cited length-extrapolation advantages are not exercised at all by this setup.
6. **No statistical-significance test.** With one run per arm, the appropriate framing is "directionally consistent with the published literature for RoPE on small-scale tasks" — *not* "RoPE outperforms learned PE by a statistically-significant 12.1% on this corpus."

## Reproducibility

Two paths.

**Fast (reuse the trained checkpoints, ~2 min):**

1. Open the notebook on Kaggle (or upload `notebooks/rf_rope_ablation.ipynb`).
2. Attach this notebook's most recent successful Version output as an Input:
   *Add Input → Notebook Output Files → notebook38072f32cf → latest Version*.
3. Attach the `WANDB_API_KEY` Kaggle Secret (get the key at [wandb.ai/authorize](https://wandb.ai/authorize)).
4. *Save Version → Save & Run All (Commit)*. The bootstrap cell copies cached `.bin` and `.pt` artifacts into `/kaggle/working/`; every training cell sees its checkpoint and prints `[cached] … skipped`. Total runtime ~2 minutes.

**Full from-scratch run (~30 min on a T4):**

1. Same setup, but *do not* attach the prior Notebook Output as an Input.
2. *Save Version → Save & Run All (Commit)*.
3. The notebook rebuilds the corpus, trains the BPE tokenizer, encodes the token bins, runs a 100-step smoke test, trains the 5000-step baseline, trains the 5000-step RoPE, and produces the head-to-head comparison.

The notebook itself is a single file with no external imports beyond the Kaggle base image, `wandb`, and `einops`. The `arXiv` dataset is fetched via `kagglehub` so the path is never hardcoded.

Trained models will additionally be released on HuggingFace Hub (uploading shortly):
- `rjrajpoot01/rf-gpt-baseline-14m` (learned positional embedding)
- `rjrajpoot01/rf-gpt-rope-14m` (RoPE)

## Acknowledgments & references

- **Su et al., 2021.** *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).
- **Andrej Karpathy**, *nanoGPT* and the *Let's build GPT* / *Let's reproduce GPT-2* videos — code style and pedagogy.
- The arXiv metadata snapshot: Cornell University, on Kaggle as *[arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)*.

## License

MIT — see [`LICENSE`](LICENSE).

---

*Author: Muhammad Hanzala Iqbal — Week-1 deliverable of a 6-week pivot from wireless researcher into LLM engineering. Feedback and pull requests welcome.*
