# RoPE vs Learned Positional Embeddings: A Strict Single-Variable Ablation on RF/Telecom Text

**TL;DR.** Two 13.8M-parameter decoder-only transformers, trained from scratch on ~9.5M tokens of arXiv `eess.SP` abstracts. The two runs are byte-identical in every architectural choice, every training hyperparameter, and every random seed — *except* for the positional encoding (learned vs. RoPE). At matched 5000-step compute, evaluated at each model's best-val checkpoint, **RoPE achieves val perplexity 35.72 vs. 40.66 for the learned baseline — a 12.1 % improvement, with 98,304 fewer parameters** (RoPE has no learned position table at all).

This post is the engineering log of why I ran it, how I made the comparison defensible, what I had to debug, and what I wouldn't claim.

## Why this experiment

Rotary Position Embedding (RoPE; Su et al., 2021) is the positional encoding behind essentially every modern open-weight LLM — LLaMA, Mistral, GPT-NeoX, Qwen, Falcon. Its empirical advantage over learned positional embeddings is treated as common knowledge at scale.

But the published evidence sits squarely in the billion-parameter / trillion-token regime. **How cleanly does RoPE's advantage transfer to small, from-scratch models on narrow domain text?** That's the regime small teams, applied researchers, and hobbyists actually operate in, and the question is not obvious from the existing literature.

This project is a controlled answer for one specific domain: arXiv signal-processing abstracts. The answer turned out to be *yes, by a measurable, single-variable-attributable margin* — but the interesting part isn't the number. It's how the comparison was constructed so the number is *defensible* rather than merely *observed*.

## What's new vs. how most "RoPE vs learned PE" comparisons get done

Most reports comparing positional encodings I've encountered change multiple things at once: different model widths, different schedules, different data, sometimes different codebases. That cannot tell you *which variable caused the gap*. A 12 % improvement could be the positional encoding, or the schedule, or a different init, or — most likely — some combination.

This project commits to a discipline that's almost annoying to enforce but is the entire point:

> **Exactly one independent variable changes between the two runs: the `pos_encoding` config field. Everything else — seed, data order, optimizer, schedule, dropout, weight init, training compute, evaluation protocol — is byte-identical.**

Operationally, that means a single shared model class with `pos_encoding="learned"` or `"rope"` as the only differentiator. The "clean swap." If I'd written two separate model classes, a half-dozen tiny implementation drifts (different init scale, off-by-one in mask construction, different dropout placement) could each explain the perplexity gap and "RoPE caused this" becomes unfalsifiable. One class, one config field, automatic attribution.

## The recipe (the boring necessities)

- **Model:** ~14M-parameter nanoGPT-style decoder-only transformer — 6 transformer blocks, 6 attention heads, `n_embd = 384`, `head_dim = 64`, `block_size = 256`.
- **Tokenizer:** byte-level BPE trained on the RF corpus, vocab 8000. Domain words (`OFDM`, `MIMO`) encode as a single token; off-domain words shatter into 5–6 byte pieces. Measured compression: **5.16 chars/token**.
- **Data:** 9.45 M training tokens from 39,818 cleaned arXiv `eess.SP` abstracts. Packed into one stream with `<|endoftext|>` separators.
- **Optimizer:** AdamW, peak LR `3e-4` with 200-step linear warmup → cosine decay to `3e-5`, β = (0.9, 0.95), weight decay 0.1.
- **Precision:** **fp16 + GradScaler**. (Not bf16. The Tesla T4 is Turing — SM 7.5 — with FP16 tensor cores but no bf16 acceleration. More on this below.)
- **Gradient clipping** at global norm 1.0.
- **Compute:** 5000 steps × batch 32 × ctx 256 = 40.96 M tokens seen, ≈ 4.3 passes over the corpus.
- **Init:** GPT-2-style — `N(0, 0.02)` everywhere, with `std = 0.02/√(2 · n_layer)` on the residual output projections.
- **Eval:** 5 % held-out split; best-val checkpointed (not final-step — more on this too).
- **Logging:** Weights & Biases throughout.
- **Wall clock:** ~10 min per training run on a free Kaggle T4. The whole pipeline — corpus rebuild + BPE training + token-bin generation + smoke test + both 5000-step training runs + comparison — is ~30 min end-to-end. Reproducibility was a primary design constraint, not an afterthought.

The recipe is deliberately vanilla. The point isn't novel training tricks; it's a clean comparison.

## What RoPE actually is, in signal-processing language

If you have a DSP background, RoPE will feel familiar — it's an old trick in a new venue.

A token's *position* in the sequence becomes a *phase ramp*. Position `m` rotates the token's query and key vectors by angle `m · θ_k`, where `θ_k = 10000^(-2k/d)` is a geometrically-spaced family of frequencies — a frequency comb across feature dimensions. When attention then computes a dot product between query at position `m` and key at position `n`, the absolute rotations `m · θ_k` and `n · θ_k` cancel and what's left depends only on the *relative* position `(n − m) · θ_k`.

This is *exactly* the coherent-correlation property RF engineers rely on every day. When you correlate two phasor-modulated signals, a common carrier-phase offset on both sides cancels in the correlator output and only the relative phase (a function of the relative delay) appears. RoPE rides on that. **The attention dot product is the correlator. Absolute position is the common phase. Relative position is the phase difference that survives the correlation.** And because the rotation is a deterministic mathematical operation — not a learned table — there are *zero* learned positional parameters.

The contrast with the learned baseline is sharp:

- **Learned PE:** a literal `(block_size × n_embd) = (256 × 384) = 98,304`-entry table, one 384-vector per absolute position slot, trained by gradient descent. The model has to *memorize* what each slot "means."
- **RoPE:** zero parameters. Position is computed fresh on every forward pass as a fixed cos/sin rotation, never stored. (Within training-length range it's deterministic geometry; *beyond* training lengths is a separate, well-known story about extrapolation that this setup doesn't probe.)

The parameter delta between the two runs is exactly 98,304 — the entire learned positional table, *gone* in the RoPE arm. So RoPE wins with *fewer* parameters, which means the comparison is structurally biased *against* it on a pure-capacity basis. It still wins.

## The result

| Metric | Baseline (learned PE) | RoPE | Δ |
|---|---:|---:|---:|
| Parameters | 13,797,120 | 13,698,816 | **−98,304** |
| Best-val step | 4999 | 4999 | — |
| Val loss | 3.7051 | 3.5757 | −0.1295 |
| **Val perplexity** | **40.66** | **35.72** | **+12.1 %** |

RoPE is ~12 % lower perplexity at matched compute. Both arms reached their best val at the final step (val never turned up — overfitting is mild on this corpus at this scale). Qualitative generations from both models are recognisable RF/telecom prose with the expected small-model failure modes (n-gram looping, semantic looseness on long generations). The qualitative gap between them is much smaller than the perplexity gap suggests — at 36 vs. 40 PPL you don't see a dramatic narrative difference in a 100-token continuation.

## Two methodology points that mattered

### 1. Best-val checkpointing, not fixed-step comparison

Suppose you compared the two arms at a fixed step — say, both at step 5000. Sounds fair.

It isn't, in general. If the two architectures *overfit at different rates*, a fixed-step comparison measures *when you happened to look*, not which architecture is fundamentally better. Imagine RoPE reaches its best val at step 3000 (PPL 38) and then degrades by step 5000 to PPL 45 due to overfitting. Baseline's best is at step 5000 at PPL 40. Compare at step 5000 → "RoPE 45 vs baseline 40 — RoPE loses." Compare at each model's own best → "RoPE 38 vs baseline 40 — RoPE wins." *Opposite conclusions from the identical runs.*

The mechanical fix: checkpoint on best val per arm; report each model at its own best-val step. This is the early-stopping discipline applied at the comparison level.

In this run, both arms happened to still be improving at step 5000, so the issue didn't bite. But the protocol has to be there regardless — you cannot know ex ante which arm will overfit faster, and it's structurally plausible that learned PE's extra 98 k parameters would have overfit a small corpus *faster* than parameter-free RoPE.

### 2. Hardware-correct mixed precision (fp16 on T4, not bf16)

The original project plan called for bf16 mixed precision on T4 via `autocast`. That's hardware-wrong. The Tesla T4 is Turing architecture (compute capability 7.5); it has FP16 tensor cores but *no* native bf16 acceleration — bf16 tensor cores arrived with Ampere (SM 8.0+). On T4, bf16 autocast runs but no tensor-core path engages for bf16, the speedup vanishes, and you've made the precision choice for the wrong reason.

The correct choice on Turing is **fp16 autocast + GradScaler**. fp16's narrow 5-bit exponent makes tiny gradients prone to underflow; GradScaler multiplies the loss by a large factor before backprop so gradients land in fp16's representable window, then unscales before the optimizer step. Mathematically identical to no-scaling fp32 backprop, just numerically safe in fp16. bf16's 8-bit exponent (matching fp32) doesn't need this — but again, it's not accelerated on T4, so that tradeoff doesn't apply.

This is the kind of detail invisible from published literature (which mostly runs on Ampere / Hopper) but matters when you're actually trying to reproduce on free-tier hardware.

## Three debugging moments that didn't make the headline

Real projects have these. Published versions usually don't, but everyone who's done one knows what I'm talking about.

**1. Random-init loss = 242 instead of ≈ ln(8000) ≈ 9.**

A random-init transformer with vocab 8000 should output a near-uniform softmax at step 0 → cross-entropy ≈ `ln(8000) ≈ 8.99`. If you see 242 instead, the model isn't actually random-initialized — its weights are way too big and logits are exploding.

The bug: PyTorch's default `nn.Embedding` initialization is `N(0, 1)`. GPT-style transformers want `N(0, 0.02)` — about 50× smaller. Without a custom `_init_weights` setting the embedding scale and applying GPT-2's `std = 0.02/√(2·n_layer)` scaling on residual output projections, the model starts in a broken regime, training spends many steps merely climbing back to "random uniform" before it begins to actually learn, and depending on the recipe it can also just diverge.

The fix is canonical. The *useful* part is the lesson: **random-init loss is a sanity check, not telemetry.** It must be ≈ `ln(vocab_size)`. If it isn't, you stop and fix it before burning training compute.

**2. A 40-minute kernel hang on a step that should take ~10 seconds.**

BPE tokenizer training on a 48 MB corpus — Rust-backed HuggingFace `tokenizers` should finish in under a minute. Mine was hanging for 40 minutes. The code was fine. The Kaggle interactive kernel had effectively dead-locked. Same code, restarted kernel: 10 seconds.

The actionable rule: **know what each operation should take, and treat a large overrun as a fault to diagnose, not slowness to wait out.** A 40× overrun is a signal, not a long wait.

**3. The wrong-checkpoint comparison that "showed" RoPE losing −233 %.**

After re-running, my head-to-head cell printed a comparison where the RoPE arm had ~3× *worse* perplexity than baseline. Massively suspicious — and the cell would have produced a polished markdown results table I could have just committed without thinking too hard.

The signal that something was wrong was in the metadata, not the headline number: the row `best-val step = 4999 | 500`. Baseline reached its best at step 4999. RoPE's best was at step 500. That's the giveaway — both arms ran the same recipe, so "best at step 500" only happens if the run was *killed* at step ~500. The local `rope.pt` file was the partial output of an earlier, interrupted run that never got overwritten with the full-5000-step result. The comparison cell silently loaded the wrong file.

Generalised lesson: **structural asymmetries in your comparison metadata are the fastest way to catch silent data bugs.** Baseline at 4999, RoPE at 500 — that single mismatch makes the whole table immediately suspect, regardless of how dramatic the primary number looks.

## Honest limitations — read these before citing the +12.1 %

1. **Single seed per arm.** One realisation. The population mean of the gap is unknown. A defensible publication would run 3–5 seeds per arm and report mean ± std. The direction is consistent with the broader literature; the precise magnitude is a single data point.
2. **Small corpus, small model, multi-epoch.** 9.45 M training tokens against a 14 M-parameter model is ~50× below Chinchilla compute-optimal. Both arms make ~4.3 passes over the corpus; both overfit modestly. They overfit *symmetrically*, which preserves the ablation conclusion, but the absolute PPL numbers (40, 36) don't reflect what either architecture could do with proper compute.
3. **Domain-specific tokenizer.** Byte-level BPE on `eess.SP` is excellent on RF/telecom text and degrades sharply off-domain. The absolute PPL numbers are not comparable across corpora — this is a deliberate scope choice, not a hidden flaw, but it does cap the generalisation claim.
4. **256-token context only.** RoPE's much-cited length-extrapolation advantages are not exercised by this setup. This is "within training distribution lengths only."
5. **No statistical-significance test.** With one run per arm, the appropriate framing is *"directionally consistent with the published RoPE literature on small-scale tasks"* — not *"statistically significant 12.1 % improvement."*

## What v2 would look like

Roughly in priority order:

- **3–5 seeds per arm** — by far the cheapest, highest-value next experiment; converts the point estimate into a credibility interval.
- **More compute per run** so both arms train past the obvious multi-epoch regime — does the gap widen or narrow with training?
- **Held-out long-context evaluation** — the more famous RoPE advantage is length-extrapolation; this setup doesn't probe it.
- **Broader comparison** — RoPE vs. learned PE vs. ALiBi vs. no positional encoding at all, same recipe. The "learned vs. RoPE" axis is one cut; the landscape is bigger.

## Repo + reproducibility

- **GitHub:** [github.com/MuhammadHanzalaIqbal/nanoGPT-rf-rope-ablation](https://github.com/MuhammadHanzalaIqbal/nanoGPT-rf-rope-ablation)
- **W&B (training curves, both runs):** [wandb.ai/rjrajpoot01-namal-university/rf-rope-ablation](https://wandb.ai/rjrajpoot01-namal-university/rf-rope-ablation)

The notebook is a single file with cache-or-build guards on every expensive step. Full run from scratch on a free Kaggle T4 is ~30 min; reusing the cached checkpoints from a prior Commit (attached as an Input) takes ~2 min. The arXiv dataset is fetched via `kagglehub` so paths are never hardcoded.

---

*Comments, criticism, and pull requests welcome.*
