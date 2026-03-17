# Empirical Notes

What works, what doesn't, and why. Discovered across multiple experiments.

## GPU Configuration

**Use single GPU `device_map={"": 0}`.** Multi-GPU via `device_map="auto"` causes two problems:

1. **OOM between cases**: accelerate's `AlignDevicesHook` leaks state across forward passes. Internal buffers accumulate and aren't freed. First case runs fine, second OOMs despite identical size.
2. **Hook ordering**: accelerate registers `AlignDevicesHook` on each decoder layer. User-registered forward hooks fire AFTER accelerate's hooks, so `send_to_device(output, input_device)` executes before your hook can capture or replace the attention matrix.

If the model doesn't fit on one GPU, use a smaller model or quantization.

### Memory Estimation

Rule of thumb: `model_params * 2 bytes + 5GB headroom` (fp16).

- Per-layer attention matrix: `batch * heads * seq_len * seq_len * 4 bytes` (float32 before our hook replaces it with None)
- Qwen3-32B: 64GB weights + 5GB overhead = 69GB peak on 80GB H100
- Llama-3-8B: 16GB weights + 3GB overhead = 19GB peak

## Attention Capture

**Hook `self_attn`, not the decoder layer.** The self_attn submodule has no accelerate hooks, so user hooks fire immediately after attention computation.

**`attn_implementation="eager"` is mandatory.** Flash attention doesn't materialize the attention matrix. Eager attention always does, regardless of `output_attentions` flag.

**Immediate None-replacement** of the attention output in hooks saves ~2.2GB per layer. Without this, HuggingFace accumulates 64 copies of the attention matrix (~140GB for a 64-layer model).

**`output_attentions` flag is irrelevant.** Many models' eager attention implementations return the full attention matrix regardless of this flag. The decoder layer discards attention unconditionally (`hidden_states, _ = self.self_attn(...)`). Hooks on self_attn capture it before the layer discards it.

## Tokenization

**Piecewise tokenization with full-sequence validation** is the only reliable way to establish piece boundaries. Tokenize each content piece independently, locate it in the `apply_chat_template()` output via subsequence matching, and validate token-by-token.

**Cumulative decode mapping** is the only robust char-to-token resolution for large regions. Two alternatives were tried and failed:

- **Offset mapping** (`return_offsets_mapping=True`): Returns `(0, 0)` for all special tokens, making them invisible. Chat template tokens are special tokens, so piece boundaries can't be established.
- **Subsequence matching**: Tokenize region text independently, search for that subsequence in the full sequence. Fails for regions >100 tokens because BPE tokenization of a substring differs from the same text in context. In testing, 10/11 system prompt regions failed.

The working approach: decode the full token sequence with `tokenizer.decode()`, find content strings via `str.find()`, map char positions to token indices via binary search with progressive prefix decoding. For sub-regions within pieces, use cumulative per-token `tokenizer.decode([tid])`.

**SentencePiece leading-space markers** (`▁`) cause per-token decode concatenation to differ from full-sequence decode. Always use full-sequence decode for piece-level boundary detection.

## Models Without System Role

Gemma and similar models raise `TemplateError: System role not supported`. The engine catches this automatically and merges system content into the first user message with a `\n` separator. Region boundaries remain correct because the system prompt text is still findable as a substring.

## Visualization

**Rank-based histogram equalization** for heatmaps eliminates power-law hotspot blowouts where 2-3 tokens have 100x the attention of the mean. Every percentile gets equal visual weight.

**`--mask-chatml`** excludes ChatML structural tokens (`<|im_start|>`, `<|im_end|>`) from rank normalization. Without it, these tokens dominate the colormap and compress the useful range for content tokens.

**Per-region normalization** (`--normalize per-region`) for cooking curves when comparing trajectory shapes across regions of vastly different magnitude. Use `raw` mode for actual attention budget allocation.

## Analysis

**4-layer terminal average** as the "what the model decided" signal. More stable than single-layer measurements. For a 64-layer model, this is L60-63.

**Per-token density** (`attention / n_tokens`) is the fair cross-region comparison. A 500-token region naturally captures more total attention than a 20-token region, even if the smaller region is per-token more important.

**Region ratios** (e.g., `conv_turns / current_message`) reveal relative priority better than absolute values. Absolute values shift with prompt length and content.

**Cooking curves** reveal dynamics that terminal-layer metrics miss entirely. Rules may peak at L0-8 with 14x more attention than at terminal. This absorption pattern is invisible in terminal-only analysis.

**Multi-seed testing** catches seed-dependent instabilities. A variant that looks good on one seed may regress on another. Test with 3-4 seeds before declaring success.

## Prompt Engineering

**Don't fight the recency gradient.** Placing directive content at the end of the prompt to leverage recency made the problem worse in testing: a terminal `<primary_focus>` directive increased context bleed by 19% and destroyed format compliance (`<` token rank 34K at L63, never recovers from the L48 crash). Place instructions where the model naturally reads them (early). Use structural markers (rare-token anchors) for salience rather than positional tricks.
