# TeaLeaves

<img src="docs/images/tealeaves_icon.png" alt="TeaLeaves icon" width="128">

This is a people-friendly way of seeing how the model is interpreting your prompts in preparation for the inevitable next token prediction. My theory when making this (that was validated after I used/built TeaLeaves) was that very small models (1b,3b) were nearly as capable as larger (8b,12b) models at really high quality (but tightly bounded) structured outputs. There is a maxiumum "attention & skill" density in every model by the limit of the weights. If the layers arent' singing in harmony then they're not going to be as effective. TeaLeaves allows you to visualize in a very fast-and-loose way if your prompt changes made a demonstrible difference in the chances of the output being what you asked for. For example, if you are asking a model for {structured_output} and you can get that *understanding* out of the way early in the generation process the model has more *time* to focus on the rest of your task(s).

TeaLeaves is designed for LLM-as-a-first-class-citize usage. There are runbooks and documentation your Model can invoke. The repo exposes all the functionality you need to input data into a testbed (local) model, recieve the output, and have Claude or whoever interpret the floats.

The reason for development was that I needed: entity extraction, relevant passage filtering, complexity classification, and query expansion on 4k tokens in 750ms or less RTT. That means using models that are not built for 5 disparate tasks in one yield. TeaLeaves was my solution. Sometimes it results in FUNKY looking prompts (to the human eye at least) but they're very effective.

---

Mechanistic interpretability pipeline for prompts. Annotate regions in any prompt, run attention capture and logit lens on any HuggingFace model, get back heatmaps, cooking curves, animated layer sweeps, and comparative analysis showing what changed and where.

Existing MI tools (TransformerLens, NNsight, pyvene) are libraries: they give you hooks and activation access to build your own analysis. TeaLeaves is a pipeline that takes a structured prompt and produces attention heatmaps, cooking curves, and variant comparisons.

## What it does

- **Region annotation**: Define named spans in your prompt via JSON config (markers, regex, or char ranges). The pipeline maps these to token positions, handling BPE boundary effects and chat template artifacts automatically.
- **Attention capture**: Hooks every attention layer to extract head-averaged attention weights at configurable query positions
- **Logit lens**: Projects the residual stream through the final norm + LM head at each layer to track token rank trajectories
- **Visualization**: Four renderers: per-token heatmaps, per-region cooking curves, animated layer sweeps, and multi-sample aggregates with confidence bands
- **Variant comparison**: Automated N-variant comparison with delta tables, multi-seed stability analysis, and markdown reports

### Before and after

The cooking curves below show the same prompt before and after iterative tuning (both per-region normalized to 0–1 for direct comparison).

The "before" curve is from a first-draft prompt. Several regions show artifact peaks at L0 from near-zero values. `current_message` has a narrow, isolated spike around L42–45 but no sustained dominance. `task_entity` unexpectedly takes over the output formatting layers (L56+), and the mid-layers are noisy with no clear phase differentiation:

![Before tuning, attention is unfocused](docs/images/cooking_before.png)

After several rounds of restructuring guided by the pipeline's output (adjusting region boundaries, reordering task sections, adding structural markers), the curves show clean phase progression. `directive` and `conversation_turns` peak first (L3), rules regions differentiate through the early-mid layers, `current_message` sustains dominance through the focus layers (L40–50), and `output_format` cleanly takes over the final layers. The mid-layer oscillations are structured rather than noisy:

![After tuning, attention is focused and phase-separated](docs/images/cooking_after.png)

Each change was validated by re-running the pipeline and comparing curves against the baseline.

## Quick start

```bash
# Install (local, rendering and analysis only)
pip install -e .

# Install with GPU dependencies (for running the engine)
pip install -e ".[gpu]"
```

### 1. Define regions

Create a `regions.json` describing named spans in your prompt:

```json
{
  "system_prompt": {
    "regions": [
      {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
      {"name": "examples", "start_marker": "## Examples", "end_marker": null}
    ]
  },
  "user_message": {
    "regions": [
      {"name": "context", "start_marker": "Previous:", "end_marker": "Current:"},
      {"name": "current", "start_marker": "Current:", "end_marker": null}
    ]
  }
}
```

### 2. Prepare inputs

```bash
python -m tealeaves.prep.inputs \
    --prompt system_prompt.txt \
    --regions regions.json \
    --conversations conversations.json \
    --output test_cases.json
```

### 3. Run analysis on GPU

```bash
# scp the self-contained engine, setup script, and test cases to your GPU box
scp src/engine/run_analysis.py gpu:/workspace/
scp infra/vastai_setup.sh gpu:/workspace/
scp test_cases.json gpu:/workspace/

# Bootstrap the GPU box (MODEL_ID is required)
ssh gpu 'MODEL_ID=meta-llama/Llama-3-8B bash /workspace/vastai_setup.sh'

# Run analysis
ssh gpu 'python /workspace/run_analysis.py \
    --input /workspace/test_cases.json \
    --output /workspace/results/ \
    --model-path /workspace/models/Llama-3-8B \
    --tracked-tokens "<" "keyword"'
```

### 4. Render results

```bash
# Per-token attention heatmap
python -m tealeaves.render.heatmap --result results/case_0.json --mask-chatml

# Per-region attention trajectories across layers
python -m tealeaves.render.cooking_curves --result results/case_0.json --normalize per-region

# Animated layer sweep
python -m tealeaves.render.layer_gif --result results/case_0.json --mask-chatml

# Multi-sample aggregate with confidence bands
python -m tealeaves.render.aggregate --base-dir results/ --variants baseline:Baseline
```

### 5. Compare variants

```bash
python -m tealeaves.analysis.compare \
    --base-dir results/ \
    --variants baseline:Baseline modified:Modified \
    --ratio context:current_message

python -m tealeaves.analysis.report \
    --base-dir results/ \
    --experiments baseline:Baseline:results_baseline modified:Modified:results_modified \
    --output-dir reports/
```

## Input formats

The pipeline takes three input files. All content is model-agnostic: any system prompt, any conversation structure, any region definitions.

### `system_prompt.txt`

Plain text file containing the full system prompt. This is the exact text that will be inserted into the chat template's system role.

### `conversations.json`

Array of conversation objects. Each object represents one test case for MI analysis.

```json
[
  {
    "id": "case_0",
    "user_message": "What's the weather like in Tokyo?",
    "response": "Tokyo is currently experiencing mild temperatures around 18°C."
  },
  {
    "id": "case_1",
    "user_message": "Tell me more about the forecast.",
    "response": "The week ahead shows increasing cloud cover with rain expected Thursday."
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Unique case identifier (used for output filenames) |
| `user_message` | yes | The user turn to analyze |
| `response` | yes | The assistant response (can be empty string `""` if analyzing pre-response attention) |
| `user_regions` | no | Per-case region defs for user message (overrides global `user_message.regions`) |
| `response_regions` | no | Per-case region defs for response (overrides global `response.regions`) |

Conversations can come from any source: exported from a chat database, hand-written, pulled from logs, or generated synthetically.

### `regions.json`

Defines named text spans to track attention over. Regions are matched against the assembled prompt text at character level, then resolved to token positions.

```json
{
  "system_prompt": {
    "regions": [
      {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
      {"name": "examples", "start_marker": "## Examples", "end_marker": null}
    ]
  },
  "user_message": {
    "regions": [
      {"name": "context", "start_marker": "Previous:", "end_marker": "Current:"},
      {"name": "current", "start_marker": "Current:", "end_marker": null}
    ]
  },
  "response": {
    "regions": [
      {"name": "answer", "start_pattern": "^\\w", "end_pattern": null}
    ]
  },
  "query_positions": {
    "terminal": "last_token",
    "decision": {"after_text": "Answer:"}
  },
  "tracked_tokens": ["<", "yes", "no"]
}
```

**Region detection strategies**: each region def needs a `name` and one of these boundary strategies:

| Strategy | Fields | Use when |
|----------|--------|----------|
| Marker | `start_marker`, `end_marker` | Boundaries are literal text strings in the prompt |
| Regex | `start_pattern`, `end_pattern` | Boundaries need pattern matching |
| Character range | `start_char`, `end_char` | You know exact character offsets |

Set `end_marker`, `end_pattern`, or `end_char` to `null` to extend to end of text. Regions can also be nested by including a `regions` array inside a region def.

**Query positions** define where in the token sequence to probe attention and logit lens:

| Value | Meaning |
|-------|---------|
| `"last_token"` | Last token of the user message |
| `{"after_text": "..."}` | First non-whitespace token after the specified text in the response |
| `{"at_text": "..."}` | Token at the specified text in the response |

**Tracked tokens** are specific tokens to monitor rank and probability for across all layers in the logit lens output.

## Why this exists

Prompt engineering typically relies on eyeballing model outputs after each change. A tweak that seems to improve one case might silently degrade others.

TeaLeaves captures how the model distributes attention across every region of your prompt at every layer, so you can measure whether a change helped or hurt, and where in the forward pass the effect occurs.

The pipeline was originally built to tune the subcortical prompt for [Mira](https://github.com/taylorsatula/mira-OSS), a persistent digital entity with self-directed memory and context window management. Because subcortical runs on a small model in a single forward pass with no reasoning, every word in the prompt either drives attention or wastes it. The techniques generalize to any prompt and any model.

### Layer-by-layer attention heatmap

![Per-token attention heatmap animated across layers](docs/images/heatmap.gif)

## Model support

The engine auto-discovers model architecture from any HuggingFace decoder-only transformer:
- Reads layer count, head counts, hidden size, vocab size from `model.config`
- Walks the module tree to find attention submodules, LM head, and final norm
- Phase annotations and layer-dependent rendering scale automatically to any layer count

### Verified model families

| Family | Example model | Chat template | Layers | Notes |
|--------|--------------|---------------|--------|-------|
| **Qwen** | Qwen3-32B | ChatML | 64 | Full support |
| **Llama 3** | Llama-3.1-8B-Instruct | Llama 3 format | 32 | Full support |
| **Mistral** | Mistral-7B-Instruct-v0.1 | `[INST]` format | 32 | Full support |
| **Gemma** | Gemma-2-9B-IT | Gemma format | 42 | System role auto-merged into user |
| **GPT (OpenAI)** | gpt-oss-20b | OpenAI format | 24 | Full support (MoE) |

Models without a system role (Gemma) are handled automatically: the engine merges system content into the first user message. Any HuggingFace model with a chat template and eager attention support should work.

Requirements: `attn_implementation="eager"` (flash attention doesn't materialize the attention matrix).

## GPU requirements

Rule of thumb: `model_params * 2 bytes + 5GB headroom` (fp16 weights + attention capture overhead).

| Model | VRAM needed | Recommended GPU |
|-------|-------------|-----------------|
| 8B params | ~21GB | A100 40GB |
| 32B params | ~69GB | H100 80GB |
| 70B params | ~145GB | Won't fit single GPU; use quantization |

See [Empirical Notes](docs/EMPIRICAL_NOTES.md) for memory estimation details and OOM prevention.

## Documentation

- **[SKILL.md](SKILL.md)**: Operational reference for running the full pipeline with all flags
- **[Pipeline Explained](docs/PIPELINE_EXPLAINED.md)**: How region annotation, attention hooks, logit lens, and per-token capture work mechanically
- **[Empirical Notes](docs/EMPIRICAL_NOTES.md)**: What broke, why, and what works — failure modes and validated patterns
