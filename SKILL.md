# MI Analysis Operational Reference

Operational guide for running the tealeaves pipeline from prep through analysis. Written for a collaborator fluent in transformer internals, attention mechanics, and logit lens methodology.

## Pipeline Orchestration

### Stage 1: Region Annotation (local)

Define regions via `regions.json` (named character spans in the prompt text). Three detection strategies:

- **Marker-based**: `"start_marker": "## Rules", "end_marker": "## Examples"` (literal text boundaries)
- **Regex-based**: `"start_pattern": "Task \\d+:", "end_pattern": "Task \\d+:|$"` (for repeating structures)
- **Character range**: `"start_char": 0, "end_char": 500` (explicit offsets)

Assemble test cases:
```bash
python -m tealeaves.prep.inputs \
    --prompt system_prompt.txt \
    --regions regions.json \
    --conversations conversations.json \
    --output test_cases.json
```

The output `test_cases.json` contains the full prompt text, character-level region annotations, query position definitions, and tracked token list.

### Stage 2: GPU-Side Analysis

`run_analysis.py` is self-contained: no package imports, only torch/transformers/stdlib. Deploy via scp:

```bash
scp src/engine/run_analysis.py gpu:/workspace/
scp test_cases.json gpu:/workspace/
ssh gpu

# On GPU box:
bash /workspace/vastai_setup.sh  # installs deps, downloads model
python /workspace/run_analysis.py \
    --input /workspace/test_cases.json \
    --output /workspace/results/ \
    --model-path /workspace/models/YourModel \
    --tracked-tokens "<" "keyword" \
    --query-positions terminal
```

Key flags:
- `--model-path`: Local path to HF model weights (required)
- `--tracked-tokens`: Tokens to track in logit lens (default: none)
- `--query-positions`: Which positions to analyze (default: from test_cases.json)
- `--cases`: Specific case indices to run (default: all)
- `--no-per-token`: Skip per-token attention capture (faster, but no heatmaps)
- `--top-k`: Number of top logit lens predictions to save per layer (default: 10)

The engine auto-discovers model architecture (layer count, head config, attention module paths, LM head, final norm) from `model.config` and module tree walking.

### Stage 3: Result Retrieval

```bash
scp -r gpu:/workspace/results/ ./data/results/
```

Each result JSON contains: per-token attention weights (all layers), logit lens projections (tracked tokens), region map (token-level), and metadata.

### Stage 4: Visualization

```bash
# Heatmap: "where is the model looking at position X?"
python -m tealeaves.render.heatmap \
    --result data/results/case_0.json \
    --position terminal \
    --layers L60-63 \
    --mask-chatml \
    --clip-low 0.05 \
    --colormap inferno

# Cooking curves: "how does attention to each region evolve through the forward pass?"
python -m tealeaves.render.cooking_curves \
    --result data/results/case_0.json \
    --position terminal \
    --normalize per-region   # or 'raw' for absolute values

# Layer GIF: "watch attention flow through all layers"
python -m tealeaves.render.layer_gif \
    --result data/results/case_0.json \
    --position terminal \
    --mask-chatml \
    --fps 4 --stride 1

# Aggregate: "is this pattern stable across samples?"
python -m tealeaves.render.aggregate \
    --base-dir data/results/ \
    --variants baseline:Baseline modified:Modified
```

### Stage 5: Comparative Analysis

```bash
# N-variant comparison with delta tables
python -m tealeaves.analysis.compare \
    --base-dir data/results/ \
    --variants baseline:Baseline anchor:Anchor trim3:Trim3 \
    --ratio conv_turns:current_message \
    --logit-lens-tokens "<" \
    --by-seed \
    --metrics all

# Markdown experiment reports
python -m tealeaves.analysis.report \
    --base-dir data/results/ \
    --experiments baseline:Baseline:results_baseline anchor:Anchor:results_anchor \
    --output-dir reports/
```

## Vast.ai Workflow

### Provisioning

Search for instances with CUDA 12.x base image and sufficient VRAM. Single-GPU instances only. See [Empirical Notes](docs/EMPIRICAL_NOTES.md) for memory estimation and why multi-GPU fails.

### Bootstrap

```bash
scp infra/vastai_setup.sh gpu:/workspace/
scp src/engine/run_analysis.py gpu:/workspace/
scp test_cases.json gpu:/workspace/

# MODEL_ID is required (no default)
ssh gpu 'HF_TOKEN=your_token MODEL_ID=meta-llama/Llama-3-8B bash /workspace/vastai_setup.sh'
```

The setup script installs torch/transformers/accelerate, authenticates with HuggingFace (if HF_TOKEN set), and downloads model weights. `MODEL_ID` must be specified; the script will error if it's missing.

### Execution Monitoring

Watch for:
- Model loading: "Loading model from..." (takes 2-5 min for 32B models)
- Per-case progress: "Processing case N/M" (each case takes 30-90s depending on sequence length)
- Memory: peak should be near `model_params * 2 + 5GB`. If it grows between cases, there's a leak.
- Attention capture confirmation: "Registered hooks on N layers"

### Result Size

Each result JSON is 5-50MB depending on sequence length and number of layers. Budget ~500MB per 10-case experiment.

## Visualization Guide

### Which renderer for which question?

| Question | Renderer | Key flags |
|----------|----------|-----------|
| Where does the model look at position X? | `heatmap` | `--layers`, `--mask-chatml` |
| How does region attention evolve across layers? | `cooking_curves` | `--normalize` |
| What's the full layer-by-layer attention dynamics? | `layer_gif` | `--fps`, `--stride` |
| Is this pattern stable across samples? | `aggregate` | `--variants` |

### Flag guidance

- `--mask-chatml`: Exclude ChatML structural tokens from rank normalization. Use for content-focused analysis. Without it, `<|im_start|>` and `<|im_end|>` dominate the colormap.
- `--clip-low N`: Set attention values below the Nth percentile to zero. Use 0.05-0.10 to cut noise floor.
- `--normalize per-region`: Each region's cooking curve normalized to its own [0,1] range. Use for comparing trajectory SHAPES across regions of vastly different magnitude.
- `--normalize raw`: Absolute attention values. Use for seeing actual attention budget allocation. Current_message will dominate.
- `--layers L60-63`: Analyze specific layers. "L60-63" = final 4 layers = "what the model decided." "L0-8" = early attention = rules absorption phase.
- `--smoothing N`: Gaussian smoothing sigma for heatmaps. Higher = smoother. Default 0 (no smoothing).

## Data Conventions

### Directory structure

```
data/
    results_baseline/
        sample_0.json
        sample_1.json
        ...
    results_variant_a/
        sample_0.json
        ...
```

### Result JSON schema

```json
{
  "case_id": "case_0",
  "model": "Qwen/Qwen3-32B",
  "num_layers": 64,
  "num_tokens": 2048,
  "region_map": {"rules": [100, 250], "examples": [250, 400], ...},
  "piece_boundaries": {"system": [5, 500], "user": [502, 800], "assistant": [802, 810]},
  "positions": {
    "terminal": {
      "token_idx": 2047,
      "per_token_attention": {"0": [...], "1": [...], ...},
      "logit_lens": {"0": {"top_k": [...], "tracked": {...}}, ...}
    }
  }
}
```

- `region_map`: Maps region name to `[start_token, end_token)` half-open intervals
- `piece_boundaries`: Maps chat piece name to token range
- `per_token_attention`: Layer index (string) -> array of float32 attention weights, one per token. Head-averaged.
- `logit_lens.tracked`: Token string -> `{"rank": int, "prob": float}` at each layer
