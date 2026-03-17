# TeaLeaves

Mechanistic interpretability pipeline: annotate regions in any prompt, capture attention and logit lens on any HuggingFace model, visualize and compare variants with empirical evidence.

## How to Think About Results

This pipeline measures how a model distributes attention across every region of a prompt at every layer. When interpreting results or helping with analysis:

- **Cooking curves are the primary diagnostic.** They show per-region attention trajectories across all layers. The shape matters more than the magnitude: clean phase separation (rules peak early, current_message dominates mid-layers, output_format takes over final layers) indicates a well-structured prompt.
- **Per-region normalization** (`--normalize per-region`) compares trajectory shapes across regions of different magnitudes. Raw mode shows actual attention budget allocation.
- **Terminal attention** (final 4 layers averaged) represents "what the model decided." Use `avg_final_layers()` for this. Single-layer measurements are noisy.
- **Per-token density** (`attention / n_tokens`) is the fair cross-region comparison. Raw attention sums are dominated by region length.
- **Region ratios** (e.g., `conv_turns / current_message`) reveal relative priority. A ratio >2x at terminal position indicates context bleed.
- **Logit lens rank trajectories** show when the model "decides" on a token. A rank crash (e.g., rank 4 to rank 64K) means competing representations overwrote the prediction. Recovery by terminal layers means the conflict resolved.
- **Multi-seed testing is mandatory** before declaring a variant successful. A metric that varies >50% across seeds is fragile.

### Layer Phases

Phases scale dynamically to any layer count (`display_phases()` and `analysis_phases()` in `constants.py`). For a 64-layer model:

| Phase | Layers | What happens |
|-------|--------|--------------|
| Broad read | 0-6 | Model reads everything. Rules get 14x more attention than at terminal. |
| Absorption | 7-11 | Rule content absorbed into residual stream. Attention drops sharply. |
| Compression | 12-31 | Quiet middle. Information is being integrated. |
| Re-engagement | 32-47 | Current message dominates. Context-dependent processing. |
| Output prep | 48-63 | Model commits to output. Examples and format tokens dominate. |

### Red Flags

- Context bleed ratio >2x: conversation history dominates over current message
- Cooking curve that never peaks: the region has no influence at any layer
- Region with <0.1% per-token density at its expected influence point: model ignores it
- Format token rank >1000 at terminal layer: format compliance likely broken
- Rules still high at output prep layers: persistent influence (or confusion)

## Constraints That Will Cause Bugs If Violated

- **`run_analysis.py` must stay self-contained.** Zero package imports (no `from tealeaves...`). It gets scp'd to GPU boxes as a single file. Model discovery logic is inlined from `model_adapter.py`. Changes to model discovery must be synced in both files manually.
- **Single GPU only** (`device_map={"": 0}`). Multi-GPU causes OOM between cases due to accelerate's `AlignDevicesHook` leaking state.
- **`attn_implementation="eager"` is mandatory.** Flash attention doesn't materialize the attention matrix.
- **Hook `self_attn`, not the decoder layer.** Accelerate's hooks on decoder layers fire before user hooks, corrupting capture.
- **No hardcoded layer numbers in renderers.** All phase boundaries come from `display_phases(num_layers)` and `analysis_phases(num_layers)`.
- **SKIP_REGIONS in `constants.py`** lists container regions (like `system_prompt`, `chat_template`) that should never be plotted as individual curves.

## Running

```bash
# Tests
pytest

# Prep (local)
python -m tealeaves.prep.inputs --prompt X --regions X --conversations X --output X

# Engine (GPU box, standalone)
python run_analysis.py --input X --output X --model-path X [--tracked-tokens "<"] [--no-per-token]

# Render
python -m tealeaves.render.heatmap --result X [--mask-chatml] [--clip-low 0.05]
python -m tealeaves.render.cooking_curves --result X [--normalize per-region]
python -m tealeaves.render.layer_gif --result X [--mask-chatml] [--fps 4]
python -m tealeaves.render.aggregate --base-dir X --variants name:Label

# Analysis
python -m tealeaves.analysis.compare --base-dir X --variants name:Label [--ratio a:b] [--by-seed]
python -m tealeaves.analysis.report --base-dir X --experiments key:label:dir --output-dir X
```

## Conventions

- All experimental data goes in `data/` (gitignored): `data/inputs/`, `data/results_variant_name/sample_N.json`, `data/reports/`
- Renderers follow: `main()` + `if __name__` block, import shared utilities from `render/_shared.py`, data loading from `render/loaders.py`
- New metrics go in `analysis/metrics.py`, used by `compare.py` or `report.py`
- Region detection strategies go in `prep/regions.py`

## Deep-Dive Documentation

- [PIPELINE_EXPLAINED.md](docs/PIPELINE_EXPLAINED.md): How region annotation, tokenization, attention hooks, and logit lens work mechanically
- [PITFALLS.md](docs/PITFALLS.md): Failure modes discovered empirically (OOM, hook ordering, BPE boundaries, offset mapping)
- [KNOWN_GOOD_APPROACHES.md](docs/KNOWN_GOOD_APPROACHES.md): Patterns validated across multiple experiments
- [SKILL.md](SKILL.md): Full operational reference for running the pipeline end-to-end
