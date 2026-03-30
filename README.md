# Operator-Level HE/MPC Execution Framework

This project is an operator-level execution framework for BERT-style inference experiments across:

- all operators on MPC
- all operators on HE
- mixed HE/MPC execution with automatic domain conversion

The main repository value is the framework itself: routing, operator abstraction, capability tracking, conversion insertion, validation scripts, and plotting scripts. The `baseline/` directory is mostly a local workspace for third-party baselines plus a small amount of project-specific reproduction glue and result summaries.

## Current Scope

Implemented repository components include:

1. Operator abstraction and routing for BERT-style operators:
   - `Attention_QK_MatMul`
   - `Softmax`
   - `Attention_V_MatMul`
   - `Residual_Add`
   - `LayerNorm`
   - `GeLU`
   - `FFN_Linear_1`
   - `FFN_Linear_2`
2. Runtime-configurable backend placement via JSON configs.
3. Automatic insertion of conversion operators:
   - `HE_to_MPC`
   - `MPC_to_HE`
4. Validation and experiment entry points for real and mocked backend paths.
5. Baseline comparison scripts for CPU and GPU figures.

## Repository Layout

- `analysis/`: operator inventory, integration notes, and planning notes.
- `backends/`: backend adapters and backend-specific helper logic.
- `bridge/`: bridge-layer code used by wrapped backend kernels.
- `compiler/`: execution planning and min-cut related logic.
- `configs/`: shared configuration assets.
- `docs/`: architecture notes and handoff documentation.
- `experiments/`: runnable validation and experiment scripts.
- `figure/`: scripts and generated figures for baseline comparison.
- `framework/`: capability registry, adapter mapping, and routing core.
- `ir/`: intermediate representation utilities.
- `operators/`: operator definitions and wrappers.
- `runtime/`: runtime conversion and profiling logic.
- `tests/`: lightweight repository tests.
- `baseline/`: third-party baselines plus local reproduction helpers and result summaries.

## Quick Start

Run from `he_compiler/operator_execution_framework`:

```bash
python experiments/run_experiment.py --config operator_backend_config.json
python experiments/run_experiment.py --config experiments/configs/all_mpc.json
python experiments/run_experiment.py --config experiments/configs/all_he.json
python experiments/run_experiment.py --config experiments/configs/mixed_he_mpc.json
```

Useful validation entry points:

```bash
python experiments/validate_gelu_wrapper.py
python experiments/validate_softmax_wrapper.py
python experiments/validate_layernorm_wrapper.py
python experiments/validate_ffn_linear1_wrapper.py
python experiments/validate_ffn_linear2_mpc_wrapper.py
python experiments/validate_qk_matmul_wrapper.py
python experiments/validate_attn_v_matmul_wrapper.py
```

HE/NEXUS-oriented validation entry points:

```bash
python experiments/validate_gelu_he_nexus_wrapper.py
python experiments/validate_softmax_he_nexus_wrapper.py
python experiments/validate_layernorm_he_nexus_wrapper.py
python experiments/validate_ffn_linear1_he_nexus_wrapper.py
python experiments/validate_ffn_linear2_he_nexus_wrapper.py
python experiments/validate_qk_matmul_he_nexus_wrapper.py
python experiments/validate_attn_v_matmul_he_nexus_wrapper.py
```

## Baseline Policy

`baseline/` currently mixes two kinds of content:

- local copies of large upstream baseline repositories
- small project-specific files that are actually worth keeping with this framework

For GitHub, the default recommendation is:

- keep the framework code outside `baseline/`
- keep only the project-specific baseline helpers, traces, and summary files
- do not commit nested `.git/`, `build/`, `output/`, local run logs, or other generated artifacts from the third-party baselines

## Baseline Files Worth Keeping

These files are the ones most worth preserving because they look project-specific, lightweight, or directly tied to your reproduction workflow:

- `baseline/BLB/scripts/run_local_blb_bert_large_repro.py`
- `baseline/BLB/scripts/simulate_blb_communication.py`
- `baseline/BLB/0001-.patch`
- `baseline/baseline_res.txt`
- `baseline/baseline_gpu_res.txt`
- `baseline/blb_communication.json`
- `baseline/bumble_communication.json`
- `baseline/SHAFT_communication.json`
- `baseline/mpc_part_latency.txt`
- `baseline/communication_statistic_prompt.txt`

These are reasonable to keep only if they are part of the paper/report/figure pipeline:

- `baseline/BLB/results/`
- `figure/cpu/latency_bar_log.png`
- `figure/gpu/latency_bar_log.png`

These baseline directories are mostly upstream code snapshots and are usually better handled as submodules, setup scripts, or documented external dependencies instead of direct uploads:

- `baseline/EzPC/`
- `baseline/HE_GPU/`
- `baseline/NEXUS/`
- `baseline/OpenBumbleBee/`
- `baseline/SHAFT/`
- `baseline/einhops/`
- `baseline/fhelipe/`
- `baseline/spu/`

`baseline/BLB/` is mixed: it contains a large upstream codebase, but also local scripts and result folders that are useful for reproduction. If you trim it before publishing, keep the project-specific files listed above and exclude build artifacts such as `SCI/build/`, `SCI/tests/build/`, `SCI/output/`, `local_runs/`, and `local_runs_stage/`.

## Where To Reproduce Baselines

If you keep only lightweight reproduction helpers in GitHub, document the actual reproduction entry points as follows.

BLB / MPC large-block reproduction:

- workspace root: `baseline/BLB/`
- local runner: `baseline/BLB/scripts/run_local_blb_bert_large_repro.py`
- communication replay: `baseline/BLB/scripts/simulate_blb_communication.py`
- expected binary location used by the runner: `baseline/BLB/SCI/tests/build/bin/ckks_bert_large_main`
- upstream build details: `baseline/BLB/README.md`
- SCI build details: `baseline/BLB/SCI/README.md`

NEXUS / HE reproduction:

- workspace root: `baseline/NEXUS/`
- CUDA backend notes: `baseline/NEXUS/cuda/README.md`
- general project build details: `baseline/NEXUS/README.md`

OpenBumbleBee / SPU-style reproduction:

- workspace root: `baseline/OpenBumbleBee/`
- setup notes: `baseline/OpenBumbleBee/README.md`
- installation notes: `baseline/OpenBumbleBee/INSTALLATION.md`

SHAFT reproduction:

- workspace root: `baseline/SHAFT/`
- usage notes: `baseline/SHAFT/README.md`

Other comparison baselines:

- `baseline/EzPC/README.md`
- `baseline/HE_GPU/CAT/README.md`
- `baseline/fhelipe/README.md`
- `baseline/einhops/README.md`
- `baseline/spu/README.md`

In practice, this means the GitHub version of this repository should describe `baseline/` as a reproduction workspace, not as the canonical source of all third-party systems.

## Plotting Baseline Comparisons

CPU and GPU comparison scripts live here:

- `figure/cpu/baseline_compare.py`
- `figure/gpu/baseline_compare.py`

If the `.png` files are regenerated easily, prefer keeping the scripts and regenerating the figures locally.

## Capability Registry

The framework exposes a queryable capability/status registry:

- module: `framework/capabilities.py`
- adapter mapping: `framework/adapters.py`
- status values: `real-integrated`, `mock`, `unsupported`

The general workflow is to keep the router and operator interface stable while incrementally replacing mock implementations with real wrapped HE/MPC kernels.

## Suggested GitHub Upload Set

For a clean public or internal GitHub repository, the recommended committed set is:

- all source directories outside `baseline/`
- experiment scripts and JSON configs
- docs and analysis notes
- plotting scripts
- selected lightweight files from `baseline/` listed in `Baseline Files Worth Keeping`
- the root `.gitignore`

The recommended excluded set is:

- nested baseline repositories as raw vendored copies
- nested `.git/`
- `__pycache__/`
- `build/`, `output/`, and local run directories
- generated binaries and temporary files

## Notes

- This directory did not previously have a root `.gitignore`; one has now been added to exclude obvious caches, build outputs, nested baseline `.git/`, and temporary files.
- The older README referenced paths such as `EzPC_bolt/...` that do not match the current repository layout. This README has been updated to point at the actual in-tree baseline locations under `baseline/`.
