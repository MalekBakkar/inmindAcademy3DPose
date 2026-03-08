# Point Cloud Registration Pipeline

This project registers a source point cloud to a target point cloud using:
- Global feature-based alignment (RANSAC + FPFH)
- Local refinement (robust point-to-plane ICP)
- A tuner that searches configuration parameters

## Setup

```bash
uv sync
```

## Config Files

All YAML config files are under `config/`:
- `config/config.yaml`: baseline configuration
- `config/best_config.yaml`: best config found by tuning

## Two Main Runs

1. Tune parameters and save the best config:

```bash
uv run tune_config.py
```

2. Run registration using the tuned best config:

```bash
uv run main.py --config config/best_config.yaml
```

If you want a baseline comparison, run:

```bash
uv run main.py --config config/config.yaml
```

## `registration.py` Algorithm

`register(pcd1, pcd2)` follows this sequence:

1. Preprocessing
- Voxel downsampling with `voxel_size`
- Normal estimation with:
  - `normals.radius_multiplier`
  - `normals.max_nn`
- FPFH feature extraction with:
  - `features.radius_multiplier`
  - `features.max_nn`

2. Global registration (RANSAC)
- Uses `registration_ransac_based_on_feature_matching`
- Distance threshold: `voxel_size * ransac.distance_multiplier`
- Key parameters:
  - `ransac.ransac_n`
  - `ransac.edge_length_threshold`
  - `ransac.max_iterations`
  - `ransac.confidence`
- Produces a coarse initial transform (`trans_init`)

3. Local refinement (robust ICP)
- Re-estimates normals on full clouds
- Runs point-to-plane ICP with:
  - `icp.distance_threshold`
  - `icp.max_iterations`
- Uses Tukey robust loss (`robust_kernel.k`) to reduce outlier influence
- Returns the final 4x4 transform

## `tune_config.py` Search Strategy

- Starts from `config/config.yaml`
- Builds a discrete search space for all major parameters
- Evaluates up to `MAX_TRIALS` random unique samples (seeded for reproducibility)
- Includes the base config as one trial
- Each configuration is evaluated multiple times (`EVAL_RUNS_PER_CONFIG`) to reduce randomness effects from RANSAC

The tuner saves only:
- `config/best_config.yaml`

It no longer writes CSV files or plot PNGs.

## Tuning Metrics and Score Formula

Per configuration, the tuner computes:
- `rmse`: inlier alignment error (lower is better)
- `fitness`: overlap quality / correspondence ratio (higher is better)
- `rmse_std`: stability across repeated runs (lower is better)
- `runtime_sec`: average runtime (lower is better, secondary)

Composite score (lower is better):

```text
score = rmse
      + FITNESS_WEIGHT * (1 - fitness)
      + RMSE_STD_WEIGHT * rmse_std
      + TIME_WEIGHT * runtime_sec
```

Current weights in code:
- `FITNESS_WEIGHT = 0.01`
- `RMSE_STD_WEIGHT = 0.30`
- `TIME_WEIGHT = 0.0002`

Why this score:
- `rmse` is the primary geometric accuracy objective
- `(1 - fitness)` penalizes poor overlap
- `rmse_std` penalizes unstable configs that only work occasionally
- `runtime_sec` is included as a small tie-breaker, not a dominant term

## Why Evaluation Threshold Is Fixed During Tuning

During tuning, all configs are evaluated with the same distance threshold (from base config, unless overridden by `FIXED_EVAL_THRESHOLD`).

Reason:
- If each config uses its own ICP/evaluation threshold, comparisons become biased
- Larger thresholds can inflate fitness/correspondence statistics
- A fixed threshold makes metrics directly comparable across candidate configs

## Troubleshooting (Linux Wayland)

If Open3D window creation fails, try:

```bash
XDG_SESSION_TYPE=x11 uv run main.py --config config/best_config.yaml
```
