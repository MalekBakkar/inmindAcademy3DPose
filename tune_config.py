# tune_config.py

import copy
import itertools
import math
import time

import yaml
import numpy as np
import open3d as o3d

import registration  # this imports your register() and config dict


BASE_CONFIG_PATH = "config/config.yaml"
BEST_CONFIG_PATH = "config/best_config.yaml"

MAX_TRIALS = 1000
EVAL_RUNS_PER_CONFIG = 2
RNG_SEED = 42

# Evaluation threshold used for all configurations (fair comparison).
# If None, use base config's icp.distance_threshold as the fixed threshold.
FIXED_EVAL_THRESHOLD = None

# Composite objective (lower is better):
# score = rmse + fitness_penalty + stability_penalty + tiny_runtime_penalty
FITNESS_WEIGHT = 0.01
RMSE_STD_WEIGHT = 0.30
TIME_WEIGHT = 0.0002

# Degenerate registration guardrails.
# These are not tuning objectives; they only reject clearly invalid alignments.
MIN_VALID_FITNESS = 0.05
MIN_VALID_CORRESPONDENCES = 50
INVALID_ALIGNMENT_PENALTY = 1.0


def to_builtin(obj):
    """Recursively convert NumPy types to normal Python types."""
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_config(path=BASE_CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path):
    clean_config = to_builtin(config)
    with open(path, "w") as f:
        yaml.safe_dump(clean_config, f, sort_keys=False)


def set_nested(d, keys, value):
    """Set nested dict value using tuple/list of keys."""
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def get_nested(d, keys):
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def build_search_space(base_cfg):
    """
    Wider but still reasonable bounded search space.
    These ranges are expanded for deeper tuning while staying practical.
    """
    return {
        # Point cloud resolution
        ("voxel_size",): [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],

        # Normal estimation
        ("normals", "radius_multiplier"): [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        ("normals", "max_nn"): [20, 30, 40, 50, 75, 100],

        # FPFH feature computation
        ("features", "radius_multiplier"): [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        ("features", "max_nn"): [50, 75, 100, 125, 150, 200],

        # RANSAC global registration
        ("ransac", "distance_multiplier"): [0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        ("ransac", "ransac_n"): [3, 4, 5],
        ("ransac", "edge_length_threshold"): [0.75, 0.8, 0.85, 0.9, 0.95],
        ("ransac", "max_iterations"): [20000, 50000, 100000, 200000, 500000],
        ("ransac", "confidence"): [0.99, 0.995, 0.999, 0.9999],

        # ICP refinement
        ("icp", "distance_threshold"): [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08],
        ("icp", "max_iterations"): [200, 500, 1000, 2000, 5000],

        # Robust kernel
        ("robust_kernel", "k"): [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
    }


def generate_configs(base_cfg, search_space, max_trials=MAX_TRIALS):
    """
    Full Cartesian sweep over all parameters here would be far too slow.
    So we create a bounded practical sweep:
      1) random sampling from the Cartesian product
      2) always include the base config
    """
    keys = list(search_space.keys())
    value_lists = [search_space[k] for k in keys]

    total_combos = math.prod(len(v) for v in value_lists)
    print(f"Total raw combinations in full grid: {total_combos}")
    print(f"Running up to {max_trials} sampled trials")

    # If full grid is small enough, use it all
    if total_combos <= max_trials:
        for combo in itertools.product(*value_lists):
            cfg = copy.deepcopy(base_cfg)
            for k, v in zip(keys, combo):
                set_nested(cfg, k, v)
            yield cfg
        return

    rng = np.random.default_rng(RNG_SEED)
    yielded = set()

    # Always include base config first
    yield copy.deepcopy(base_cfg)

    while len(yielded) < (max_trials - 1):
        cfg = copy.deepcopy(base_cfg)
        signature = []
        for k in keys:
            v = rng.choice(search_space[k])
            set_nested(cfg, k, v)
            signature.append((k, str(v)))
        signature = tuple(signature)
        if signature not in yielded:
            yielded.add(signature)
            yield cfg


def load_misaligned_clouds():
    """
    Load fresh point clouds every trial so transformations do not accumulate.
    Matches your main.py setup.
    """
    demo_icp_pcds = o3d.data.DemoICPPointClouds()

    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    additional_rotation = o3d.geometry.get_rotation_matrix_from_xyz([0.3, 0.5, 0.2])
    additional_translation = np.array([2.0, 1.5, 1.0])

    additional_transform = np.eye(4)
    additional_transform[:3, :3] = additional_rotation
    additional_transform[:3, 3] = additional_translation

    source.transform(additional_transform)

    return source, target


def evaluate_once(cfg, eval_threshold):
    """
    Evaluate one config once by:
      1) injecting it into registration.config
      2) running register(source, target)
      3) applying returned transformation
      4) computing RMSE / fitness / correspondence count
    """
    registration.config = copy.deepcopy(cfg)

    source, target = load_misaligned_clouds()
    start_time = time.perf_counter()
    transformation = registration.register(source, target)
    runtime_sec = time.perf_counter() - start_time

    source.transform(transformation)

    threshold = eval_threshold
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold)

    return {
        "rmse": float(evaluation.inlier_rmse),
        "fitness": float(evaluation.fitness),
        "correspondences": int(len(evaluation.correspondence_set)),
        "runtime_sec": float(runtime_sec),
        "eval_threshold_used": float(threshold),
    }


def evaluate_config(cfg, eval_threshold, runs=EVAL_RUNS_PER_CONFIG):
    """
    Evaluate one config across multiple runs to reduce RANSAC randomness.
    Returns aggregate metrics and a composite ranking score.
    """
    run_metrics = [evaluate_once(cfg, eval_threshold) for _ in range(runs)]

    rmse_vals = np.array([m["rmse"] for m in run_metrics], dtype=float)
    fitness_vals = np.array([m["fitness"] for m in run_metrics], dtype=float)
    corr_vals = np.array([m["correspondences"] for m in run_metrics], dtype=float)
    runtime_vals = np.array([m["runtime_sec"] for m in run_metrics], dtype=float)

    rmse_mean = float(np.mean(rmse_vals))
    fitness_mean = float(np.mean(fitness_vals))
    corr_mean = float(np.mean(corr_vals))
    runtime_mean = float(np.mean(runtime_vals))
    eval_threshold_used = float(np.mean([m["eval_threshold_used"] for m in run_metrics]))

    rmse_std = float(np.std(rmse_vals))
    fitness_penalty = FITNESS_WEIGHT * (1.0 - fitness_mean)
    stability_penalty = RMSE_STD_WEIGHT * rmse_std
    runtime_penalty = TIME_WEIGHT * runtime_mean
    score = rmse_mean + fitness_penalty + stability_penalty + runtime_penalty

    # Open3D returns rmse=0 when no correspondences; treat as invalid registration.
    is_invalid_alignment = (
        corr_mean < MIN_VALID_CORRESPONDENCES
        or fitness_mean <= MIN_VALID_FITNESS
        or (rmse_mean == 0.0 and corr_mean == 0.0)
    )
    invalid_penalty = INVALID_ALIGNMENT_PENALTY if is_invalid_alignment else 0.0
    score += invalid_penalty

    return {
        "rmse": rmse_mean,
        "fitness": fitness_mean,
        "correspondences": int(round(corr_mean)),
        "runtime_sec": runtime_mean,
        "rmse_std": rmse_std,
        "fitness_std": float(np.std(fitness_vals)),
        "runtime_std": float(np.std(runtime_vals)),
        "fitness_penalty": float(fitness_penalty),
        "stability_penalty": float(stability_penalty),
        "runtime_penalty": float(runtime_penalty),
        "invalid_penalty": float(invalid_penalty),
        "is_invalid_alignment": bool(is_invalid_alignment),
        "score": float(score),
        "eval_threshold": float(eval_threshold_used),
    }


def run_sweep():
    base_cfg = load_config(BASE_CONFIG_PATH)
    search_space = build_search_space(base_cfg)
    eval_threshold = (
        base_cfg["icp"]["distance_threshold"]
        if FIXED_EVAL_THRESHOLD is None
        else FIXED_EVAL_THRESHOLD
    )

    best_cfg = None
    best_metrics = None
    best_score = float("inf")

    print(
        "Objective: score = rmse + "
        f"{FITNESS_WEIGHT}*(1-fitness) + "
        f"{RMSE_STD_WEIGHT}*rmse_std + {TIME_WEIGHT}*runtime_sec + invalid_penalty"
    )
    print(f"Evaluation threshold mode: fixed for all trials = {eval_threshold}")
    print(f"Runs per config: {EVAL_RUNS_PER_CONFIG}")

    for i, cfg in enumerate(generate_configs(base_cfg, search_space, max_trials=MAX_TRIALS), start=1):
        try:
            metrics = evaluate_config(cfg, eval_threshold, runs=EVAL_RUNS_PER_CONFIG)

            print(
                f"[{i:04d}] "
                f"RMSE={metrics['rmse']:.6f}, "
                f"Score={metrics['score']:.6f}, "
                f"Fitness={metrics['fitness']:.4f}, "
                f"Corr={metrics['correspondences']}, "
                f"Valid={not metrics['is_invalid_alignment']}"
            )

            if metrics["score"] < best_score:
                best_score = metrics["score"]
                best_cfg = copy.deepcopy(cfg)
                best_metrics = metrics
                print("   --> new best config found")

        except Exception as e:
            print(f"[{i:04d}] FAILED: {e}")

    if best_cfg is None:
        raise RuntimeError("No valid configuration completed successfully.")

    # Save best config
    save_config(best_cfg, BEST_CONFIG_PATH)

    print("\nBest score:", best_score)
    print("Best RMSE:", best_metrics["rmse"])
    print("Best fitness:", best_metrics["fitness"])
    print("Best runtime_sec:", best_metrics["runtime_sec"])
    print("Best correspondences:", best_metrics["correspondences"])
    print(f"Saved best config to {BEST_CONFIG_PATH}")
    return best_cfg, best_metrics


if __name__ == "__main__":
    run_sweep()
