import math
from os import path
import numpy as np
import sys
import argparse
import logging
from copy import copy
import json
import os
import re
from pathlib import Path

try:
    from gpt import select_next_node_v2 as select_next_node
except:
    from gpt import select_next_node

_ORIGINAL_DEFAULT_RNG = np.random.default_rng


def _euclidean_distance_matrix(node_positions: np.ndarray) -> np.ndarray:
    diff = node_positions[:, None, :] - node_positions[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def eval_heuristic_from_distance_matrix(dist_mat: np.ndarray, start_node: int = 0) -> tuple[float, list[int]]:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.

    Parameters
    ----------
    dist_mat : np.ndarray
        Distance matrix of shape (problem_size, problem_size).

    Returns
    -------
    obj : float
        The length of the generated tour.
    solution : list[int]
        Constructed TSP tour.
    '''
    problem_size = dist_mat.shape[0]
    start_node = int(start_node) % problem_size
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")

    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj, solution


def _set_eval_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng = lambda seed_arg=None: _ORIGINAL_DEFAULT_RNG(seed if seed_arg is None else seed_arg)


def eval_repeated_heuristic_from_distance_matrix(
    dist_mat: np.ndarray,
    seeds: list[int],
) -> tuple[float, list[int], int]:
    """Run one deterministic eval or several stochastic evals and keep the best tour."""
    best_obj = float("inf")
    best_solution: list[int] = []
    eval_runs = 0
    for seed in seeds:
        _set_eval_seed(seed)
        obj, solution = eval_heuristic_from_distance_matrix(dist_mat, start_node=0)
        eval_runs += 1
        if obj < best_obj:
            best_obj = float(obj)
            best_solution = solution
    return best_obj, best_solution, eval_runs


def eval_heuristic(node_positions: np.ndarray) -> float:
    dist_mat = _euclidean_distance_matrix(node_positions)
    obj, _ = eval_heuristic_from_distance_matrix(dist_mat)
    return obj


def _cocoga_root(root_dir: str) -> Path:
    env_root = os.getenv("COCOGA_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # ReEvo lives under <repo>/orther_LHH_code/reevo.
    return Path(root_dir).resolve().parents[1]


def _read_yaml(path_obj: Path) -> dict:
    import yaml

    with path_obj.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path_obj}")
    return data


def _split_instance_names(base_cfg: dict, split: str) -> list[str]:
    if split == "train":
        return list(base_cfg.get("train_instances") or base_cfg.get("default_instances") or [])
    if split in {"val", "test"}:
        names = list(base_cfg.get("test_instances") or [])
        return names or list(base_cfg.get("train_instances") or base_cfg.get("default_instances") or [])
    raise ValueError(f"Unknown split: {split}")


def _resolve_instance_path(repo_root: Path, data_dir: str, instance_name: str) -> Path:
    candidate = Path(instance_name)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".tsp")
    if not candidate.is_absolute():
        candidate = repo_root / data_dir / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"TSP instance not found: {candidate}")
    return candidate.resolve()


def _positive_int_env(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _taskga_n_runs(config_dir: Path) -> int:
    override = _positive_int_env("COCOGA_LLH_N_RUNS")
    if override is not None:
        return override
    ga_cfg = _read_yaml(config_dir / "ga.yaml")
    n_runs = int(ga_cfg.get("n_runs") or 1)
    if n_runs <= 0:
        raise ValueError(f"configs/ga.yaml n_runs must be positive, got {n_runs}")
    return n_runs


def _generated_code_uses_random(root_dir: str) -> tuple[bool, list[str]]:
    gpt_path = Path(root_dir) / "problems" / "tsp_constructive" / "gpt.py"
    if not gpt_path.exists():
        gpt_path = Path(__file__).with_name("gpt.py")
    code = gpt_path.read_text(encoding="utf-8")
    patterns = [
        r"\bimport\s+random\b",
        r"\bfrom\s+random\s+import\b",
        r"\brandom\.",
        r"\bnp\.random\b",
        r"\bnumpy\.random\b",
        r"\bdefault_rng\b",
        r"\brandint\b",
        r"\brandrange\b",
        r"\bshuffle\s*\(",
        r"\bsample\s*\(",
        r"\bchoices?\s*\(",
        r"\buniform\s*\(",
        r"\bnormal\s*\(",
        r"\bseed\s*\(",
        r"\bsecrets\b",
        r"\bos\.urandom\b",
        r"\buuid\b",
    ]
    matches = [pattern for pattern in patterns if re.search(pattern, code)]
    return bool(matches), matches


def _evaluation_seeds(root_dir: str, config_dir: Path) -> tuple[list[int], bool, list[str]]:
    stochastic, matches = _generated_code_uses_random(root_dir)
    if not stochastic:
        return [0], False, matches
    n_runs = _taskga_n_runs(config_dir)
    seed_base = int(os.getenv("COCOGA_LLH_SEED_BASE", "0"))
    return [seed_base + i for i in range(n_runs)], True, matches


def _run_cocoga_tsplib_eval(root_dir: str, mood: str) -> bool:
    if os.getenv("COCOGA_LLH_EVAL_MODE", "").strip().lower() != "tsplib":
        return False

    repo_root = _cocoga_root(root_dir)
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from cocoga.problems.tsp.instance import TSPInstance

    config_dir = repo_root / os.getenv("COCOGA_CONFIG_DIR", "configs")
    base_cfg = _read_yaml(config_dir / "base.yaml")
    data_dir = str(base_cfg.get("data_dir", "data/tsp/instances/tsplib"))
    split = "train" if mood == "train" else os.getenv("COCOGA_LLH_TEST_SPLIT", "test")
    instance_names = _split_instance_names(base_cfg, split)
    if not instance_names:
        raise RuntimeError(f"No instances configured for split={split} in {config_dir / 'base.yaml'}")

    seeds, stochastic, random_matches = _evaluation_seeds(root_dir, config_dir)
    print(
        "[*] CoCo-GA fair TSPLIB eval mode: "
        f"split={split}, instances={instance_names}, "
        f"stochastic={stochastic}, eval_runs_per_instance={len(seeds)}, seeds={seeds}"
    )
    if random_matches:
        print(f"[*] Random-reference patterns: {random_matches}")
    gaps = []
    distances = []
    total_eval_runs = 0
    for name in instance_names:
        inst = TSPInstance.from_tsplib(_resolve_instance_path(repo_root, data_dir, name))
        obj, solution, eval_runs = eval_repeated_heuristic_from_distance_matrix(
            inst.dist_matrix,
            seeds=seeds,
        )
        total_eval_runs += eval_runs
        distances.append(float(obj))
        if inst.best_known:
            gap = (float(obj) - float(inst.best_known)) / float(inst.best_known) * 100.0
            gaps.append(gap)
            print(f"[*] Instance {inst.name}: distance={obj}, gap_percent={gap}, eval_runs={eval_runs}")
        else:
            print(f"[*] Instance {inst.name}: distance={obj}, gap_percent=null, eval_runs={eval_runs}")
        if len(solution) != inst.n or len(set(solution)) != inst.n:
            raise ValueError(f"Constructed invalid tour for {inst.name}")

    print(f"[*] Total eval runs:")
    print(int(total_eval_runs))
    print("[*] Average:")
    if gaps:
        print(float(np.mean(gaps)))
    else:
        print(float(np.mean(distances)))
    return True
    

if __name__ == '__main__':
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    if _run_cocoga_tsplib_eval(root_dir, mood):
        raise SystemExit(0)

    basepath = path.join(path.dirname(__file__), "dataset")
    if not path.isfile(path.join(basepath, "train50_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = path.join(basepath, f"train{problem_size}_dataset.npy")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        for i in range(n_instances):
            obj = eval_heuristic(node_positions[i])
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)
        
        print("[*] Average:")
        print(np.mean(objs))
    
    else:
        for problem_size in [20, 50, 100, 200]:
            dataset_path = path.join(basepath, f"val{problem_size}_dataset.npy")
            logging.info(f"[*] Evaluating {dataset_path}")
            node_positions = np.load(dataset_path)
            n_instances = node_positions.shape[0]
            objs = []
            for i in range(n_instances):
                obj = eval_heuristic(node_positions[i])
                objs.append(obj)
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")
