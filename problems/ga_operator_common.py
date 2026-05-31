from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable


def _repo_root(root_dir: str) -> Path:
    env_root = os.getenv("COCOGA_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(root_dir).resolve().parents[1]


def _add_cocoga_src(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _baseline_crossover(parent1: list[int], parent2: list[int], rng, ctx: dict[str, Any]) -> list[int]:
    n = len(parent1)
    if n <= 2:
        return parent1[:]
    a, b = sorted(rng.choice(n, size=2, replace=False).astype(int).tolist())
    child = [-1] * n
    child[a:b] = parent1[a:b]
    used = set(child[a:b])
    fill = [x for x in parent2 if x not in used]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[j]
            j += 1
    return child


def _baseline_mutate(route: list[int], rng, ctx: dict[str, Any]) -> list[int]:
    n = len(route)
    if n <= 2:
        return route[:]
    child = route[:]
    i, j = sorted(rng.choice(n, size=2, replace=False).astype(int).tolist())
    child[i], child[j] = child[j], child[i]
    return child


def _load_module(code_path: Path):
    spec = importlib.util.spec_from_file_location(f"reevo_generated_{code_path.parent.name}", code_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generated operator module: {code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_callable(module: Any, names: list[str]) -> Callable | None:
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _pair_from_factory(factory: Callable) -> tuple[Callable, Callable]:
    value = factory()
    if isinstance(value, dict):
        cx = value.get("crossover") or value.get("crossover_fn")
        mut = value.get("mutate") or value.get("mutation") or value.get("mutation_fn")
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        cx, mut = value[0], value[1]
    else:
        raise RuntimeError("ga_operators must return {'crossover': fn, 'mutate': fn} or (crossover, mutate)")
    if not callable(cx) or not callable(mut):
        raise RuntimeError("Generated ga_operators returned non-callable crossover/mutation")
    return cx, mut


def _load_crossover_from_path(code_path: Path) -> Callable:
    module = _load_module(code_path)
    factory = _first_callable(module, ["ga_operators_v2", "ga_operators"])
    if factory is not None:
        cx, _ = _pair_from_factory(factory)
        return cx
    cx = _first_callable(module, ["crossover_v2", "crossover"])
    if not callable(cx):
        raise RuntimeError(f"Fixed crossover file does not define crossover_v2/crossover: {code_path}")
    return cx


def _load_mutation_from_path(code_path: Path) -> Callable:
    module = _load_module(code_path)
    factory = _first_callable(module, ["ga_operators_v2", "ga_operators"])
    if factory is not None:
        _, mut = _pair_from_factory(factory)
        return mut
    mut = _first_callable(module, ["mutate_v2", "mutation_v2", "mutate", "mutation"])
    if not callable(mut):
        raise RuntimeError(f"Fixed mutation file does not define mutate_v2/mutate: {code_path}")
    return mut


def load_generated_pair(problem_name: str, root_dir: str, mode: str) -> tuple[Callable, Callable]:
    code_path = Path(root_dir) / "problems" / problem_name / "gpt.py"
    if not code_path.exists():
        code_path = Path(__file__).resolve().parent / problem_name / "gpt.py"
    module = _load_module(code_path)

    if mode == "pair":
        factory = _first_callable(module, ["ga_operators_v2", "ga_operators"])
        if factory is not None:
            return _pair_from_factory(factory)
        cx = _first_callable(module, ["crossover_v2", "crossover"])
        mut = _first_callable(module, ["mutate_v2", "mutation_v2", "mutate", "mutation"])
        if callable(cx) and callable(mut):
            return cx, mut
        raise RuntimeError("Pair mode requires ga_operators_v2() or both crossover/mutate functions")

    if mode == "crossover":
        cx = _first_callable(module, ["crossover_v2", "crossover"])
        if not callable(cx):
            raise RuntimeError("Crossover mode requires crossover_v2(parent1, parent2, rng, ctx)")
        fixed_mutation = os.getenv("COCOGA_REEVO_FIXED_MUTATION", "").strip()
        mut = _load_mutation_from_path(Path(fixed_mutation)) if fixed_mutation else _baseline_mutate
        return cx, mut

    if mode == "mutation":
        mut = _first_callable(module, ["mutate_v2", "mutation_v2", "mutate", "mutation"])
        if not callable(mut):
            raise RuntimeError("Mutation mode requires mutate_v2(route, rng, ctx)")
        fixed_crossover = os.getenv("COCOGA_REEVO_FIXED_CROSSOVER", "").strip()
        cx = _load_crossover_from_path(Path(fixed_crossover)) if fixed_crossover else _baseline_crossover
        return cx, mut

    raise ValueError(f"Unknown GA operator eval mode: {mode}")


def _run_eval(problem_name: str, mode: str, root_dir: str, mood: str) -> float:
    repo_root = _repo_root(root_dir)
    _add_cocoga_src(repo_root)

    from cocoga.core.operator_interface import LoadedOperator, OperatorKind, OperatorMeta
    from cocoga.cocoga.experiment_setup import load_experiment_setup
    from cocoga.eval.pair_eval import evaluate_pair_best

    config_dir = repo_root / os.getenv("COCOGA_CONFIG_DIR", "configs")
    problem_id = os.getenv("COCOGA_PROBLEM_ID", "tsp")
    setup = load_experiment_setup(config_dir, problem_id)
    instances = setup.train_instances if mood == "train" else (setup.test_instances or setup.train_instances)

    cx_fn, mut_fn = load_generated_pair(problem_name, root_dir, mode)
    cx_op = LoadedOperator(
        meta=OperatorMeta(
            operator_id=f"reevo_{problem_name}_cx",
            kind=OperatorKind.CROSSOVER,
            encoding_type="permutation",
            problem_scope=problem_id,
        ),
        fn=cx_fn,
    )
    mut_op = LoadedOperator(
        meta=OperatorMeta(
            operator_id=f"reevo_{problem_name}_mut",
            kind=OperatorKind.MUTATION,
            encoding_type="permutation",
            problem_scope=problem_id,
        ),
        fn=mut_fn,
    )

    n_runs = max(1, int(setup.config.ga.n_runs))
    seed_base = int(os.getenv("COCOGA_LLH_SEED_BASE", "0"))
    run_seeds = [seed_base + i for i in range(n_runs)]
    fitness, result = evaluate_pair_best(
        cx_op,
        mut_op,
        instances,
        setup.config,
        seed=seed_base,
        run_seeds=run_seeds,
    )

    print(
        "[*] CoCo-GA taskGA eval: "
        f"problem={problem_id}, reevo_problem={problem_name}, mode={mode}, "
        f"split={mood}, instances={[inst.name for inst in instances]}, run_seeds={run_seeds}"
    )
    if mode == "crossover":
        print(f"[*] Fixed mutation partner: {os.getenv('COCOGA_REEVO_FIXED_MUTATION') or 'baseline_mutate'}")
    elif mode == "mutation":
        print(f"[*] Fixed crossover partner: {os.getenv('COCOGA_REEVO_FIXED_CROSSOVER') or 'baseline_crossover'}")
    for item in getattr(result, "instance_fitnesses", []) or []:
        print(
            "[*] Instance "
            f"{item.get('instance')}: best_distance={item.get('best_distance')} "
            f"gap_percent={item.get('best_gap_percent')}"
        )
    if not math.isfinite(float(fitness)):
        return float("inf")
    return float(fitness)


def main(problem_name: str, mode: str) -> None:
    print("[*] Running ...")
    if len(sys.argv) < 4:
        raise SystemExit("usage: eval.py <problem_size> <root_dir> <train|val>")
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    if mood not in {"train", "val"}:
        raise ValueError(f"Unknown eval split: {mood}")
    objective = _run_eval(problem_name, mode, root_dir, mood)
    print("[*] Average:")
    print(objective)
