from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import torch

from cugo_ppo_nets50 import make_seeds
from cugo_torch50_ext import load_extension


def require_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        raise RuntimeError(f"{name} mismatch")


def require_outputs_equal(
    name: str,
    actual: tuple[torch.Tensor, ...] | list[torch.Tensor],
    expected: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> None:
    if len(actual) != len(expected):
        raise RuntimeError(f"{name} output count mismatch")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        require_equal(f"{name}[{index}]", left, right)


def first_legal_actions(legal: torch.Tensor) -> torch.Tensor:
    if bool((legal.sum(dim=1) == 0).any().item()):
        raise RuntimeError("expected at least one legal action per row")
    return legal.to(dtype=torch.int64).argmax(dim=1)


def clone_step(ext: Any, states: torch.Tensor, env_ids: torch.Tensor, actions: torch.Tensor):
    children = ext.clone_indexed(states, env_ids)
    outputs = ext.step(children, actions)
    return (children, *outputs)


def benchmark(
    label: str,
    fn: Callable[[], Any],
    batch: int,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    branches_per_second = batch * iters / seconds
    print(
        f"{label} batch={batch} iters={iters} seconds={seconds:.6f} "
        f"branches_per_s={branches_per_second:.3f}"
    )
    return branches_per_second


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if args.parents < 8:
        raise SystemExit("--parents must be at least 8")
    if args.batch < 1 or args.warmup < 0 or args.iters < 1:
        raise SystemExit("invalid benchmark arguments")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable")

    device = torch.device("cuda")
    ext = load_extension(verbose=args.verbose_build)
    seeds, first = make_seeds(args.parents, 0, device)
    parents = ext.create(seeds, first)

    # Move away from identical fresh-deal structure before equivalence checks.
    _features, legal, _players, done, status = ext.observe(parents)
    active_ids = torch.nonzero((~done) & (status == int(ext.STATUS_OK)), as_tuple=False).flatten()
    parents = parents.index_select(0, active_ids).contiguous()
    legal = legal.index_select(0, active_ids)
    ext.step(parents, first_legal_actions(legal))
    _features, _legal, _players, done, status = ext.observe(parents)
    active_ids = torch.nonzero((~done) & (status == int(ext.STATUS_OK)), as_tuple=False).flatten()
    parents = parents.index_select(0, active_ids).contiguous()
    parent_count = parents.size(0)
    if parent_count < 8:
        raise RuntimeError("not enough active parents after one step")
    parent_snapshot = parents.clone()

    clone_ids = torch.tensor([0, 7, 0, 3, 7, 1], dtype=torch.int64, device=device)
    cloned = ext.clone_indexed(parents, clone_ids)
    require_equal("clone_indexed", cloned, parents.index_select(0, clone_ids))
    require_equal("clone parent immutability", parents, parent_snapshot)

    branch_ids = torch.arange(1024, dtype=torch.int64, device=device).mul(37).remainder(parent_count)
    _features, branch_legal, _players, branch_done, branch_status = ext.observe_indexed(
        parents, branch_ids
    )
    if bool(branch_done.any().item()) or bool((branch_status != int(ext.STATUS_OK)).any().item()):
        raise RuntimeError("branch parents are unexpectedly terminal or invalid")
    branch_actions = first_legal_actions(branch_legal)
    branch_actual = tuple(ext.branch_indexed(parents, branch_ids, branch_actions))
    branch_expected = clone_step(ext, parents, branch_ids, branch_actions)
    require_outputs_equal("branch equivalence", branch_actual, branch_expected)
    require_equal("branch parent immutability", parents, parent_snapshot)

    _features, all_legal, _players, _done, _status = ext.observe(parents)
    legal_counts = all_legal.sum(dim=1)
    multi_parent = int(legal_counts.argmax().item())
    choices = torch.nonzero(all_legal[multi_parent], as_tuple=False).flatten()[:4].to(dtype=torch.int64)
    if choices.numel() < 2:
        raise RuntimeError("expected at least two legal actions for multi-branch validation")
    repeated_ids = torch.full((choices.numel(),), multi_parent, dtype=torch.int64, device=device)
    multi_actual = tuple(ext.branch_indexed(parents, repeated_ids, choices))
    multi_expected = clone_step(ext, parents, repeated_ids, choices)
    require_outputs_equal("same-parent multi-action branch", multi_actual, multi_expected)
    if bool((multi_actual[4] != int(ext.STATUS_OK)).any().item()):
        raise RuntimeError("legal same-parent branches returned a non-OK status")
    require_equal("multi-branch parent immutability", parents, parent_snapshot)

    invalid_ids = torch.tensor([-1, parent_count], dtype=torch.int64, device=device)
    invalid_id_actions = torch.zeros(2, dtype=torch.int64, device=device)
    invalid_id_actual = tuple(ext.branch_indexed(parents, invalid_ids, invalid_id_actions))
    indexed_reference_states = parents.clone()
    invalid_id_expected = tuple(ext.step_indexed(indexed_reference_states, invalid_ids, invalid_id_actions))
    require_outputs_equal("invalid index status", invalid_id_actual[1:], invalid_id_expected)
    if bool(invalid_id_actual[0].any().item()):
        raise RuntimeError("invalid parent ids must produce zero-filled child rows")
    require_equal("invalid-index parent immutability", parents, parent_snapshot)

    invalid_action_ids = torch.tensor([0, 1], dtype=torch.int64, device=device)
    invalid_actions = torch.tensor([-1, int(ext.ACTION_COUNT)], dtype=torch.int64, device=device)
    invalid_action_actual = tuple(ext.branch_indexed(parents, invalid_action_ids, invalid_actions))
    invalid_action_expected = clone_step(ext, parents, invalid_action_ids, invalid_actions)
    require_outputs_equal("invalid action status", invalid_action_actual, invalid_action_expected)
    require_equal("invalid-action parent immutability", parents, parent_snapshot)

    empty_ids = torch.empty(0, dtype=torch.int64, device=device)
    empty_actions = torch.empty(0, dtype=torch.int64, device=device)
    empty_clone = ext.clone_indexed(parents, empty_ids)
    if tuple(empty_clone.shape) != (0, int(ext.STATE_BYTES)):
        raise RuntimeError(f"unexpected empty clone shape: {tuple(empty_clone.shape)}")
    empty_branch = tuple(ext.branch_indexed(parents, empty_ids, empty_actions))
    if tuple(empty_branch[0].shape) != (0, int(ext.STATE_BYTES)):
        raise RuntimeError(f"unexpected empty branch state shape: {tuple(empty_branch[0].shape)}")
    if any(tensor.numel() != 0 for tensor in empty_branch[1:]):
        raise RuntimeError("empty branch metadata outputs must be empty")

    print(
        "correctness PASS "
        f"state_bytes={int(ext.STATE_BYTES)} parents={parent_count} "
        f"equivalence_rows={branch_ids.numel()} multi_actions={choices.numel()}"
    )

    _features, parent_legal, _players, parent_done, parent_status = ext.observe(parents)
    if bool(parent_done.any().item()) or bool((parent_status != int(ext.STATUS_OK)).any().item()):
        raise RuntimeError("benchmark parents are unexpectedly terminal or invalid")
    parent_actions = first_legal_actions(parent_legal)
    benchmark_ids = torch.arange(args.batch, dtype=torch.int64, device=device).remainder(parent_count)
    benchmark_actions = parent_actions.index_select(0, benchmark_ids)

    fused_rate = benchmark(
        "branch_indexed",
        lambda: ext.branch_indexed(parents, benchmark_ids, benchmark_actions),
        args.batch,
        args.warmup,
        args.iters,
    )
    split_rate = benchmark(
        "clone_indexed+step",
        lambda: clone_step(ext, parents, benchmark_ids, benchmark_actions),
        args.batch,
        args.warmup,
        args.iters,
    )
    require_equal("benchmark parent immutability", parents, parent_snapshot)
    print(
        f"benchmark PASS fused_over_split={fused_rate / split_rate:.4f}x "
        f"fused_branches_per_s={fused_rate:.3f} split_branches_per_s={split_rate:.3f}"
    )


if __name__ == "__main__":
    main()
