"""Shared helpers for repertoire diagram generators."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

AUTO_SIDE_SAMPLE_MAX_PLY = 40


class RepertoireSide(Enum):
    WHITE = "white"
    BLACK = "black"


@dataclass
class BranchingStats:
    branch_total: int = 0
    branching_nodes: int = 0

    def add(self, child_count: int) -> None:
        self.branch_total += child_count
        if child_count > 1:
            self.branching_nodes += 1


ChildGetter = Callable[[Any], Iterable[Any]]


def collect_branching_stats(root: Any, child_getter: ChildGetter) -> dict[int, BranchingStats]:
    stats = {0: BranchingStats(), 1: BranchingStats()}
    stack = [root]
    while stack:
        node = stack.pop()
        ply = getattr(node, "ply", 0)
        children = tuple(child_getter(node))
        if 0 < ply <= AUTO_SIDE_SAMPLE_MAX_PLY:
            stats[ply % 2].add(len(children))
        stack.extend(children)
    return stats


def decide_side_from_stats(stats: dict[int, BranchingStats]) -> RepertoireSide:
    white_stats = stats[0]
    black_stats = stats[1]
    white_branching_nodes = white_stats.branching_nodes
    black_branching_nodes = black_stats.branching_nodes

    if white_branching_nodes > black_branching_nodes:
        return RepertoireSide.BLACK
    if black_branching_nodes > white_branching_nodes:
        return RepertoireSide.WHITE
    return RepertoireSide.WHITE

def format_single_move(ply: int, move: str, use_black_ellipsis: bool = True) -> str:
    number = (ply + 1) // 2
    if ply % 2 == 1:
        prefix = f"{number}."
    else:
        prefix = f"{number}..." if use_black_ellipsis else f"{number}."
    return f"{prefix} {move}"


def format_sequence(sequence: Sequence[str]) -> str:
    if not sequence:
        return "(start)"
    parts: list[str] = []
    move_number = 1
    idx = 0
    while idx < len(sequence):
        white_move = sequence[idx]
        segment = f"{move_number}. {white_move}"
        idx += 1
        if idx < len(sequence):
            segment += f" {sequence[idx]}"
            idx += 1
        parts.append(segment)
        move_number += 1
    return " ".join(parts)


def mermaid_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
    )


def safe_relative(path: Path, base: Path | None) -> str:
    if base is not None:
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            return str(path)
    return str(path)


def format_path_label(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if not parts:
        return rel_path
    lines = [f"{segment}/" for segment in parts[:-1]]
    lines.append(parts[-1])
    return "<br/>".join(lines)


def requires_startpos_root(node: Any) -> bool:
    children = getattr(node, "children", None)
    if children is None:
        return False
    try:
        return len(children) > 1
    except TypeError:  # pragma: no cover - non-sized iterables
        return True
