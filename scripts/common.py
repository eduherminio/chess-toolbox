"""Shared helpers for repertoire diagram generators."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

AUTO_SIDE_SAMPLE_MAX_PLY = 40

PIECE_SYMBOLS = str.maketrans(
    {
        "K": "♔",
        "Q": "♕",
        "R": "♖",
        "B": "♗",
        "N": "♘",
    }
)


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

def format_single_move(
    ply: int,
    move: str,
    use_black_ellipsis: bool = True,
    use_piece_symbols: bool = False,
) -> str:
    number = (ply + 1) // 2
    if ply % 2 == 1:
        prefix = f"{number}."
    else:
        prefix = f"{number}..." if use_black_ellipsis else f"{number}."
    rendered_move = move.translate(PIECE_SYMBOLS) if use_piece_symbols else move
    return f"{prefix} {rendered_move}"


def format_sequence(sequence: Sequence[str], use_piece_symbols: bool = False) -> str:
    if not sequence:
        return "(start)"
    parts: list[str] = []
    move_number = 1
    idx = 0
    while idx < len(sequence):
        white_move = (
            sequence[idx].translate(PIECE_SYMBOLS)
            if use_piece_symbols
            else sequence[idx]
        )
        segment = f"{move_number}. {white_move}"
        idx += 1
        if idx < len(sequence):
            black_move = (
                sequence[idx].translate(PIECE_SYMBOLS)
                if use_piece_symbols
                else sequence[idx]
            )
            segment += f" {black_move}"
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
    meta_label = ""
    base_path = rel_path
    if "::" in rel_path:
        base_path, meta_label = rel_path.split("::", 1)
    base_path = base_path or rel_path
    parts = Path(base_path).parts or (base_path,)
    lines = [f"{segment}/" for segment in parts[:-1]]
    lines.append(parts[-1])
    meta_label = meta_label.strip()
    if meta_label:
        lines.append(meta_label)
    return "<br/>".join(lines)


def requires_startpos_root(node: Any) -> bool:
    children = getattr(node, "children", None)
    if children is None:
        return False
    try:
        return len(children) > 1
    except TypeError:  # pragma: no cover - non-sized iterables
        return True


_SLUG_REGEX = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    normalized = value.strip().lower()
    slug = _SLUG_REGEX.sub("_", normalized)
    return slug.strip("_")


def describe_game_from_headers(headers: Mapping[str, str], fallback_index: int) -> tuple[str, str]:
    event = headers.get("Event", "").strip()
    opening = headers.get("Opening", "").strip()

    title = ""
    if event:
        title = event
    elif opening:
        title = opening

    if not title:
        title = f"Game {fallback_index}"

    slug_source = title or f"game_{fallback_index}"
    slug = slugify(slug_source)
    if not slug:
        slug = f"game_{fallback_index}"
    return title, slug
