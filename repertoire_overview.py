#!/usr/bin/env python3
"""Generate a Mermaid flowchart by inspecting a tree of PGN files."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from common import (
    AUTO_SIDE_SAMPLE_MAX_PLY,
    BranchingStats,
    RepertoireSide,
    decide_side_from_stats,
    format_path_label,
    format_sequence,
    format_single_move,
    mermaid_escape,
    requires_startpos_root,
    safe_relative,
)

try:  # pragma: no cover - dependency guard
    import chess.pgn
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "python-chess is required. Install it with 'python -m pip install chess'."
    ) from exc


class OpeningNode:
    """Tree node for the accumulated move sequences."""

    __slots__ = ("move", "ply", "sequence", "children", "sources")

    def __init__(self, move: str | None = None, ply: int = 0, sequence: tuple[str, ...] = ()) -> None:
        self.move = move
        self.ply = ply
        self.sequence = sequence
        self.children: dict[str, OpeningNode] = {}
        self.sources: set[str] = set()


@dataclass(frozen=True)
class DiagramNode:
    id: str
    label: str
    sequence: tuple[str, ...]
    sources: Sequence[str]
    kind: str = "move"


@dataclass(frozen=True)
class DiagramEdge:
    source: str
    target: str
    label: str


@dataclass
class RepertoireBucket:
    root_label: str
    root: OpeningNode
    sources: set[str]


@dataclass
class DiagramSection:
    title: str
    side: RepertoireSide
    nodes: Sequence[DiagramNode]
    edges: Sequence[DiagramEdge]
    sources: Sequence[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgn-root",
        type=Path,
        default=None,
        help="Required. Directory that will be scanned recursively for .pgn files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination markdown file. Defaults to <pgn-root>/PGN_variations_overview.md.",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=4,
        help="Base half-move limit per PGN (branches may extend further to reach a unique source; 0 = ilimitado).",
    )
    parser.add_argument(
        "--side",
        choices=("auto", RepertoireSide.WHITE.value, RepertoireSide.BLACK.value),
        default="auto",
        help="Orientation for the repertoire. Auto mode infers the side from branching heuristics.",
    )
    parser.add_argument(
        "--piece-symbols",
        action="store_true",
        help="Render moves with Unicode chess piece symbols instead of SAN letters.",
    )
    return parser.parse_args(argv)


def collect_branching_stats_from_game(game: chess.pgn.Game) -> dict[int, BranchingStats]:
    stats = {0: BranchingStats(), 1: BranchingStats()}

    def walk(node: chess.pgn.GameNode, ply: int) -> None:
        variations = node.variations
        if 0 < ply <= AUTO_SIDE_SAMPLE_MAX_PLY:
            stats[ply % 2].add(len(variations))
        for variation in variations:
            walk(variation, ply + 1)

    walk(game, 0)
    return stats


def read_main_line(path: Path) -> tuple[list[str], str | None, dict[str, str], dict[int, BranchingStats]]:
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        return [], None, {}, {0: BranchingStats(), 1: BranchingStats()}
    board = game.board()
    moves: list[str] = []
    for move in game.mainline_moves():
        san = board.san(move)
        moves.append(san)
        board.push(move)
    stats = collect_branching_stats_from_game(game)
    return moves, game.headers.get("FEN"), dict(game.headers), stats


def add_line(root: OpeningNode, moves: Iterable[str], source: str) -> None:
    root.sources.add(source)
    node = root
    for ply, move in enumerate(moves, start=1):
        child = node.children.get(move)
        if child is None:
            child = OpeningNode(move=move, ply=ply, sequence=node.sequence + (move,))
            node.children[move] = child
        child.sources.add(source)
        node = child


def ensure_bucket(
    buckets: dict[tuple[str, RepertoireSide], RepertoireBucket],
    root_key: str,
    side: RepertoireSide,
    root_label: str,
) -> RepertoireBucket:
    bucket = buckets.get((root_key, side))
    if bucket is None:
        bucket = RepertoireBucket(root_label=root_label, root=OpeningNode(), sources=set())
        buckets[(root_key, side)] = bucket
    return bucket


def infer_section_label(headers: dict[str, str], default_label: str) -> str:
    tag = headers.get("RepertoireGroup") or headers.get("Group")
    if tag:
        candidate = tag.strip()
        if candidate:
            return candidate
    return default_label


def build_diagram(
    root: OpeningNode,
    side: RepertoireSide,
    max_ply: int | None,
    root_label: str | None = None,
    use_piece_symbols: bool = False,
) -> tuple[list[DiagramNode], list[DiagramEdge]]:
    visible_parity = 1 if side == RepertoireSide.WHITE else 0
    opponent_parity = 1 - visible_parity
    include_root = bool(root_label)
    nodes: list[DiagramNode] = []
    edges: list[DiagramEdge] = []
    node_ids: dict[OpeningNode, str] = {}
    node_counter = 0

    def next_node_id() -> str:
        nonlocal node_counter
        node_id = f"N{node_counter}"
        node_counter += 1
        return node_id

    def should_expand_children(node: OpeningNode) -> bool:
        if not node.children:
            return False
        if max_ply is None:
            return True
        if node.ply < max_ply:
            return True
        return len(node.sources) > 1

    def ensure_node(node: OpeningNode) -> str:
        node_id = node_ids.get(node)
        if node_id:
            return node_id
        node_id = next_node_id()
        node_ids[node] = node_id
        if node.ply == 0:
            base_label = root_label or "Initial position"
        else:
            base_label = format_single_move(
                node.ply,
                node.move or "",
                use_piece_symbols=use_piece_symbols,
            )
        count = len(node.sources)
        label = f"{base_label} ({count} PGN)" if count > 1 else base_label
        nodes.append(
            DiagramNode(
                id=node_id,
                label=label,
                sequence=node.sequence,
                sources=tuple(sorted(node.sources)) if node.sources else (),
                kind="move",
            )
        )
        return node_id

    def attach_path_block(node: OpeningNode, source_id: str) -> None:
        if not node.sources:
            return
        path_label = sorted(node.sources)[0]
        file_label = format_path_label(path_label)
        path_node_id = next_node_id()
        nodes.append(
            DiagramNode(
                id=path_node_id,
                label=file_label,
                sequence=node.sequence,
                sources=(path_label,),
                kind="path",
            )
        )
        edges.append(DiagramEdge(source=source_id, target=path_node_id, label=""))

    def connect_from_opponent(source_node: OpeningNode, source_id: str, force: bool = False) -> None:
        if not force and not should_expand_children(source_node):
            attach_path_block(source_node, source_id)
            return
        for move in sorted(source_node.children):
            opponent_node = source_node.children[move]
            if opponent_node.ply % 2 != opponent_parity:
                continue
            if not opponent_node.children:
                attach_path_block(source_node, source_id)
                continue
            edge_label = format_single_move(
                opponent_node.ply,
                opponent_node.move or "",
                use_piece_symbols=use_piece_symbols,
            )
            branch_connected = False
            for reply in sorted(opponent_node.children):
                reply_node = opponent_node.children[reply]
                if reply_node.ply % 2 != visible_parity:
                    continue
                target_id = ensure_node(reply_node)
                edges.append(DiagramEdge(source=source_id, target=target_id, label=edge_label))
                visit_visible(reply_node)
                branch_connected = True
            if not branch_connected:
                attach_path_block(source_node, source_id)

    def visit_visible(node: OpeningNode) -> None:
        source_id = ensure_node(node)
        if not should_expand_children(node):
            attach_path_block(node, source_id)
            return
        connect_from_opponent(node, source_id)

    root_id = ensure_node(root) if include_root else None
    if include_root and root_id:
        if visible_parity == 1:
            for move in sorted(root.children):
                child = root.children[move]
                if child.ply % 2 != visible_parity:
                    continue
                target_id = ensure_node(child)
                edges.append(DiagramEdge(source=root_id, target=target_id, label=""))
                visit_visible(child)
        else:
            connect_from_opponent(root, root_id, force=True)
    else:
        if visible_parity == 1:
            for move in sorted(root.children):
                child = root.children[move]
                if child.ply % 2 != visible_parity:
                    continue
                visit_visible(child)
        else:
            for move in sorted(root.children):
                opponent_node = root.children[move]
                if opponent_node.ply % 2 != opponent_parity:
                    continue
                for reply in sorted(opponent_node.children):
                    reply_node = opponent_node.children[reply]
                    if reply_node.ply % 2 != visible_parity:
                        continue
                    visit_visible(reply_node)

    return nodes, edges

def render_mermaid(nodes: Sequence[DiagramNode], edges: Sequence[DiagramEdge]) -> str:
    lines = ["flowchart TD"]
    path_style_entries: list[str] = []
    for node in nodes:
        label = mermaid_escape(node.label)
        lines.append(f"    {node.id}[\"{label}\"]")
        if node.kind == "path":
            path_style_entries.append(
                f"    style {node.id} fill:#0f172a,stroke:#38bdf8,stroke-width:3px,color:#e0f2fe"
            )
    lines.extend(path_style_entries)
    for edge in edges:
        label = mermaid_escape(edge.label)
        if label:
            lines.append(f"    {edge.source} -->|{label}| {edge.target}")
        else:
            lines.append(f"    {edge.source} --> {edge.target}")
    return "\n".join(lines)


def render_table(nodes: Sequence[DiagramNode], use_piece_symbols: bool = False) -> str:
    rows = ["| Node | File | Sequence |", "| --- | --- | --- |"]
    for node in nodes:
        if node.kind != "path":
            continue
        sequence = format_sequence(node.sequence, use_piece_symbols=use_piece_symbols)
        if node.sources:
            file_paths = "<br/>".join(node.sources)
        else:
            file_paths = "-"
        rows.append(f"| {node.id} | {file_paths} | {sequence} |")
    return "\n".join(rows)


def build_markdown(
    title: str,
    sections: Sequence[DiagramSection],
    use_piece_symbols: bool = False,
) -> str:
    parts = [
        f"# {title}",
        "Automatically generated overview of the main repertoire lines.",
        "",
    ]
    for section in sections:
        heading = section.title
        parts.append(f"## {heading}")
        parts.extend([
            "",
            "```mermaid",
            render_mermaid(section.nodes, section.edges),
            "```",
            "",
            "### Terminal references",
            render_table(section.nodes, use_piece_symbols=use_piece_symbols),
            "",
        ])
    parts.append("> Source: scripts/generate_main_variations_multipgn.py")
    return "\n".join(parts)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pgn_root is None:
        print("--pgn-root must be provided", file=sys.stderr)
        return 1
    pgn_root = args.pgn_root
    if not pgn_root.exists():
        print(f"PGN directory does not exist: {pgn_root}", file=sys.stderr)
        return 1

    if args.output is None:
        output_path = pgn_root / "PGN_variations_overview.md"
    else:
        output_candidate = args.output
        if output_candidate.exists() and output_candidate.is_dir():
            output_path = output_candidate / "PGN_variations_overview.md"
        elif output_candidate.suffix:
            output_path = output_candidate
        else:
            output_path = output_candidate / "PGN_variations_overview.md"
    max_ply = None if args.max_ply <= 0 else args.max_ply
    title = f"Repertoire diagram ({pgn_root.name})"

    pgn_files = sorted(pgn_root.rglob("*.pgn"))
    if not pgn_files:
        print(f"No PGN files were found in {pgn_root}", file=sys.stderr)
        return 1

    buckets: dict[tuple[str, RepertoireSide], RepertoireBucket] = {}
    root_labels: dict[str, str] = {}
    root_origins: dict[str, str] = {}

    for path in pgn_files:
        moves, fen, headers, branching_stats = read_main_line(path)
        if not moves:
            continue
        rel = safe_relative(path, pgn_root)
        root_key = fen if fen else "startpos"
        root_label = f"FEN: {fen}" if fen else "Starting position"
        root_labels.setdefault(root_key, root_label)
        root_origins.setdefault(root_key, "starting position" if root_key == "startpos" else root_key)
        section_label = infer_section_label(headers, root_labels[root_key])

        if args.side == "auto":
            target_side = decide_side_from_stats(branching_stats)
        else:
            target_side = RepertoireSide(args.side)

        bucket = ensure_bucket(buckets, root_key, target_side, root_labels[root_key])
        add_line(bucket.root, moves, rel)
        bucket.sources.add(rel)

    sections: list[DiagramSection] = []
    if args.side == "auto":
        side_order = (RepertoireSide.WHITE, RepertoireSide.BLACK)
    else:
        forced_side = RepertoireSide(args.side)
        side_order = (forced_side,)

    root_order = sorted(root_labels.items(), key=lambda item: item[1].casefold())

    for root_key, stored_root_label in root_order:
        for side in side_order:
            bucket = buckets.get((root_key, side))
            if bucket is None:
                continue
            if not bucket.root.children:
                continue
            if root_key == "startpos":
                effective_root_label = (
                    "Starting position"
                    if requires_startpos_root(bucket.root) or side == RepertoireSide.BLACK
                    else None
                )
            else:
                effective_root_label = stored_root_label

            nodes, edges = build_diagram(
                bucket.root,
                side,
                max_ply,
                root_label=effective_root_label,
                use_piece_symbols=args.piece_symbols,
            )
            if len(nodes) <= 1 and not edges:
                continue
            origin_label = "starting position" if root_key == "startpos" else root_origins[root_key]
            section_title = f"{side.value.capitalize()} repertoire from {origin_label}"
            sections.append(
                DiagramSection(
                    title=section_title,
                    side=side,
                    nodes=nodes,
                    edges=edges,
                    sources=sorted(bucket.sources),
                )
            )

    if not sections:
        print("No diagrams could be generated from the provided PGN files.", file=sys.stderr)
        return 1

    markdown = build_markdown(title, sections, use_piece_symbols=args.piece_symbols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
