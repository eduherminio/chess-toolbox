#!/usr/bin/env python3
"""Generate a Mermaid diagram per PGN file, including all embedded variations."""
from __future__ import annotations

import argparse
import io
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from common import (
    RepertoireSide,
    collect_branching_stats,
    decide_side_from_stats,
    format_single_move,
    mermaid_escape,
    requires_startpos_root,
)

try:  # python-chess handles PGN parsing with nested variations
    import chess.pgn
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "python-chess is required. Install it with 'python -m pip install chess'."
    ) from exc


@dataclass
class TreeNode:
    san: str | None
    ply: int
    sequence: tuple[str, ...]
    fen_key: str
    nags: frozenset[int] = field(default_factory=frozenset)
    children: list["TreeNode"] = field(default_factory=list)
    max_path_len: int = 0


class MoveQuality(Enum):
    NEUTRAL = "neutral"
    BRILLIANT = "brilliant"
    GOOD = "good"
    INTERESTING = "interesting"
    FORCED = "forced"
    DUBIOUS = "dubious"
    BAD = "bad"


class PositionEvaluation(Enum):
    NEUTRAL = "neutral"
    WHITE_WINNING = "white-winning"
    WHITE_ADVANTAGE = "white-advantage"
    BALANCED = "balanced"
    UNCLEAR = "unclear"
    BLACK_ADVANTAGE = "black-advantage"
    BLACK_WINNING = "black-winning"


@dataclass(frozen=True)
class GlyphAnnotation:
    move_quality: MoveQuality
    position_eval: PositionEvaluation


@dataclass(frozen=True)
class DiagramNode:
    id: str
    label: str
    sequence: tuple[str, ...]
    glyphs: GlyphAnnotation


@dataclass(frozen=True)
class DiagramEdge:
    source: str
    target: str
    label: str | None = None
    move_quality: MoveQuality = MoveQuality.NEUTRAL


MOVE_QUALITY_PRIORITY: Sequence[tuple[MoveQuality, frozenset[int]]] = (
    (MoveQuality.BRILLIANT, frozenset({chess.pgn.NAG_BRILLIANT_MOVE})),
    (MoveQuality.GOOD, frozenset({chess.pgn.NAG_GOOD_MOVE})),
    (
        MoveQuality.INTERESTING,
        frozenset({chess.pgn.NAG_SPECULATIVE_MOVE, chess.pgn.NAG_NOVELTY}),
    ),
    (
        MoveQuality.FORCED,
        frozenset({chess.pgn.NAG_FORCED_MOVE, chess.pgn.NAG_SINGULAR_MOVE}),
    ),
    (MoveQuality.DUBIOUS, frozenset({chess.pgn.NAG_DUBIOUS_MOVE})),
    (
        MoveQuality.BAD,
        frozenset({chess.pgn.NAG_BLUNDER, chess.pgn.NAG_MISTAKE, chess.pgn.NAG_WORST_MOVE}),
    ),
)


POSITION_EVAL_PRIORITY: Sequence[tuple[PositionEvaluation, frozenset[int]]] = (
    (
        PositionEvaluation.WHITE_WINNING,
        frozenset({chess.pgn.NAG_WHITE_DECISIVE_ADVANTAGE, chess.pgn.NAG_WHITE_DECISIVE_COUNTERPLAY}),
    ),
    (
        PositionEvaluation.WHITE_ADVANTAGE,
        frozenset(
            {
                chess.pgn.NAG_WHITE_MODERATE_ADVANTAGE,
                chess.pgn.NAG_WHITE_SLIGHT_ADVANTAGE,
            }
        ),
    ),
    (
        PositionEvaluation.BLACK_WINNING,
        frozenset({chess.pgn.NAG_BLACK_DECISIVE_ADVANTAGE, chess.pgn.NAG_BLACK_DECISIVE_COUNTERPLAY}),
    ),
    (
        PositionEvaluation.BLACK_ADVANTAGE,
        frozenset(
            {
                chess.pgn.NAG_BLACK_MODERATE_ADVANTAGE,
                chess.pgn.NAG_BLACK_SLIGHT_ADVANTAGE,
            }
        ),
    ),
    (
        PositionEvaluation.BALANCED,
        frozenset({chess.pgn.NAG_DRAWISH_POSITION, chess.pgn.NAG_QUIET_POSITION}),
    ),
    (
        PositionEvaluation.UNCLEAR,
        frozenset(
            {
                chess.pgn.NAG_UNCLEAR_POSITION,
                chess.pgn.NAG_ACTIVE_POSITION,
                chess.pgn.NAG_WHITE_MODERATE_COUNTERPLAY,
                chess.pgn.NAG_BLACK_MODERATE_COUNTERPLAY,
            }
        ),
    ),
)


def classify_move_quality(nags: frozenset[int]) -> MoveQuality:
    for quality, markers in MOVE_QUALITY_PRIORITY:
        if nags & markers:
            return quality
    return MoveQuality.NEUTRAL


def classify_position_eval(nags: frozenset[int]) -> PositionEvaluation:
    for evaluation, markers in POSITION_EVAL_PRIORITY:
        if nags & markers:
            return evaluation
    return PositionEvaluation.NEUTRAL


def interpret_glyphs(nags: frozenset[int]) -> GlyphAnnotation:
    return GlyphAnnotation(
        move_quality=classify_move_quality(nags),
        position_eval=classify_position_eval(nags),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgn-root",
        type=Path,
        default=None,
        help="Required. Directory scanned recursively for .pgn files or a single .pgn file.",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory where markdown files will be written. If a filename is provided, "
            "only the name overrides the default but files stay next to each PGN."
        ),
    )
    return parser.parse_args()


def node_label_text(node: TreeNode, root_label: str | None, piece_symbols: bool) -> str:
    if node.ply == 0:
        return root_label or "Initial position"
    return format_single_move(node.ply, node.san or "", use_piece_symbols=piece_symbols)


def canonical_fen(board: chess.Board) -> str:
    parts = board.fen().split(" ")
    return " ".join(parts[:3])


def build_tree(game: chess.pgn.Game) -> TreeNode:
    board = game.board()
    root = TreeNode(san=None, ply=0, sequence=(), fen_key=canonical_fen(board))

    def dfs(chess_node: chess.pgn.GameNode, tree_node: TreeNode, board_state: chess.Board) -> None:
        for variation in chess_node.variations:
            move = variation.move
            san = board_state.san(move)
            board_state.push(move)
            child = TreeNode(
                san=san,
                ply=tree_node.ply + 1,
                sequence=tree_node.sequence + (san,),
                fen_key=canonical_fen(board_state),
                nags=frozenset(variation.nags),
            )
            tree_node.children.append(child)
            dfs(variation, child, board_state)
            board_state.pop()

    dfs(game, root, board.copy())
    return root


def annotate_longest_paths(node: TreeNode) -> int:
    if not node.children:
        node.max_path_len = node.ply
        return node.max_path_len

    max_len = node.ply
    for child in node.children:
        child_len = annotate_longest_paths(child)
        if child_len > max_len:
            max_len = child_len
    node.max_path_len = max_len
    return node.max_path_len


def select_canonical_nodes(root: TreeNode) -> dict[str, TreeNode]:
    canonical: dict[str, TreeNode] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        best = canonical.get(node.fen_key)
        if best is None or node.max_path_len > best.max_path_len or (
            node.max_path_len == best.max_path_len and node.sequence < best.sequence
        ):
            canonical[node.fen_key] = node
        stack.extend(node.children)
    return canonical


def build_adjacency(root: TreeNode) -> dict[str, OrderedDict[str, str]]:
    adjacency: dict[str, OrderedDict[str, str]] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        order = adjacency.setdefault(node.fen_key, OrderedDict())
        for child in node.children:
            move = child.san or ""
            if move not in order:
                order[move] = child.fen_key
            stack.append(child)
    return adjacency


def tree_to_diagram(
    root: TreeNode,
    side: RepertoireSide,
    root_label: str | None = None,
    piece_symbols: bool = False,
) -> tuple[list[DiagramNode], list[DiagramEdge]]:
    annotate_longest_paths(root)
    canonical = select_canonical_nodes(root)
    adjacency = build_adjacency(root)

    visible_parity = 1 if side == RepertoireSide.WHITE else 0
    opponent_parity = 1 - visible_parity
    include_root = bool(root_label)

    def is_visible(node: TreeNode) -> bool:
        return node.ply == 0 or node.ply % 2 == visible_parity

    nodes: list[DiagramNode] = []
    edges: list[DiagramEdge] = []
    fen_to_id: dict[str, str] = {}

    def ensure_node(fen: str) -> str | None:
        if fen in fen_to_id:
            return fen_to_id[fen]
        node = canonical[fen]
        if not is_visible(node):
            return None
        node_id = f"N{len(fen_to_id)}"
        fen_to_id[fen] = node_id
        label = node_label_text(node, root_label, piece_symbols=piece_symbols)
        glyphs = interpret_glyphs(node.nags)
        nodes.append(DiagramNode(id=node_id, label=label, sequence=node.sequence, glyphs=glyphs))
        return node_id

    root_id = ensure_node(root.fen_key) if include_root else None

    def connect_via_opponent_moves(source_id: str, move_map: OrderedDict[str, str]) -> None:
        for move_san, opponent_fen in move_map.items():
            opponent_node = canonical.get(opponent_fen)
            if opponent_node is None or opponent_node.ply % 2 != opponent_parity:
                continue
            opponent_children = adjacency.get(opponent_fen, OrderedDict())
            if not opponent_children:
                continue
            edge_label = format_single_move(opponent_node.ply, move_san, use_piece_symbols=piece_symbols)
            edge_quality = classify_move_quality(opponent_node.nags)
            for _reply_san, reply_fen in opponent_children.items():
                target_id = ensure_node(reply_fen)
                if target_id:
                    edges.append(
                        DiagramEdge(
                            source=source_id,
                            target=target_id,
                            label=edge_label,
                            move_quality=edge_quality,
                        )
                    )

    for fen, moves in adjacency.items():
        node = canonical[fen]
        if node.ply == 0:
            if include_root and root_id:
                if visible_parity == 1:
                    for _move_san, child_fen in moves.items():
                        target_id = ensure_node(child_fen)
                        if target_id:
                            edges.append(
                                DiagramEdge(
                                    source=root_id,
                                    target=target_id,
                                    move_quality=MoveQuality.NEUTRAL,
                                )
                            )
                else:
                    connect_via_opponent_moves(root_id, moves)
            continue

        if node.ply % 2 != visible_parity:
            continue  # skip nodes that do not belong to the repertoire side

        source_id = ensure_node(fen)
        if not source_id:
            continue

        connect_via_opponent_moves(source_id, moves)

    return nodes, edges


FRIENDLY_WIN_STYLE = {"fill": "#0ea5e9", "stroke": "#0369a1", "color": "#082f49"}
FRIENDLY_PLUS_STYLE = {"fill": "#22c55e", "stroke": "#15803d", "color": "#052e16"}
HOSTILE_STYLE = {"fill": "#dc2626", "stroke": "#991b1b", "color": "#ffffff"}
BALANCED_STYLE = {"fill": "#e2e8f0", "stroke": "#475569", "color": "#0f172a"}
UNCLEAR_STYLE = {"fill": "#fde047", "stroke": "#ca8a04", "color": "#0f172a"}

POSITION_STYLE_SCHEMES: dict[RepertoireSide, dict[PositionEvaluation, dict[str, str]]] = {
    RepertoireSide.WHITE: {
        PositionEvaluation.WHITE_WINNING: FRIENDLY_WIN_STYLE,
        PositionEvaluation.WHITE_ADVANTAGE: FRIENDLY_PLUS_STYLE,
        PositionEvaluation.BLACK_ADVANTAGE: HOSTILE_STYLE,
        PositionEvaluation.BLACK_WINNING: HOSTILE_STYLE,
        PositionEvaluation.BALANCED: BALANCED_STYLE,
        PositionEvaluation.UNCLEAR: UNCLEAR_STYLE,
        PositionEvaluation.NEUTRAL: BALANCED_STYLE,
    },
    RepertoireSide.BLACK: {
        PositionEvaluation.BLACK_WINNING: FRIENDLY_WIN_STYLE,
        PositionEvaluation.BLACK_ADVANTAGE: FRIENDLY_PLUS_STYLE,
        PositionEvaluation.WHITE_ADVANTAGE: HOSTILE_STYLE,
        PositionEvaluation.WHITE_WINNING: HOSTILE_STYLE,
        PositionEvaluation.BALANCED: BALANCED_STYLE,
        PositionEvaluation.UNCLEAR: UNCLEAR_STYLE,
        PositionEvaluation.NEUTRAL: BALANCED_STYLE,
    },
}

POSITION_LEGEND_ORDER: dict[RepertoireSide, tuple[PositionEvaluation, ...]] = {
    RepertoireSide.WHITE: (
        PositionEvaluation.WHITE_WINNING,
        PositionEvaluation.WHITE_ADVANTAGE,
        PositionEvaluation.BALANCED,
        PositionEvaluation.UNCLEAR,
        PositionEvaluation.BLACK_ADVANTAGE,
        PositionEvaluation.BLACK_WINNING,
    ),
    RepertoireSide.BLACK: (
        PositionEvaluation.BLACK_WINNING,
        PositionEvaluation.BLACK_ADVANTAGE,
        PositionEvaluation.BALANCED,
        PositionEvaluation.UNCLEAR,
        PositionEvaluation.WHITE_ADVANTAGE,
        PositionEvaluation.WHITE_WINNING,
    ),
}


def position_style_for(evaluation: PositionEvaluation, side: RepertoireSide) -> dict[str, str]:
    return POSITION_STYLE_SCHEMES[side].get(evaluation, BALANCED_STYLE)

POSITION_DESCRIPTIONS: dict[PositionEvaluation, str] = {
    PositionEvaluation.WHITE_WINNING: "White has a decisive advantage",
    PositionEvaluation.WHITE_ADVANTAGE: "White is clearly better",
    PositionEvaluation.BALANCED: "Balanced or quiet position",
    PositionEvaluation.UNCLEAR: "Unclear / highly dynamic position",
    PositionEvaluation.BLACK_ADVANTAGE: "Black is clearly better",
    PositionEvaluation.BLACK_WINNING: "Black has a decisive advantage",
}

MOVE_QUALITY_COLORS: dict[MoveQuality, str] = {
    MoveQuality.BRILLIANT: "#16a34a",
    MoveQuality.GOOD: "#16a34a",
    MoveQuality.INTERESTING: "#0ea5e9",
    MoveQuality.FORCED: "#a855f7",
    MoveQuality.DUBIOUS: "#f97316",
    MoveQuality.BAD: "#dc2626",
}

MOVE_QUALITY_DESCRIPTIONS: dict[MoveQuality, str] = {
    MoveQuality.BRILLIANT: "Brilliant move (!!)",
    MoveQuality.GOOD: "Strong move (!)",
    MoveQuality.INTERESTING: "Interesting / speculative move (!?)",
    MoveQuality.FORCED: "Forced or only move",
    MoveQuality.DUBIOUS: "Dubious move (?!)",
    MoveQuality.BAD: "Mistake or blunder (? / ??)",
}

MOVE_STROKE_STYLES: dict[MoveQuality, dict[str, str]] = {
    quality: {"stroke": color, "stroke-width": "3px"}
    for quality, color in MOVE_QUALITY_COLORS.items()
}

EDGE_STYLE_BY_QUALITY: dict[MoveQuality, str] = {
    quality: f"stroke:{color},stroke-width:2px,color:{color}"
    for quality, color in MOVE_QUALITY_COLORS.items()
}

DEFAULT_NODE_STYLE = {"fill": "#ffffff", "stroke": "#1f2937", "color": "#111827"}


def format_style(style: dict[str, str]) -> str:
    return ",".join(f"{key}:{value}" for key, value in style.items())


def render_mermaid(nodes: Sequence[DiagramNode], edges: Sequence[DiagramEdge], side: RepertoireSide) -> str:
    lines = ["flowchart TD"]
    node_styles: list[tuple[str, str]] = []

    for node in nodes:
        label = mermaid_escape(node.label)
        lines.append(f"    {node.id}[\"{label}\"]")
        style = dict(DEFAULT_NODE_STYLE)
        style.update(position_style_for(node.glyphs.position_eval, side))
        style.update(MOVE_STROKE_STYLES.get(node.glyphs.move_quality, {}))
        node_styles.append((node.id, format_style(style)))

    link_styles: list[tuple[int, str]] = []
    for idx, edge in enumerate(edges):
        if edge.label:
            label = mermaid_escape(edge.label)
            lines.append(f"    {edge.source} -->|{label}| {edge.target}")
        else:
            lines.append(f"    {edge.source} --> {edge.target}")
        style = EDGE_STYLE_BY_QUALITY.get(edge.move_quality)
        if style:
            link_styles.append((idx, style))

    for node_id, style in node_styles:
        lines.append(f"    style {node_id} {style}")

    for edge_idx, style in link_styles:
        lines.append(f"    linkStyle {edge_idx} {style}")

    return "\n".join(lines)
def color_chip(fill: str, border: str | None = "#1f2937") -> str:
    border_css = f";border:1px solid {border}" if border else ""
    return (
        f"<span style=\"display:inline-block;width:1.2em;height:1.2em;"
        f"background-color:{fill}{border_css};border-radius:2px;margin-right:0.4em;\"></span>"
    )


def render_color_legend(side: RepertoireSide) -> str:
    lines: list[str] = ["## Color legend", "", "**Node evaluation (fill colors)**", ""]
    lines.append("| Color | Meaning |")
    lines.append("| --- | --- |")
    chip = color_chip(DEFAULT_NODE_STYLE["fill"], DEFAULT_NODE_STYLE["stroke"])
    lines.append(f"| {chip} | No highlighted evaluation |")
    for evaluation in POSITION_LEGEND_ORDER[side]:
        style = position_style_for(evaluation, side)
        desc = POSITION_DESCRIPTIONS[evaluation]
        chip = color_chip(style["fill"], style["stroke"])
        lines.append(f"| {chip} | {desc} |")

    lines.extend([
        "",
        "**Move quality (borders and arrows)**",
        "",
        "| Color | Meaning |",
        "| --- | --- |",
    ])
    move_entries: list[tuple[str, str]] = [
        (
            MOVE_QUALITY_COLORS[MoveQuality.GOOD],
            "Brilliant (!!) or strong (!) move",
        ),
        (
            MOVE_QUALITY_COLORS[MoveQuality.INTERESTING],
            MOVE_QUALITY_DESCRIPTIONS[MoveQuality.INTERESTING],
        ),
        (
            MOVE_QUALITY_COLORS[MoveQuality.FORCED],
            MOVE_QUALITY_DESCRIPTIONS[MoveQuality.FORCED],
        ),
        (
            MOVE_QUALITY_COLORS[MoveQuality.DUBIOUS],
            MOVE_QUALITY_DESCRIPTIONS[MoveQuality.DUBIOUS],
        ),
        (
            MOVE_QUALITY_COLORS[MoveQuality.BAD],
            MOVE_QUALITY_DESCRIPTIONS[MoveQuality.BAD],
        ),
    ]
    for fill, desc in move_entries:
        chip = color_chip(fill, border=None)
        lines.append(f"| {chip} | {desc} |")
    lines.append(f"| {color_chip('#1f2937', border=None)} | Move without annotations |")
    return "\n".join(lines)


def render_diagram_and_legend(
    nodes: Sequence[DiagramNode],
    edges: Sequence[DiagramEdge],
    side: RepertoireSide,
) -> tuple[str, str]:
    diagram = render_mermaid(nodes, edges, side)
    legend = render_color_legend(side)
    return diagram, legend


def build_markdown(
    title: str,
    diagram: str,
    legend: str,
    pgn_name: str,
    side: RepertoireSide,
    origin_label: str,
) -> str:
    section_heading = (
        f"## {pgn_name} - {side.value.capitalize()} repertoire from {origin_label}"
    )
    parts = [
        f"# {title}",
        f"Diagram with every recorded variation from `{pgn_name}`.",
        "",
        section_heading,
        "",
        "```mermaid",
        diagram,
        "```",
        "",
        legend,
        "",
        "> Source: eduherminio/chess-toolbox",
    ]
    return "\n".join(parts)


def diagram_artifacts_from_game(
    game: chess.pgn.Game,
    side_preference: str,
    piece_symbols: bool = False,
) -> tuple[list[DiagramNode], list[DiagramEdge], RepertoireSide, str]:
    tree = build_tree(game)
    if side_preference == "auto":
        stats = collect_branching_stats(tree, lambda node: node.children)
        side = decide_side_from_stats(stats)
    else:
        side = RepertoireSide(side_preference)

    fen_header = game.headers.get("FEN")
    if fen_header:
        root_label = f"FEN: {fen_header}"
        origin_label = fen_header
    elif requires_startpos_root(tree) or side == RepertoireSide.BLACK:
        root_label = "Starting position"
        origin_label = "starting position"
    else:
        root_label = None
        origin_label = "starting position"

    nodes, edges = tree_to_diagram(tree, side, root_label, piece_symbols=piece_symbols)
    return nodes, edges, side, origin_label


def get_pgn_from_file(path: Path) -> chess.pgn.Game | None:
    with path.open(encoding="utf-8") as handle:
        return chess.pgn.read_game(handle)


def get_pgn_from_io(pgn_text: str) -> chess.pgn.Game | None:
    return chess.pgn.read_game(io.StringIO(pgn_text))


def parse_pgn(
    game: chess.pgn.Game | None,
    side_preference: str = "auto",
    piece_symbols: bool = False,
) -> tuple[str, str, RepertoireSide, str]:
    if game is None:
        return ("flowchart TD\nA[Invalid PGN]", "", RepertoireSide.WHITE, "invalid PGN")
    nodes, edges, side, origin_label = diagram_artifacts_from_game(
        game,
        side_preference,
        piece_symbols=piece_symbols,
    )
    diagram, legend = render_diagram_and_legend(nodes, edges, side)
    return diagram, legend, side, origin_label


def generate_markdown(
    path: Path,
    side_preference: str,
    output_dir: Path | None,
    output_name_override: str | None,
    piece_symbols: bool = False,
) -> bool:
    game = get_pgn_from_file(path)
    if game is None:
        print(f"[WARN] Could not read any game in {path}", file=sys.stderr)
        return False

    diagram, legend, side, origin_label = parse_pgn(
        game,
        side_preference,
        piece_symbols=piece_symbols,
    )
    markdown = build_markdown(
        f"Full diagram: {path.stem}",
        diagram,
        legend,
        path.name,
        side,
        origin_label,
    )
    default_filename = f"{path.stem}_diagram.md"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / default_filename
    elif output_name_override is not None:
        output_path = path.with_name(output_name_override)
    else:
        output_path = path.with_name(default_filename)
    output_path.write_text(markdown, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    if args.pgn_root is None:
        print("--pgn-root must be provided", file=sys.stderr)
        return 1
    pgn_root = args.pgn_root
    if not pgn_root.exists():
        print(f"Path does not exist: {pgn_root}", file=sys.stderr)
        return 1
    output_dir: Path | None = None
    output_name_override: str | None = None
    if args.output is not None:
        if args.output.exists() and args.output.is_dir():
            output_dir = args.output
        elif args.output.suffix:
            output_name_override = args.output.name
        else:
            output_dir = args.output

    if pgn_root.is_file():
        if pgn_root.suffix.lower() != ".pgn":
            print(f"File does not have a .pgn extension: {pgn_root}", file=sys.stderr)
            return 1
        pgn_files = [pgn_root]
    else:
        pgn_files = sorted(pgn_root.rglob("*.pgn"))
    if not pgn_files:
        print(f"No PGN files were found in {pgn_root}", file=sys.stderr)
        return 1

    ok = True
    for path in pgn_files:
        ok &= generate_markdown(
            path,
            args.side,
            output_dir,
            output_name_override,
            piece_symbols=args.piece_symbols,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
