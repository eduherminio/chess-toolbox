# Chess Toolbox – AI Guide
- Repo ships two PGN-to-Mermaid generators: Python CLI scripts under `scripts/` and Pyodide web front-ends under `/pgn-tree` and `/repertoire-overview`.
- `scripts/common.py` centralizes move formatting, automatic side detection, Unicode piece toggles, and path labeling; reuse it instead of reimplementing helpers.
- `python-chess` (declared in `pyproject.toml`) is the single dependency; both CLI and browser runtimes load it, so stay within its PGN/Board APIs.

## Runtime & Workflows
- Use Python 3.12+ with `uv`; e.g. `uv run scripts/repertoire_overview.py --pgn-root sample_pgns/edge_cases` or `uv run scripts/pgn_tree.py --pgn-root sample_pgns/edge_cases --piece-symbols`.
- Serve the repo root with `python -m http.server 8000` (or similar) when tweaking the UIs so `fetch("../scripts/*.py")` and ES module imports resolve, then open `http://localhost:8000`.
- No automated tests exist; validate changes by rerunning the relevant CLI commands and reloading the browser tools.

## CLI Generators
- `repertoire_overview.py` walks entire PGN folders, infers sections per starting FEN/tag, builds flowcharts plus “Terminal references” tables, and writes markdown near the PGNs (default `PGN_variations_overview.md`).
- `pgn_tree.py` renders every variation of one PGN, annotating move quality (NAGs) and position evals, and emits `<name>_diagram.md` next to the input file.
- Auto-orientation logic samples the first `AUTO_SIDE_SAMPLE_MAX_PLY` (40) plies via `collect_branching_stats`/`decide_side_from_stats`; update both scripts if those heuristics change.
- Shared helpers such as `format_path_label`, `format_sequence`, `requires_startpos_root`, and `PIECE_SYMBOLS` should be updated in `common.py` so both tools and the UIs pick up the change.

## Web Front-Ends
- `pgn-tree/index.html` and `repertoire-overview/index.html` bootstrap Pyodide, run `micropip.install("chess")`, and fetch the Python sources verbatim; any Python change instantly affects the browser tools.
- JS orchestrates uploads (`get_pgn_from_io` vs. folder staging under `/tmp/repertoire_uploads`), calls `parse_pgn` or `main(argv=…)`, and feeds Mermaid definitions back into `<pre class="mermaid">` nodes.
- Theme toggles set `data-theme` and CSS variables from `tools-shared.css`; keep new UI elements using those tokens so dark/light rendering stays consistent with download backgrounds.
- Download buttons reuse `lastDiagramSvg`/`lastMarkdown`, while helpers like `diagramDefinitions`, `rerenderDiagramForTheme`, and `parseOverviewMarkdown` assume Mermaid blocks are fenced with ```mermaid```; preserve that structure when emitting markdown.

## Data & Conventions
- Sample repertoires live under `sample_pgns/` (imported in docs and demos); update or duplicate them when introducing new PGN headers or edge cases.
- Markdown legends/tables expect HTML-safe strings; always run SAN strings through `mermaid_escape` and `format_sequence` before embedding.
- Mermaid snippets must stay compact enough for GitHub Pages; keep IDs stable (`N0`, `N1`, …) and avoid inline HTML beyond `<br/>` provided by `format_path_label`.
- Entry points must stay guarded by `if __name__ == "__main__": raise SystemExit(main())` so Pyodide can `import` modules without triggering CLI behavior.

## Environment
- Use `python3` whenever running scripts locally to validate data.