# Chess Toolbox

Collection of chess-related tools and helpers.
- Web version available in [eduherminio.github.io/chess-toolbox](https://eduherminio.github.io/chess-toolbox).
- CLI version available in [/scripts](/scripts).


>ℹ️ Disclaimer: _Some of these tools have been generated with AI help, since I'm not a Python expert or someone who enjoys web development._
>_They've been partially 'vibe-coded', so don't treat them as production-ready unless you're willing to 'vibe-debug' them._

## Repertoire

Tools that are meant to be run with `--pgn-root <dir\>`, where <dir\> is where a folder where you're storing your PGNs

### repertoire-overview.py

Creates one or multiple diagrams that show an overview of what your repertoire looks like under `--root-dir`: main lines and files where they're stored.

```bash
uv run scripts/repertoire_overview.py --help
usage: repertoire_overview.py [-h] [--pgn-root PGN_ROOT] [--output OUTPUT] [--max-ply MAX_PLY] [--side {auto,white,black}] [--piece-symbols]

Generate a Mermaid flowchart by inspecting a tree of PGN files.

options:
  -h, --help            show this help message and exit
  --pgn-root PGN_ROOT   Required. Directory that will be scanned recursively for .pgn files.
  --output OUTPUT       Destination markdown file. Defaults to <pgn-root>/PGN_variations_overview.md.
  --max-ply MAX_PLY     Base half-move limit per PGN (branches may extend further to reach a unique source; 0 = ilimitado).
  --side {auto,white,black} Orientation for the repertoire. Auto mode infers the side from branching heuristics.
  --piece-symbols       Render moves with Unicode chess piece symbols instead of SAN letters.
```

### pgn-tre.py

Creates a detailed diagram for each pgn under `--root-dir`, showing all variations and highlighting the quality of moves and positions.

```bash
uv run scripts/pgn_tree.py --help
usage: pgn_tree.py [-h] [--pgn-root PGN_ROOT] [--side {auto,white,black}] [--piece-symbols] [--output OUTPUT]

Generate a Mermaid diagram per PGN file, including all embedded variations.

options:
  -h, --help            show this help message and exit
  --pgn-root PGN_ROOT   Required. Directory scanned recursively for .pgn files or a single .pgn file.
  --side {auto,white,black} Orientation for the repertoire. Auto mode infers the side from branching heuristics.
  --piece-symbols       Render moves with Unicode chess piece symbols instead of SAN letters.
  --output OUTPUT       Directory where markdown files will be written. If a filename is provided, only the name overrides the default but files stay next to each PGN.
```

### Sample usage

```bash
uv run scripts/repertoire-overview.py --pgn-root repertoire
uv run scripts/repertoire-overview.py --pgn-root repertoire/White_e4 --output white-e4-rep-overview.md
uv run scripts/pgn-tree.py --pgn-root repertoire --piece-symbols

python3 -m http.server 8000
```

Given the following folder structure:

```
repertoire
└── White_e4
|   └── Italian
│   │   │   Two_Knights_Defense.pgn
│   │   │   Giuoco_Piano.pgn
│   │   │
│   │   └── Other
│   │       │   Hungarian_Defense.pgn
│   │       │   Paris_Defense.pgn
│   │       
|   └── Sicilian
│       │   Modern.pgn
│       │   O_Kelly.pgn
│       │   French.pgn
│       │   Hyperaccelerated_Dragon.pgn
│   
├── Queens_Gambit_Black
│   ├── Slav.pgn
│   └── SemiSlav.pgn
│   │
│   └── Queens_Gambit_Declined
│       │   Queens_Knight_Variation.pgn
│       │   Other.pgn
```
