from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import itertools
import threading
import typer
import typer.rich_utils as _ru
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn  # Progress/SpinnerColumn used by timeline
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree
from rich import box


_ru.STYLE_COMMANDS_TABLE_FIRST_COLUMN = "#a8d4b5"   # light sage — command names
_ru.STYLE_OPTION = "#88c9a0"                        # soft sage — long flags
_ru.STYLE_SWITCH = "#88c9a0"
_ru.STYLE_NEGATIVE_OPTION = "#8bbfa8"               # muted seafoam — negative flags
_ru.STYLE_NEGATIVE_SWITCH = "#8bbfa8"
_ru.STYLE_METAVAR = "#8bbfa8"                       # muted seafoam — type hints
_ru.STYLE_OPTION_DEFAULT = "#8bbfa8"
_ru.STYLE_OPTION_ENVVAR = "#8bbfa8"
_ru.STYLE_OPTIONS_PANEL_BORDER = "#4a7a5e"          # muted forest border
_ru.STYLE_COMMANDS_PANEL_BORDER = "#4a7a5e"


app = typer.Typer(
    help=(
        "[bold #4ade80]granum[/bold #4ade80]  persistent semantic memory for Claude Code\n\n"
        "Granum stores decisions, preferences, constraints and file state as vector chunks "
        "that Claude retrieves automatically at the start of each turn."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# Brand colors
ORANGE = "#88c9a0"   # soft sage green — primary accent
GREEN  = "#6abf85"   # medium sage — success
AMBER  = "#d4b483"   # soft amber — warnings
RED    = "#d47f7f"   # soft rose — errors
GRAY   = "#4a7a5e"   # muted forest — deprecated/de-emphasized
BODY   = "#d4ead9"   # very light mint — body text
MUTED  = "#a8d4b5"   # light sage — secondary text

TYPE_ICONS = {
    "decision": "◆",
    "constraint": "▲",
    "preference": "★",
    "file_state": "▪",
    "spec": "◇",
}
TYPE_COLORS = {
    "decision":   ORANGE,    # soft sage
    "constraint": AMBER,     # soft amber
    "preference": "#a8d4b5", # light sage
    "file_state": "#8bbfa8", # muted seafoam
    "spec":       GRAY,      # muted forest
}
STATUS_ICONS = {"active": "✓", "deprecated": "○", "deleted": "×", "superseded": "◇"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_granum_dir(path: Optional[Path] = None) -> Path:
    return (path or Path.cwd()) / ".granum"


def _load_config(granum_dir: Path) -> dict:
    config_path = granum_dir / "config.json"
    if not config_path.exists():
        console.print(f"[{RED}]✗ No .granum/config.json found. Run: granum init[/{RED}]")
        raise typer.Exit(1)
    return json.loads(config_path.read_text())


def _save_config(granum_dir: Path, config: dict) -> None:
    (granum_dir / "config.json").write_text(json.dumps(config, indent=2))


def _make_project_id(git_root: str, branch: str) -> str:
    return hashlib.md5(f"{git_root}:{branch}".encode()).hexdigest()


def _git_root() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=3
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=3
        )
        return result.stdout.strip() if result.returncode == 0 else "main"
    except Exception:
        return "main"


def _get_db(config: dict, granum_dir: Path):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mcp_server.db import GranumDB
    return GranumDB(
        db_path=granum_dir / "kuzu.db",
        ndjson_path=granum_dir / "chunks.ndjson",
        stale_threshold_days=config.get("stale_threshold_days", 7),
    )


def _ipc_call(granum_dir: Path, method: str, params: dict) -> Optional[any]:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mcp_server.ipc import ipc_call
    return ipc_call(granum_dir, method, params)


def _ipc_query(granum_dir: Path, query: str, config: dict) -> Optional[list]:
    raw = _ipc_call(granum_dir, "query_context", {
        "query": query,
        "memory_limit": config.get("memory_retrieval_limit", 7),
        "spec_limit": config.get("spec_retrieval_limit", 3),
    })
    if raw is None:
        return None
    # query_context now returns {chunks, unresolved_conflicts}
    if isinstance(raw, dict):
        return raw.get("chunks", [])
    return raw  # fallback: old list format


def _ipc_chunks(granum_dir: Path, project_id: str, include_deprecated: bool = False) -> Optional[list]:
    return _ipc_call(granum_dir, "list_chunks", {
        "project_id": project_id,
        "include_deprecated": include_deprecated,
    })


def _ipc_edges(granum_dir: Path, chunk_id: str, edge_type: Optional[str] = None, depth: int = 1) -> Optional[list]:
    return _ipc_call(granum_dir, "get_edges", {"chunk_id": chunk_id, "edge_type": edge_type, "depth": depth})


def _ipc_all_edges(granum_dir: Path, project_id: str) -> Optional[list]:
    return _ipc_call(granum_dir, "get_all_edges", {"project_id": project_id})


def _ipc_spec_chunks(granum_dir: Path, project_id: str) -> Optional[list]:
    return _ipc_call(granum_dir, "list_spec_chunks", {"project_id": project_id})


def _ipc_chunk_history(granum_dir: Path, chunk_id: str) -> Optional[list]:
    return _ipc_call(granum_dir, "get_chunk_history", {"chunk_id": chunk_id})


def _get_spec_chunks(granum_dir: Path, project_id: str, config: dict) -> list:
    raw = _ipc_spec_chunks(granum_dir, project_id)
    if raw is not None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mcp_server.models import Chunk
        return [Chunk.from_dict(d) for d in raw]
    return _get_db(config, granum_dir).get_spec_chunks(project_id)


def _get_chunks(granum_dir: Path, project_id: str, config: dict, include_deprecated: bool = False, _status=None):
    """Fetch chunks via IPC if server running, else direct DB load."""
    raw = _ipc_chunks(granum_dir, project_id, include_deprecated)
    if raw is not None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mcp_server.models import Chunk
        return [Chunk.from_dict(d) for d in raw]
    if _status:
        _status.update("Querying memory store (direct)")
    db = _get_db(config, granum_dir)
    return db.get_all_memory_chunks(project_id, include_deprecated=include_deprecated)


def _age_seconds(updated_at: str) -> float:
    try:
        then = datetime.fromisoformat(updated_at)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - then).total_seconds()
    except Exception:
        return 0.0


def _age_str(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    days = int(seconds // 86400)
    return f"{days}d ago"


def _age_days_from_seconds(seconds: float) -> int:
    return int(seconds // 86400)


def _stale_style(age_days: int, stale_threshold: int) -> Optional[str]:
    if age_days > 14:
        return RED
    if age_days > stale_threshold:
        return AMBER
    return None


_COL_DEFS: dict[str, dict] = {
    "id":     {"header": "ID",     "style": MUTED, "width": 9},
    "type":   {"header": "Type",   "width": 16},
    "title":  {"header": "Title",  "min_width": 24, "max_width": 52, "no_wrap": True, "overflow": "ellipsis"},
    "age":    {"header": "Age",    "width": 14},
    "imp":    {"header": "Imp",    "width": 4},
    "status": {"header": "Status", "width": 14},
    "score":  {"header": "Score",  "width": 7},
    "sim":    {"header": "Sim",    "width": 7},
}

_DEFAULT_COLS = ["id", "type", "title", "age", "imp", "status"]


def _make_chunk_table(cols: list[str] | None = None) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {ORANGE}",
        border_style=GRAY, padding=(0, 1), show_edge=False,
    )
    for col in (cols or _DEFAULT_COLS):
        defn = _COL_DEFS[col]
        table.add_column(defn["header"], **{k: v for k, v in defn.items() if k != "header"})
    return table


def _table_panel(table: Table, title: str) -> Panel:
    return Panel(table, title=f"[dim {MUTED}]{title}[/dim {MUTED}]", border_style=GRAY, padding=(0, 1), title_align="left", expand=False)


def _add_chunk_row(table: Table, chunk, stale_threshold: int, cols: list[str] | None = None) -> None:
    cols = cols or _DEFAULT_COLS
    icon = TYPE_ICONS.get(chunk.type, "·")
    secs = _age_seconds(chunk.updated_at)
    age_days = _age_days_from_seconds(secs)
    stale_color = _stale_style(age_days, stale_threshold)

    age_label = _age_str(secs)
    if stale_color:
        badge = " ⚠" if age_days <= 14 else " ⚠⚠"
        age_cell = f"[{stale_color}]{age_label}{badge}[/{stale_color}]"
    else:
        age_cell = f"[{MUTED}]{age_label}[/{MUTED}]"

    deprecated = chunk.status == "deprecated"
    tc = TYPE_COLORS.get(chunk.type, MUTED)
    cells = {
        "id":     f"[{MUTED}]{chunk.id[:8]}[/{MUTED}]",
        "type":   f"[{tc}]{icon} {chunk.type}[/{tc}]",
        "title":  f"[italic {GRAY}]{chunk.title}[/italic {GRAY}]" if deprecated else f"[{BODY}]{chunk.title}[/{BODY}]",
        "age":    age_cell,
        "imp":    f"[{MUTED}]{chunk.importance}[/{MUTED}]",
        "status": f"[{GRAY}]○[/{GRAY}]" if deprecated else f"[{GREEN}]✓[/{GREEN}]",
    }
    table.add_row(*[cells[c] for c in cols])


def _context_panel(memory_count: int = 0, spec_count: int = 0) -> Panel:
    from rich.text import Text
    proj = Path(_git_root() or ".").name
    branch = _git_branch()
    t = Text()
    t.append(proj, style=f"bold {ORANGE}")
    t.append(f"  {branch}", style=f"dim {MUTED}")
    parts = []
    if memory_count:
        parts.append(f"{memory_count} chunk{'s' if memory_count != 1 else ''}")
    if spec_count:
        parts.append(f"{spec_count} spec{'s' if spec_count != 1 else ''}")
    if parts:
        t.append(f"\n{'  ·  '.join(parts)}", style=f"dim {MUTED}")
    return Panel(t, border_style=GRAY, padding=(0, 1), expand=False)


def _kv_table() -> Table:
    t = Table(box=None, show_header=False, padding=(0, 3, 0, 0), show_edge=False)
    t.add_column("key", style=MUTED, width=14, no_wrap=True)
    t.add_column("value")
    return t


def _hdr(*parts: str) -> None:
    """Print graph-style header: proj  branch  ·  parts..."""
    proj = Path(_git_root() or ".").name
    branch = _git_branch()
    suffix = f"  [{MUTED}]·  {'  ·  '.join(parts)}[/{MUTED}]" if parts else ""
    console.print(f"\n  [{ORANGE}]{proj}[/{ORANGE}]  [{MUTED}]{branch}[/{MUTED}]{suffix}\n")


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

@app.command(rich_help_panel="[bold #4ade80]Setup[/bold #4ade80]")
def init(reset: bool = typer.Option(False, "--reset", help="Re-run spec detection")):
    """[bold]Initialize[/bold] Granum in the current project."""
    git_root = _git_root()
    branch = _git_branch()
    cwd = Path.cwd()
    granum_dir = cwd / ".granum"

    if not git_root:
        console.print(f"[{AMBER}]⚠ No git repo found. Using current directory.[/{AMBER}]")
        git_root = str(cwd)

    project_id = _make_project_id(git_root, branch)
    config_path = granum_dir / "config.json"

    existing_config: dict = {}
    if config_path.exists() and not reset:
        existing_config = json.loads(config_path.read_text())

    # Spec detection
    console.print(f"\n[{ORANGE}]Scanning for spec files...[/{ORANGE}]")
    detected: list[str] = []

    known_patterns = [
        "openspec/", "docs/", "AGENTS.md", "CLAUDE.md",
        ".cursorrules", "GEMINI.md",
    ]
    for pattern in known_patterns:
        p = cwd / pattern
        if p.exists():
            rel = pattern
            detected.append(rel)
            console.print(f"  [{GREEN}]✓[/{GREEN}] Found: [{BODY}]{rel}[/{BODY}]")

    # Check top-level markdown for spec keywords
    for md in cwd.glob("*.md"):
        rel = md.name
        if rel in detected:
            continue
        try:
            text = md.read_text(errors="replace")
            if any(kw in text for kw in ["SHALL", "MUST", "Given/When/Then"]):
                detected.append(rel)
                console.print(f"  [{GREEN}]✓[/{GREEN}] Found (spec keywords): [{BODY}]{rel}[/{BODY}]")
        except Exception:
            pass

    if not detected:
        console.print(f"  [{MUTED}]· No spec files detected[/{MUTED}]")

    confirmed = Confirm.ask(f"\n[{BODY}]Are these correct?[/{BODY}]", default=True)
    if not confirmed:
        detected = []

    extra = Prompt.ask(
        f"[{MUTED}]Any additional spec paths? (comma-separated, blank to skip)[/{MUTED}]",
        default="",
    )
    if extra.strip():
        for p in extra.split(","):
            p = p.strip()
            if p and p not in detected:
                detected.append(p)

    # Check bash availability
    bash_ok = subprocess.run(["which", "bash"], capture_output=True).returncode == 0
    if not bash_ok:
        console.print(
            f"\n[{AMBER}]⚠ bash not found. Granum hooks require bash (WSL on Windows).[/{AMBER}]\n"
            f"  MCP server and CLI will still work, but hooks will not fire."
        )

    # Write config
    granum_dir.mkdir(exist_ok=True)
    config = {
        "project_id": project_id,
        "spec_paths": detected,
        "compaction_threshold": existing_config.get("compaction_threshold", 50),
        "stale_threshold_days": existing_config.get("stale_threshold_days", 7),
        "freshness_decay_days": existing_config.get("freshness_decay_days", 90),
        "spec_retrieval_limit": existing_config.get("spec_retrieval_limit", 10),
        "memory_retrieval_limit": existing_config.get("memory_retrieval_limit", 10),
        "embedding_model": existing_config.get("embedding_model", "all-MiniLM-L6-v2"),
    }
    _save_config(granum_dir, config)

    # Write .granum/.gitignore
    gitignore = granum_dir / ".gitignore"
    gitignore.write_text("session.log\ntool_call_count\nkuzu.db\n")

    # Wire MCP server into .mcp.json
    _write_mcp_json(cwd)

    # Wire hooks into .claude/settings.json
    _write_claude_settings(cwd, bash_ok)

    console.print(f"\n[{GREEN}]✓[/{GREEN}] Config written to [{ORANGE}].granum/config.json[/{ORANGE}]")
    console.print(f"[{MUTED}]Project: {git_root} (branch: {branch})[/{MUTED}]")
    console.print(f"[{MUTED}]Project ID: {project_id}[/{MUTED}]\n")


@app.command("config", rich_help_panel="[bold #4ade80]Setup[/bold #4ade80]")
def config_cmd(
    action: str = typer.Argument(..., help="set"),
    key: str = typer.Argument(...),
    value: str = typer.Argument(...),
):
    """Edit config tunables. [dim]Usage: granum config set <key> <value>[/dim]"""
    if action != "set":
        console.print(f"[{RED}]✗ Only 'set' supported. Usage: granum config set <key> <value>[/{RED}]")
        raise typer.Exit(1)

    granum_dir = _find_granum_dir()
    config = _load_config(granum_dir)

    int_keys = {"compaction_threshold", "stale_threshold_days", "freshness_decay_days",
                 "spec_retrieval_limit", "memory_retrieval_limit"}
    if key in int_keys:
        try:
            config[key] = int(value)
        except ValueError:
            console.print(f"[{RED}]✗ {key} must be an integer[/{RED}]")
            raise typer.Exit(1)
    else:
        config[key] = value

    _save_config(granum_dir, config)
    console.print(f"[{GREEN}]✓[/{GREEN}] [{ORANGE}]{key}[/{ORANGE}] = [{BODY}]{config[key]}[/{BODY}]")


@app.command("list", rich_help_panel="[bold #4ade80]Memory[/bold #4ade80]")
def list_cmd(
    project: Optional[str] = typer.Option(None, "--project", help="Path to project root"),
    type_filter: Optional[str] = typer.Option(None, "--type", help="Filter by chunk type"),
    show_deprecated: bool = typer.Option(False, "--show-deprecated"),
):
    """List all memory and spec chunks."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    with _spinner("Querying memory store") as status:
        chunks = _get_chunks(granum_dir, project_id, config, include_deprecated=show_deprecated, _status=status)
        status.update("Querying spec store")
        raw_specs = _ipc_spec_chunks(granum_dir, project_id)
        if raw_specs is None:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from mcp_server.models import Chunk as _Chunk
            spec_chunks = _get_db(config, granum_dir).get_spec_chunks(project_id)
        else:
            from mcp_server.models import Chunk as _Chunk
            spec_chunks = [_Chunk.from_dict(d) for d in raw_specs]

    if type_filter:
        chunks = [c for c in chunks if c.type == type_filter]
        if type_filter != "spec":
            spec_chunks = []

    stale_threshold = config.get("stale_threshold_days", 7)

    if not chunks and not spec_chunks:
        console.print(f"[{MUTED}]· No chunks found[/{MUTED}]")
        return

    parts = []
    if chunks:     parts.append(f"{len(chunks)} chunk{'s' if len(chunks) != 1 else ''}")
    if spec_chunks: parts.append(f"{len(spec_chunks)} spec{'s' if len(spec_chunks) != 1 else ''}")
    _hdr(*parts)

    type_order = ["decision", "constraint", "preference", "file_state"]
    by_type: dict[str, list] = {}
    for c in chunks:
        by_type.setdefault(c.type, []).append(c)

    for t in type_order:
        group = by_type.get(t, [])
        if not group:
            continue
        icon = TYPE_ICONS.get(t, "·")
        color = TYPE_COLORS.get(t, ORANGE)
        console.print(f"  [{color}]{icon} {t}[/{color}]  [{MUTED}]({len(group)})[/{MUTED}]")
        for c in group:
            age_days = _age_days_from_seconds(_age_seconds(c.updated_at))
            age = _age_str(_age_seconds(c.updated_at))
            age_color = RED if age_days > 14 else (AMBER if age_days > stale_threshold else MUTED)
            dep_str = f" [{GRAY}]deprecated[/{GRAY}]" if c.status == "deprecated" else ""
            title_color = GRAY if c.status == "deprecated" else BODY
            console.print(
                f"    [{MUTED}]{c.id[:8]}[/{MUTED}] [{title_color}]{c.title[:52]}[/{title_color}]"
                f"  [{MUTED}]imp {c.importance}[/{MUTED}]  [{age_color}]{age}[/{age_color}]{dep_str}"
            )
        console.print()

    if spec_chunks:
        sc = TYPE_COLORS["spec"]
        console.print(f"  [{sc}]◇ spec[/{sc}]  [{MUTED}]({len(spec_chunks)})[/{MUTED}]")
        for c in spec_chunks:
            console.print(f"    [{MUTED}]{c.id[:8]}[/{MUTED}] [{GRAY}]{c.title[:52]}[/{GRAY}]  [{MUTED}]re-indexed[/{MUTED}]")
        console.print()


@app.command(rich_help_panel="[bold #4ade80]Memory[/bold #4ade80]")
def recent(
    n: int = typer.Option(10, "--n", help="Number of chunks to show"),
    project: Optional[str] = typer.Option(None, "--project"),
):
    """Show the [bold]N most recently[/bold] updated chunks."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    with _spinner("Querying memory store") as status:
        chunks = _get_chunks(granum_dir, project_id, config, include_deprecated=True, _status=status)
        status.update("Querying spec store")
        spec_chunks = _get_spec_chunks(granum_dir, project_id, config)

    chunks.sort(key=lambda c: c.updated_at, reverse=True)
    chunks = chunks[:n]

    if not chunks and not spec_chunks:
        console.print(f"[{MUTED}]· No chunks found[/{MUTED}]")
        return

    stale_threshold = config.get("stale_threshold_days", 7)
    total = len(chunks) + len(spec_chunks)
    _hdr(f"{total} recent")

    for c in chunks:
        icon = TYPE_ICONS.get(c.type, "·")
        color = TYPE_COLORS.get(c.type, ORANGE)
        age_days = _age_days_from_seconds(_age_seconds(c.updated_at))
        age = _age_str(_age_seconds(c.updated_at))
        age_color = RED if age_days > 14 else (AMBER if age_days > stale_threshold else MUTED)
        console.print(
            f"  [{MUTED}]{c.id[:8]}[/{MUTED}] [{color}]{icon} {c.type:<11}[/{color}]"
            f" [{BODY}]{c.title[:50]}[/{BODY}]  [{age_color}]{age}[/{age_color}]"
        )
    for c in spec_chunks:
        sc = TYPE_COLORS["spec"]
        console.print(
            f"  [{MUTED}]{c.id[:8]}[/{MUTED}] [{sc}]◇ spec      [/{sc}]"
            f" [{GRAY}]{c.title[:50]}[/{GRAY}]  [{MUTED}]re-indexed[/{MUTED}]"
        )
    console.print()


@app.command(rich_help_panel="[bold #4ade80]Memory[/bold #4ade80]")
def stale(project: Optional[str] = typer.Option(None, "--project")):
    """List [bold]stale[/bold] chunks that need review, oldest first."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]
    threshold = config.get("stale_threshold_days", 7)

    with _spinner("Loading chunks") as status:
        chunks = _get_chunks(granum_dir, project_id, config, _status=status)
        status.update("Checking freshness")

    stale_chunks = [c for c in chunks if _age_days_from_seconds(_age_seconds(c.updated_at)) > threshold]
    stale_chunks.sort(key=lambda c: c.updated_at)

    if not stale_chunks:
        _hdr("all fresh")
        console.print(f"  [{GREEN}]✓ no stale chunks[/{GREEN}]\n")
        return

    _hdr(f"{len(stale_chunks)} stale  >{threshold}d")
    for c in stale_chunks:
        icon = TYPE_ICONS.get(c.type, "·")
        color = TYPE_COLORS.get(c.type, ORANGE)
        age_days = _age_days_from_seconds(_age_seconds(c.updated_at))
        age = _age_str(_age_seconds(c.updated_at))
        age_color = RED if age_days > 14 else AMBER
        console.print(
            f"  [{MUTED}]{c.id[:8]}[/{MUTED}] [{color}]{icon} {c.type:<11}[/{color}]"
            f" [{BODY}]{c.title[:50]}[/{BODY}]  [{age_color}]{age}[/{age_color}]  [{MUTED}]imp {c.importance}[/{MUTED}]"
        )
    console.print()


@app.command(rich_help_panel="[bold #4ade80]Memory[/bold #4ade80]")
def search(
    query: str = typer.Argument(...),
    project: Optional[str] = typer.Option(None, "--project"),
):
    """[bold]Semantic search[/bold] across memory and spec chunks."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    with _spinner("Embedding query") as status:
        results = _ipc_query(granum_dir, query, config)
        if results is None:
            status.update("Running similarity search (direct)")
            db = _get_db(config, granum_dir)
            raw = db.query_context(
                project_id=project_id,
                query=query,
                memory_limit=config.get("memory_retrieval_limit", 7),
                spec_limit=config.get("spec_retrieval_limit", 3),
                freshness_decay_days=config.get("freshness_decay_days", 90),
            )
            results = raw.get("chunks", []) if isinstance(raw, dict) else raw

    if not results:
        _hdr(f'"{query}"')
        console.print(f"  [{MUTED}]· no results[/{MUTED}]\n")
        return

    _hdr(f"{len(results)} result{'s' if len(results) != 1 else ''}  ·  \"{query}\"")
    for r in results:
        icon = TYPE_ICONS.get(r["type"], "·")
        color = TYPE_COLORS.get(r["type"], MUTED)
        stale_str = f"  [{AMBER}]⚠[/{AMBER}]" if r.get("stale_warning") else ""
        via = r.get("retrieved_via", "")
        via_str = f"  [{MUTED}]via graph[/{MUTED}]" if via and via != "similarity" else ""
        console.print(
            f"  [{MUTED}]{r.get('id', '')[:8]}[/{MUTED}] [{color}]{icon} {r['type']:<11}[/{color}]"
            f" [{BODY}]{r['title'][:50]}[/{BODY}]{stale_str}"
            f"  [{ORANGE}]{r['final_score']:.2f}[/{ORANGE}]"
            f"  [{MUTED}]sim {r['similarity']:.2f}  {r['age']}[/{MUTED}]{via_str}"
        )
    console.print()


@app.command(rich_help_panel="[bold #4ade80]Memory[/bold #4ade80]")
def delete(chunk_id: str = typer.Argument(...)):
    """[bold]Soft-delete[/bold] a chunk by ID prefix."""
    granum_dir = _find_granum_dir()
    config = _load_config(granum_dir)

    confirmed = Confirm.ask(
        f"[{ORANGE}]Delete chunk {chunk_id[:12]}?[/{ORANGE}] This cannot be undone.",
        default=False,
    )
    if not confirmed:
        console.print(f"[{MUTED}]· Cancelled[/{MUTED}]")
        return

    with _spinner("Deleting chunk") as status:
        db = _get_db(config, granum_dir)
        ok = db.soft_delete(chunk_id)
        if ok:
            status.update("Saving to disk")
            db.export_ndjson(config["project_id"])

    if ok:
        console.print(f"[{GREEN}]✓[/{GREEN}] Deleted [{MUTED}]{chunk_id[:12]}[/{MUTED}]")
    else:
        # Check if already tombstoned in ndjson
        ndjson = granum_dir / "chunks.ndjson"
        already_gone = False
        if ndjson.exists():
            for line in ndjson.read_text().splitlines():
                try:
                    d = json.loads(line)
                    if d.get("id", "").startswith(chunk_id) and d.get("deleted_at"):
                        already_gone = True
                        break
                except Exception:
                    pass
        if already_gone:
            console.print(f"[{MUTED}]· Already deleted: {chunk_id[:12]}[/{MUTED}]")
        else:
            console.print(f"[{RED}]✗ Chunk not found: {chunk_id}[/{RED}]")


@app.command(rich_help_panel="[bold #4ade80]Analysis[/bold #4ade80]")
def stats(project: Optional[str] = typer.Option(None, "--project")):
    """Show project memory [bold]statistics[/bold]."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]
    threshold = config.get("stale_threshold_days", 7)

    with _spinner("Querying memory store") as status:
        all_chunks = _get_chunks(granum_dir, project_id, config, include_deprecated=True, _status=status)
        status.update("Querying spec store")
        spec_chunks = _get_spec_chunks(granum_dir, project_id, config)
        status.update("Computing stats")
        active = [c for c in all_chunks if c.status == "active"]

    deprecated = [c for c in all_chunks if c.status == "deprecated"]
    stale = [c for c in active if _age_days_from_seconds(_age_seconds(c.updated_at)) > threshold]
    top_retrieved = sorted(active, key=lambda c: getattr(c, "retrieval_count", 0), reverse=True)[:3]

    db_size = _dir_size(granum_dir / "kuzu.db")
    _hdr(f"{len(active)} active  ·  {len(spec_chunks)} specs")

    def _kv(key: str, val: str) -> None:
        console.print(f"  [{MUTED}]{key:<14}[/{MUTED}] {val}")

    spec_paths_str = ", ".join(config.get("spec_paths", [])) or "none"
    _kv("spec paths",  f"[{BODY}]{spec_paths_str}[/{BODY}]")
    _kv("memory",      f"[{GREEN}]{len(active)} active[/{GREEN}]  [{GRAY}]{len(deprecated)} deprecated[/{GRAY}]")
    _kv("specs",       f"[{BODY}]{len(spec_chunks)} indexed[/{BODY}]")
    _kv("stale",       f"[{AMBER}]{len(stale)}[/{AMBER}]" if stale else f"[{MUTED}]0[/{MUTED}]")
    _kv("db size",     f"[{BODY}]{db_size}[/{BODY}]")
    _kv("embedding",   f"[{MUTED}]{config.get('embedding_model', 'all-MiniLM-L6-v2')}[/{MUTED}]")
    if top_retrieved and getattr(top_retrieved[0], "retrieval_count", 0) > 0:
        console.print()
        console.print(f"  [{MUTED}]top retrieved[/{MUTED}]")
        for c in top_retrieved:
            icon = TYPE_ICONS.get(c.type, "·")
            color = TYPE_COLORS.get(c.type, ORANGE)
            console.print(
                f"    [{color}]{icon}[/{color}] [{BODY}]{c.title[:50]}[/{BODY}]"
                f"  [{ORANGE}]×{getattr(c, 'retrieval_count', 0)}[/{ORANGE}]"
            )
    console.print()


@app.command(rich_help_panel="[bold #4ade80]Analysis[/bold #4ade80]")
def audit(project: Optional[str] = typer.Option(None, "--project")):
    """Memory [bold]health report[/bold] — stale, duplicates, low-value chunks."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]
    threshold = config.get("stale_threshold_days", 7)

    with _spinner("Querying memory store") as status:
        all_chunks = _get_chunks(granum_dir, project_id, config, include_deprecated=True, _status=status)
        status.update("Querying spec store")
        spec_chunks = _get_spec_chunks(granum_dir, project_id, config)
        status.update("Loading graph edges")
        all_edges = _ipc_all_edges(granum_dir, project_id)
        if all_edges is None:
            db = _get_db(config, granum_dir)
            all_edges = db.get_all_edges(project_id)
        status.update("Analysing")
        active = [c for c in all_chunks if c.status == "active"]

    deprecated     = [c for c in all_chunks if c.status == "deprecated"]
    stale          = [c for c in active if _age_days_from_seconds(_age_seconds(c.updated_at)) > threshold]
    very_stale     = [c for c in active if _age_days_from_seconds(_age_seconds(c.updated_at)) > 30]
    low_value      = [c for c in active if c.importance <= 2 and _age_days_from_seconds(_age_seconds(c.updated_at)) > threshold]
    # Retrieval anomalies
    undervalued    = [c for c in active if getattr(c, "retrieval_count", 0) >= 5 and c.importance <= 2]
    never_used     = [c for c in active if getattr(c, "retrieval_count", 0) == 0 and c.importance >= 4]
    conflicts      = [(e["from_id"], e["from_title"], e["to_id"], e["to_title"], e.get("confidence", 1.0))
                     for e in (all_edges or []) if e["edge_type"] == "CONTRADICTS"]
    # Deduplicate bidirectional pairs
    seen_pairs: set[frozenset] = set()
    unique_conflicts = []
    for c in conflicts:
        key = frozenset([c[0], c[2]])
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_conflicts.append(c)

    orphans = _find_orphans(active, all_edges or [])

    _hdr(f"{len(active)} active  ·  {len(spec_chunks)} specs  ·  {len(all_edges or [])} edges")

    def _kv(key: str, val: str) -> None:
        console.print(f"  [{MUTED}]{key:<14}[/{MUTED}] {val}")

    _kv("active",      f"[{GREEN}]{len(active)}[/{GREEN}]")
    _kv("specs",       f"[{BODY}]{len(spec_chunks)}[/{BODY}]")
    _kv("edges",       f"[{BODY}]{len(all_edges or [])}[/{BODY}]")
    _kv("deprecated",  f"[{GRAY}]{len(deprecated)}[/{GRAY}]")
    _kv("stale",       f"[{AMBER}]{len(stale)}[/{AMBER}]  [{MUTED}](>{threshold}d)[/{MUTED}]" if stale else f"[{MUTED}]0[/{MUTED}]")
    _kv("very stale",  f"[{RED}]{len(very_stale)}[/{RED}]  [{MUTED}](>30d)[/{MUTED}]" if very_stale else f"[{MUTED}]0[/{MUTED}]")
    _kv("conflicts",   f"[{RED}]{len(unique_conflicts)} pair(s)[/{RED}]" if unique_conflicts else f"[{MUTED}]0[/{MUTED}]")
    _kv("low value",   f"[{AMBER}]{len(low_value)}[/{AMBER}]  [{MUTED}](imp ≤2, stale)[/{MUTED}]" if low_value else f"[{MUTED}]0[/{MUTED}]")
    _kv("orphans",     f"[{MUTED}]{len(orphans)}[/{MUTED}]  [{MUTED}](no edges)[/{MUTED}]" if orphans else f"[{MUTED}]0[/{MUTED}]")
    _kv("undervalued", f"[{AMBER}]{len(undervalued)}[/{AMBER}]  [{MUTED}](retrieved ≥5×, imp ≤2)[/{MUTED}]" if undervalued else f"[{MUTED}]0[/{MUTED}]")
    _kv("never used",  f"[{MUTED}]{len(never_used)}[/{MUTED}]  [{MUTED}](imp ≥4, 0 retrievals)[/{MUTED}]" if never_used else f"[{MUTED}]0[/{MUTED}]")

    if unique_conflicts:
        console.print()
        console.print(f"  [{RED}]⟷ conflicts[/{RED}]  [{MUTED}]({len(unique_conflicts)})[/{MUTED}]")
        for from_id, from_title, to_id, to_title, conf in unique_conflicts:
            console.print(
                f"    [{MUTED}]{from_id[:8]}[/{MUTED}] [{ORANGE}]{from_title[:36]}[/{ORANGE}]"
                f"  [{RED}]⟷[/{RED}]  "
                f"[{MUTED}]{to_id[:8]}[/{MUTED}] [{ORANGE}]{to_title[:36]}[/{ORANGE}]"
                f"  [{MUTED}]sim {conf:.2f}[/{MUTED}]"
            )

    if low_value:
        console.print()
        console.print(f"  [{MUTED}]→ {len(low_value)} low-value chunk(s) — call cleanup_context to deprecate[/{MUTED}]")

    console.print()


@app.command(rich_help_panel="[bold #4ade80]Analysis[/bold #4ade80]")
def timeline(
    project: Optional[str] = typer.Option(None, "--project"),
    months: int = typer.Option(1, "--months", "-m", help="Number of months to show"),
):
    """Calendar [bold]heatmap[/bold] of memory activity."""
    from datetime import date, timedelta
    import calendar

    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    with _spinner("Querying memory store") as status:
        all_chunks = _get_chunks(granum_dir, project_id, config, include_deprecated=True, _status=status)
        status.update("Querying spec store")
        spec_chunks = _get_spec_chunks(granum_dir, project_id, config)
        status.update("Mapping activity to calendar")

    # Count saves per day (memory + specs)
    day_counts: dict[str, int] = {}
    for chunk in all_chunks + spec_chunks:
        try:
            day_counts[chunk.updated_at[:10]] = day_counts.get(chunk.updated_at[:10], 0) + 1
        except Exception:
            pass

    today = date.today()
    max_count = max(day_counts.values()) if day_counts else 1

    def _cell(d: Optional[date]) -> str:
        if d is None:
            return "   "
        count = day_counts.get(str(d), 0)
        if count == 0:
            return f"[{GRAY}]·[/{GRAY}]  "
        intensity = count / max_count
        if intensity < 0.33:
            return f"[{BODY}]▪[/{BODY}]  "
        elif intensity < 0.66:
            return f"[{AMBER}]▪[/{AMBER}]  "
        else:
            return f"[bold {ORANGE}]▪[/bold {ORANGE}]  "

    total = len(all_chunks) + len(spec_chunks)
    _hdr(f"{total} chunk{'s' if total != 1 else ''}  ·  {months} month{'s' if months != 1 else ''}")

    for m_offset in range(months - 1, -1, -1):
        year = today.year
        month = today.month - m_offset
        while month <= 0:
            month += 12
            year -= 1

        month_name = date(year, month, 1).strftime("%B %Y")
        _, days_in_month = calendar.monthrange(year, month)
        first_dow = date(year, month, 1).weekday()  # 0=Mon

        console.print(f"\n  [{MUTED}]{month_name}[/{MUTED}]")
        console.print(f"  [{MUTED}]Mo  Tu  We  Th  Fr  Sa  Su[/{MUTED}]")

        cells: list[Optional[date]] = [None] * first_dow
        for day in range(1, days_in_month + 1):
            cells.append(date(year, month, day))
        while len(cells) % 7 != 0:
            cells.append(None)

        for week_start in range(0, len(cells), 7):
            week = cells[week_start:week_start + 7]
            console.print("  " + "".join(_cell(d) for d in week))

    total_saves = sum(day_counts.values())
    active_days = len(day_counts)
    console.print(f"\n  [{MUTED}]· none  [{BODY}]▪[/{BODY}] light  [{AMBER}]▪[/{AMBER}] medium  [{ORANGE}]▪[/{ORANGE}] heavy    {total_saves} event(s) across {active_days} day(s)  ·  {len(all_chunks)} memory  {len(spec_chunks)} specs[/{MUTED}]\n")


def _build_graph_html(chunks, edges: list[dict], git_root: Optional[str], branch: str) -> str:
    import json as _json

    proj = Path(git_root or ".").name
    type_colors = {
        "decision":   "#88c9a0",
        "constraint": "#d4b483",
        "preference": "#a8d4b5",
        "file_state": "#8bbfa8",
        "spec":       "#4a7a5e",
    }
    edge_colors = {
        "CONTRADICTS":  "#d47f7f",
        "SUPERSEDES":   "#d4b483",
        "RELATES_TO":   "#88c9a0",
        "DERIVED_FROM": "#6abf85",
        "DEPENDS_ON":   "#a8d4b5",
    }
    edge_dash = {"CONTRADICTS": "6,3", "SUPERSEDES": "4,2"}

    nodes = [
        {
            "id":         c.id,
            "label":      c.title[:40] + ("…" if len(c.title) > 40 else ""),
            "fullTitle":  c.title,
            "type":       c.type,
            "importance": c.importance,
            "status":     c.status,
            "color":      type_colors.get(c.type, "#6b7280"),
        }
        for c in chunks
    ]
    node_ids = {c.id for c in chunks}
    links = [
        {
            "source": e["from_id"],
            "target": e["to_id"],
            "type":   e["edge_type"],
            "conf":   round(e.get("confidence") or 1.0, 2),
            "color":  edge_colors.get(e["edge_type"], "#6b7280"),
            "dash":   edge_dash.get(e["edge_type"], "0"),
        }
        for e in edges
        if e["from_id"] in node_ids and e["to_id"] in node_ids
    ]

    data_json = _json.dumps({"nodes": nodes, "links": links})
    legend_items = _json.dumps([
        {"label": k, "color": v} for k, v in type_colors.items()
    ])
    edge_legend = _json.dumps([
        {"label": k, "color": v, "dash": edge_dash.get(k, "0")} for k, v in edge_colors.items()
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Granum — {proj} ({branch})</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0f0f0f; color:#e5e5e5; font-family:monospace; overflow:hidden; }}
#graph {{ width:100vw; height:100vh; }}
.node circle {{ stroke-width:1.5px; cursor:pointer; transition:r .15s; }}
.node circle:hover {{ stroke:#fff !important; }}
.node text {{ font-size:11px; fill:#d4ead9; pointer-events:none; }}
.link {{ stroke-opacity:.6; fill:none; }}
#tooltip {{
  position:fixed; background:#1a1a1a; border:1px solid #333;
  padding:10px 14px; border-radius:6px; font-size:12px; line-height:1.6;
  pointer-events:none; opacity:0; transition:opacity .15s;
  max-width:300px; z-index:10;
}}
#legend {{
  position:fixed; bottom:20px; left:20px; background:#111;
  border:1px solid #222; border-radius:6px; padding:12px 16px; font-size:11px;
}}
#legend h4 {{ color:#4ade80; margin-bottom:6px; font-size:11px; letter-spacing:.05em; }}
.leg-row {{ display:flex; align-items:center; gap:7px; margin:3px 0; color:#9ca3af; }}
.leg-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
.leg-line {{ width:18px; height:2px; flex-shrink:0; }}
#info {{
  position:fixed; top:16px; left:50%; transform:translateX(-50%);
  color:#4a7a5e; font-size:12px; letter-spacing:.05em;
}}
</style>
</head>
<body>
<svg id="graph"></svg>
<div id="tooltip"></div>
<div id="legend">
  <h4>CHUNK TYPE</h4>
  <div id="node-legend"></div>
  <h4 style="margin-top:10px">EDGE TYPE</h4>
  <div id="edge-legend"></div>
</div>
<div id="info">{proj} &nbsp;·&nbsp; {branch} &nbsp;·&nbsp; {len(nodes)} nodes &nbsp;·&nbsp; {len(links)} edges</div>
<script>
const data = {data_json};
const legendItems = {legend_items};
const edgeLegend = {edge_legend};

// Build legends
legendItems.forEach(d => {{
  const row = document.createElement('div');
  row.className = 'leg-row';
  row.innerHTML = `<div class="leg-dot" style="background:${{d.color}}"></div><span>${{d.label}}</span>`;
  document.getElementById('node-legend').appendChild(row);
}});
edgeLegend.forEach(d => {{
  const row = document.createElement('div');
  row.className = 'leg-row';
  const svg = `<svg width="18" height="10"><line x1="0" y1="5" x2="18" y2="5"
    stroke="${{d.color}}" stroke-width="2" stroke-dasharray="${{d.dash}}"/></svg>`;
  row.innerHTML = svg + `<span>${{d.label}}</span>`;
  document.getElementById('edge-legend').appendChild(row);
}});

const svg = d3.select('#graph');
const width = window.innerWidth, height = window.innerHeight;
const tooltip = document.getElementById('tooltip');

const g = svg.append('g');
svg.call(d3.zoom().scaleExtent([.1, 8]).on('zoom', e => g.attr('transform', e.transform)));

// Arrow markers per edge type
const defs = svg.append('defs');
{_json.dumps(list(edge_colors.keys()))}.forEach(type => {{
  const color = {_json.dumps(edge_colors)}[type];
  defs.append('marker')
    .attr('id', 'arrow-' + type)
    .attr('viewBox', '0 -4 8 8').attr('refX', 18).attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', color).attr('opacity', .7);
}});

const sim = d3.forceSimulation(data.nodes)
  .force('link', d3.forceLink(data.links).id(d => d.id).distance(d => d.type === 'RELATES_TO' ? 120 : 90))
  .force('charge', d3.forceManyBody().strength(-260))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collision', d3.forceCollide(28));

const link = g.append('g').selectAll('line')
  .data(data.links).join('line')
  .attr('class', 'link')
  .attr('stroke', d => d.color)
  .attr('stroke-width', 1.5)
  .attr('stroke-dasharray', d => d.dash)
  .attr('marker-end', d => `url(#arrow-${{d.type}})`);

const node = g.append('g').selectAll('g')
  .data(data.nodes).join('g')
  .attr('class', 'node')
  .call(d3.drag()
    .on('start', (e, d) => {{ if (!e.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; }})
    .on('drag',  (e, d) => {{ d.fx=e.x; d.fy=e.y; }})
    .on('end',   (e, d) => {{ if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}));

node.append('circle')
  .attr('r', d => 5 + d.importance * 1.6)
  .attr('fill', d => d.color + (d.type === 'spec' ? '55' : 'cc'))
  .attr('stroke', d => d.color);

node.append('text')
  .attr('dy', d => 8 + d.importance * 1.6)
  .attr('text-anchor', 'middle')
  .text(d => d.label);

node.on('mouseover', (e, d) => {{
  tooltip.style.opacity = 1;
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY - 10) + 'px';
  tooltip.innerHTML =
    `<div style="color:${{d.color}};font-weight:bold;margin-bottom:4px">${{d.fullTitle}}</div>` +
    `<div style="color:#a8d4b5">${{d.type}} &nbsp;·&nbsp; imp ${{d.importance}}</div>` +
    `<div style="color:#4a7a5e;font-size:10px;margin-top:4px">${{d.id.slice(0,12)}}</div>`;
}}).on('mousemove', e => {{
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY - 10) + 'px';
}}).on('mouseout', () => {{ tooltip.style.opacity = 0; }});

sim.on('tick', () => {{
  link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});

window.addEventListener('resize', () => {{
  sim.force('center', d3.forceCenter(window.innerWidth/2, window.innerHeight/2)).alpha(.1).restart();
}});
</script>
</body>
</html>"""


def _pca_2d(embeddings: list[list[float]]) -> list[tuple[float, float]]:
    import numpy as np
    X = np.array(embeddings, dtype=np.float32)
    X -= X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T
    for i in range(2):
        lo, hi = coords[:, i].min(), coords[:, i].max()
        r = hi - lo
        coords[:, i] = (coords[:, i] - lo) / r * 2 - 1 if r > 0 else coords[:, i]
    return [(float(x), float(y)) for x, y in coords]


def _build_embed_html(chunks_with_xy, edges: list[dict], git_root: Optional[str], branch: str) -> str:
    import json as _json

    proj = Path(git_root or ".").name
    type_colors = {
        "decision":   "#88c9a0",
        "constraint": "#d4b483",
        "preference": "#a8d4b5",
        "file_state": "#8bbfa8",
        "spec":       "#4a7a5e",
    }
    edge_colors = {
        "CONTRADICTS":  "#d47f7f",
        "SUPERSEDES":   "#d4b483",
        "RELATES_TO":   "#88c9a0",
        "DERIVED_FROM": "#6abf85",
        "DEPENDS_ON":   "#a8d4b5",
    }
    edge_dash = {"CONTRADICTS": "6,3", "SUPERSEDES": "4,2"}

    nodes = [
        {
            "id":         c["id"],
            "label":      c["title"][:36] + ("…" if len(c["title"]) > 36 else ""),
            "fullTitle":  c["title"],
            "content":    c["content"][:120] + ("…" if len(c["content"]) > 120 else ""),
            "type":       c["type"],
            "importance": c["importance"],
            "retrieval_count": c.get("retrieval_count", 0),
            "color":      type_colors.get(c["type"], "#6b7280"),
            "px":         c["px"],
            "py":         c["py"],
        }
        for c in chunks_with_xy
    ]
    node_ids = {n["id"] for n in nodes}
    links = [
        {
            "source": e["from_id"],
            "target": e["to_id"],
            "type":   e["edge_type"],
            "color":  edge_colors.get(e["edge_type"], "#6b7280"),
            "dash":   edge_dash.get(e["edge_type"], "0"),
        }
        for e in edges
        if e["from_id"] in node_ids and e["to_id"] in node_ids
    ]
    data_json   = _json.dumps({"nodes": nodes, "links": links})
    legend_json = _json.dumps([{"label": k, "color": v} for k, v in type_colors.items()])
    edge_leg    = _json.dumps([{"label": k, "color": v, "dash": edge_dash.get(k, "0")} for k, v in edge_colors.items()])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Granum embed — {proj} ({branch})</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0f0f0f; color:#e5e5e5; font-family:monospace; overflow:hidden; }}
#graph {{ width:100vw; height:100vh; }}
.node circle {{ stroke-width:1.5px; cursor:pointer; transition:r .15s; }}
.node circle:hover {{ stroke:#fff !important; stroke-width:2px; }}
.node text {{ font-size:10px; fill:#d4ead9; pointer-events:none; }}
.link {{ stroke-opacity:.5; fill:none; }}
#axes line {{ stroke:#222; stroke-width:1px; }}
#axes text {{ fill:#4a7a5e; font-size:10px; font-family:monospace; }}
#tooltip {{
  position:fixed; background:#1a1a1a; border:1px solid #2a2a2a;
  padding:10px 14px; border-radius:6px; font-size:12px; line-height:1.7;
  pointer-events:none; opacity:0; transition:opacity .15s;
  max-width:320px; z-index:10;
}}
#legend {{
  position:fixed; bottom:20px; left:20px; background:#111;
  border:1px solid #1e1e1e; border-radius:6px; padding:12px 16px; font-size:11px;
}}
#legend h4 {{ color:#88c9a0; margin-bottom:5px; font-size:10px; letter-spacing:.08em; }}
.leg-row {{ display:flex; align-items:center; gap:7px; margin:3px 0; color:#6b7280; }}
.leg-dot {{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }}
.leg-line {{ width:18px; height:2px; flex-shrink:0; }}
#info {{
  position:fixed; top:16px; left:50%; transform:translateX(-50%);
  color:#4a7a5e; font-size:11px; letter-spacing:.06em;
}}
#subtitle {{
  position:fixed; top:34px; left:50%; transform:translateX(-50%);
  color:#2d5a3d; font-size:10px; letter-spacing:.05em;
}}
</style>
</head>
<body>
<svg id="graph"></svg>
<div id="tooltip"></div>
<div id="legend">
  <h4>CHUNK TYPE</h4>
  <div id="node-legend"></div>
  <h4 style="margin-top:9px">EDGE TYPE</h4>
  <div id="edge-legend"></div>
</div>
<div id="info">{proj} &nbsp;·&nbsp; {branch} &nbsp;·&nbsp; {len(nodes)} chunks</div>
<div id="subtitle">PCA projection — semantic proximity</div>
<script>
const data = {data_json};
const legendItems = {legend_json};
const edgeLegend = {edge_leg};

legendItems.forEach(d => {{
  const row = document.createElement('div');
  row.className = 'leg-row';
  row.innerHTML = `<div class="leg-dot" style="background:${{d.color}}"></div><span>${{d.label}}</span>`;
  document.getElementById('node-legend').appendChild(row);
}});
edgeLegend.forEach(d => {{
  const row = document.createElement('div');
  row.className = 'leg-row';
  const svg = `<svg width="18" height="10"><line x1="0" y1="5" x2="18" y2="5"
    stroke="${{d.color}}" stroke-width="2" stroke-dasharray="${{d.dash}}"/></svg>`;
  row.innerHTML = svg + `<span>${{d.label}}</span>`;
  document.getElementById('edge-legend').appendChild(row);
}});

const W = window.innerWidth, H = window.innerHeight;
const PAD = 80;
const xScale = d3.scaleLinear().domain([-1,1]).range([PAD, W-PAD]);
const yScale = d3.scaleLinear().domain([-1,1]).range([H-PAD, PAD]);

const svg = d3.select('#graph');
const tooltip = document.getElementById('tooltip');

const g = svg.append('g');
svg.call(d3.zoom().scaleExtent([.2,12]).on('zoom', e => g.attr('transform', e.transform)));

// Axes
const ax = g.append('g').attr('id','axes');
ax.append('line').attr('x1', xScale(-1)).attr('x2', xScale(1))
  .attr('y1', yScale(0)).attr('y2', yScale(0));
ax.append('line').attr('y1', yScale(-1)).attr('y2', yScale(1))
  .attr('x1', xScale(0)).attr('x2', xScale(0));
ax.append('text').attr('x', xScale(1)+4).attr('y', yScale(0)+4).text('PC1');
ax.append('text').attr('x', xScale(0)+4).attr('y', yScale(1)-6).text('PC2');

// Arrow markers
const defs = svg.append('defs');
{_json.dumps(list(edge_colors.keys()))}.forEach(type => {{
  const color = {_json.dumps(edge_colors)}[type];
  defs.append('marker')
    .attr('id','arrow-'+type)
    .attr('viewBox','0 -4 8 8').attr('refX',16).attr('markerWidth',5).attr('markerHeight',5)
    .attr('orient','auto')
    .append('path').attr('d','M0,-4L8,0L0,4').attr('fill',color).attr('opacity',.6);
}});

// Position nodes from PCA coords
data.nodes.forEach(d => {{
  d.x = xScale(d.px);
  d.y = yScale(d.py);
}});

const nodeById = Object.fromEntries(data.nodes.map(d => [d.id, d]));

// Edges as straight lines between PCA positions
const link = g.append('g').selectAll('line')
  .data(data.links).join('line')
  .attr('class','link')
  .attr('stroke', d => d.color)
  .attr('stroke-width', 1.2)
  .attr('stroke-dasharray', d => d.dash)
  .attr('marker-end', d => `url(#arrow-${{d.type}})`)
  .attr('x1', d => nodeById[d.source]?.x ?? 0)
  .attr('y1', d => nodeById[d.source]?.y ?? 0)
  .attr('x2', d => nodeById[d.target]?.x ?? 0)
  .attr('y2', d => nodeById[d.target]?.y ?? 0);

// Nodes
const node = g.append('g').selectAll('g')
  .data(data.nodes).join('g')
  .attr('class','node')
  .attr('transform', d => `translate(${{d.x}},${{d.y}})`);

node.append('circle')
  .attr('r', d => 4 + d.importance * 1.4)
  .attr('fill', d => d.color + (d.type === 'spec' ? '44' : 'bb'))
  .attr('stroke', d => d.color);

node.append('text')
  .attr('dy', d => 7 + d.importance * 1.4)
  .attr('text-anchor','middle')
  .text(d => d.label);

node.on('mouseover', (e, d) => {{
  tooltip.style.opacity = 1;
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY - 10) + 'px';
  tooltip.innerHTML =
    `<div style="color:${{d.color}};font-weight:bold;margin-bottom:4px">${{d.fullTitle}}</div>` +
    `<div style="color:#a8d4b5;margin-bottom:6px">${{d.content}}</div>` +
    `<div style="color:#6b7280">${{d.type}} &nbsp;·&nbsp; imp ${{d.importance}} &nbsp;·&nbsp; retrieved ${{d.retrieval_count}}×</div>` +
    `<div style="color:#4a7a5e;font-size:10px;margin-top:3px">${{d.id.slice(0,12)}}</div>`;
}}).on('mousemove', e => {{
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY - 10) + 'px';
}}).on('mouseout', () => {{ tooltip.style.opacity = 0; }});
</script>
</body>
</html>"""


_EDGE_COLORS = {
    "CONTRADICTS":  RED,
    "SUPERSEDES":   AMBER,
    "RELATES_TO":   MUTED,
    "DERIVED_FROM": GREEN,
    "DEPENDS_ON":   ORANGE,
    "CONTAINS":     MUTED,
}

_EDGE_LABELS = {
    "CONTRADICTS":  "CONTRADICTS",
    "SUPERSEDES":   "SUPERSEDES",
    "RELATES_TO":   "RELATES TO",
    "DERIVED_FROM": "DERIVED FROM",
    "DEPENDS_ON":   "DEPENDS ON",
    "CONTAINS":     "CONTAINS",
}


def _edge_group_key(e: dict) -> str:
    """Spec-hierarchy RELATES_TO edges display as their own CONTAINS group."""
    if e["edge_type"] == "RELATES_TO" and e.get("created_by") == "hierarchy":
        return "CONTAINS"
    return e["edge_type"]


def _edge_str(edge_type: str, confidence: Optional[float], direction: str) -> str:
    color = _EDGE_COLORS.get(edge_type, MUTED)
    label = _EDGE_LABELS.get(edge_type, edge_type)
    arrow = "──►" if direction == "outgoing" else "◄──"
    conf_str = f" [{MUTED}]({confidence:.2f})[/{MUTED}]" if confidence and confidence < 1.0 else ""
    return f"[{color}]{arrow} {label}[/{color}]{conf_str}"


def _chunk_node_str(chunk_id: str, title: str, chunk_type: str, status: str) -> str:
    icon = TYPE_ICONS.get(chunk_type, "·")
    color = TYPE_COLORS.get(chunk_type, MUTED)
    dep = f" [{GRAY}](deprecated)[/{GRAY}]" if status == "deprecated" else ""
    return f"[{color}]{icon} {title}[/{color}]  [{MUTED}]{chunk_id[:8]}[/{MUTED}]{dep}"


@app.command(rich_help_panel="[bold #4ade80]Analysis[/bold #4ade80]")
def graph(
    query: Optional[str] = typer.Argument(None, help="Center graph on closest matching chunk"),
    project: Optional[str] = typer.Option(None, "--project"),
    depth: int = typer.Option(1, "--depth", "-d", help="Hop depth for centered view (1 or 2)"),
    open_browser: bool = typer.Option(False, "--open", "-o", help="Open interactive D3 graph in browser"),
    embed: bool = typer.Option(False, "--embed", "-e", help="Open PCA vector-space scatter plot in browser"),
):
    """Visualize memory [bold]relationship graph[/bold]. --open for Obsidian-style browser view."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    if embed:
        with _spinner("Loading chunks and embeddings") as status:
            db = _get_db(config, granum_dir)
            rows = db.get_all_chunks_with_embeddings(project_id)
            rows = [r for r in rows if r.get("embedding")]
            status.update(f"Computing PCA  {len(rows)} chunks")
            embeddings = [r["embedding"] for r in rows]
            if len(embeddings) < 2:
                console.print(f"[{AMBER}]⚠ Need at least 2 chunks with embeddings for PCA[/{AMBER}]")
                return
            coords = _pca_2d(embeddings)
            chunks_with_xy = [
                {**r, "px": coords[i][0], "py": coords[i][1]}
                for i, r in enumerate(rows)
            ]
            status.update("Loading graph edges")
            edges = db.get_all_edges(project_id)
        html_path = granum_dir / "embed.html"
        html = _build_embed_html(chunks_with_xy, edges or [], _git_root(), _git_branch())
        html_path.write_text(html)
        import webbrowser
        webbrowser.open(f"file://{html_path.resolve()}")
        console.print(f"[{GREEN}]✓[/{GREEN}] Opened [{ORANGE}]{html_path}[/{ORANGE}]")
        return

    if open_browser:
        with _spinner("Loading memory chunks") as status:
            all_chunks = _get_chunks(granum_dir, project_id, config, include_deprecated=False, _status=status)
            status.update("Loading spec chunks")
            spec_chunks = _get_spec_chunks(granum_dir, project_id, config)
            status.update("Loading graph edges")
            edges = _ipc_all_edges(granum_dir, project_id)
            if edges is None:
                db = _get_db(config, granum_dir)
                edges = db.get_all_edges(project_id)
            status.update(f"Generating graph  {len(all_chunks + spec_chunks)} nodes · {len(edges or [])} edges")
        html_path = granum_dir / "graph.html"
        html = _build_graph_html(all_chunks + spec_chunks, edges or [], _git_root(), _git_branch())
        html_path.write_text(html)
        import webbrowser
        webbrowser.open(f"file://{html_path.resolve()}")
        console.print(f"[{GREEN}]✓[/{GREEN}] Opened [{ORANGE}]{html_path}[/{ORANGE}]")
        return

    if query:
        # Centered view: find closest chunk, show its neighborhood
        with _spinner("Finding chunk") as status:
            results = _ipc_query(granum_dir, query, config)
            if results is None:
                db = _get_db(config, granum_dir)
                raw = db.query_context(
                    project_id=project_id, query=query,
                    memory_limit=1, spec_limit=0,
                    freshness_decay_days=config.get("freshness_decay_days", 90),
                )
                results = raw.get("chunks", []) if isinstance(raw, dict) else raw

        memory_only = [r for r in (results or []) if r.get("type") != "spec"]
        if not memory_only:
            console.print(f"[{MUTED}]· No matching chunk found[/{MUTED}]")
            return

        root = memory_only[0]
        with _spinner("Traversing graph") as status:
            edges = _ipc_edges(granum_dir, root["id"], depth=depth)
            if edges is None:
                db = _get_db(config, granum_dir)
                edges = db.get_edges(root["id"], depth=depth)

        tree = Tree(_chunk_node_str(root["id"], root["title"], root["type"], root.get("status", "active")))

        # Group edges by type for cleaner display
        by_type: dict[str, list] = {}
        for e in (edges or []):
            by_type.setdefault(_edge_group_key(e), []).append(e)

        for et, group in sorted(by_type.items()):
            color = _EDGE_COLORS.get(et, MUTED)
            label = _EDGE_LABELS.get(et, et)
            type_branch = tree.add(f"[{color}]{label}[/{color}]")
            for e in group:
                conf_str = f"  [{MUTED}]({e['confidence']:.2f})[/{MUTED}]" if e.get("confidence") and e["confidence"] < 1.0 else ""
                via_str = f"  [dim {MUTED}]via {e['via'][:8]}[/dim {MUTED}]" if e.get("via") else ""
                dir_arrow = "►" if e["direction"] == "outgoing" else "◄"
                node_str = _chunk_node_str(e["chunk_id"], e["title"], e["type"], e.get("status", "active"))
                type_branch.add(f"[{MUTED}]{dir_arrow}[/{MUTED}] {node_str}{conf_str}{via_str}")

        console.print(f"\n  [{MUTED}]centered on closest match to:[/{MUTED}] [{ORANGE}]{query}[/{ORANGE}]\n")
        console.print(tree)
        console.print()

    else:
        # Full project edge list
        with _spinner("Loading graph") as status:
            edges = _ipc_all_edges(granum_dir, project_id)
            if edges is None:
                db = _get_db(config, granum_dir)
                edges = db.get_all_edges(project_id)

        if not edges:
            console.print(f"[{MUTED}]· No edges yet — edges form automatically as chunks are saved[/{MUTED}]")
            return

        proj = Path(_git_root() or ".").name
        branch = _git_branch()
        console.print(f"\n  [{ORANGE}]{proj}[/{ORANGE}]  [{MUTED}]{branch}[/{MUTED}]  [{MUTED}]·  {len(edges)} edge(s)[/{MUTED}]\n")

        # Group by edge type
        by_type: dict[str, list] = {}
        for e in edges:
            by_type.setdefault(_edge_group_key(e), []).append(e)

        for et in ["CONTRADICTS", "SUPERSEDES", "DEPENDS_ON", "RELATES_TO", "DERIVED_FROM", "CONTAINS"]:
            group = by_type.get(et, [])
            if not group:
                continue
            color = _EDGE_COLORS.get(et, MUTED)
            label = _EDGE_LABELS.get(et, et)
            console.print(f"  [{color}]{label}[/{color}]  [{MUTED}]({len(group)})[/{MUTED}]")
            for e in group:
                from_icon = TYPE_ICONS.get(e["from_type"], "·")
                from_color = TYPE_COLORS.get(e["from_type"], MUTED)
                to_icon = TYPE_ICONS.get(e["to_type"], "·")
                to_color = TYPE_COLORS.get(e["to_type"], MUTED)
                conf_str = f" [{MUTED}]({e['confidence']:.2f})[/{MUTED}]" if e.get("confidence") and e["confidence"] < 1.0 else ""
                auto_str = f" [{MUTED}][auto][/{MUTED}]" if e.get("created_by") == "auto" else ""
                console.print(
                    f"    [{MUTED}]{e['from_id'][:8]}[/{MUTED}] [{from_color}]{from_icon} {e['from_title'][:36]}[/{from_color}]"
                    f"  [{color}]──►[/{color}]"
                    f"  [{MUTED}]{e['to_id'][:8]}[/{MUTED}] [{to_color}]{to_icon} {e['to_title'][:36]}[/{to_color}]"
                    f"{conf_str}{auto_str}"
                )
            console.print()


@app.command(rich_help_panel="[bold #4ade80]Memory[/bold #4ade80]")
def history(
    chunk_id: str = typer.Argument(..., help="Chunk ID prefix to inspect version history"),
    project: Optional[str] = typer.Option(None, "--project"),
):
    """Show [bold]version history[/bold] for a chunk (SUPERSEDES chain)."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    with _spinner("Loading chunks") as status:
        chunks = _get_chunks(granum_dir, project_id, config, include_deprecated=True, _status=status)
        status.update("Loading spec chunks")
        spec_chunks = _get_spec_chunks(granum_dir, project_id, config)
        all_chunks = list(chunks) + list(spec_chunks)
        matched = [c for c in all_chunks if c.id.startswith(chunk_id)]
        if not matched:
            console.print(f"[{RED}]✗[/{RED}] No chunk found with id prefix [{MUTED}]{chunk_id}[/{MUTED}]")
            raise typer.Exit(1)
        current = matched[0]
        full_id = current.id
        status.update("Loading version history")
        versions = _ipc_chunk_history(granum_dir, full_id)
        if versions is None:
            db = _get_db(config, granum_dir)
            versions = db.get_chunk_history(full_id)

    icon = TYPE_ICONS.get(current.type, "◆")
    color = TYPE_COLORS.get(current.type, ORANGE)
    age = _age_str(_age_seconds(current.updated_at or ""))
    v_count = len(versions) if versions else 0

    _hdr(f"{icon} {current.title[:48]}  ·  {v_count} version{'s' if v_count != 1 else ''}")
    console.print(f"  [{MUTED}]{full_id[:12]}[/{MUTED}]\n")
    console.print(f"  [{GREEN}]✓ current[/{GREEN}]  [{BODY}]{current.content[:120]}[/{BODY}]")
    console.print(f"  [{MUTED}]  updated {age}  ·  imp {current.importance}[/{MUTED}]")

    if not versions:
        console.print()
        console.print(f"  [{MUTED}]· No prior versions (chunk has never been updated)[/{MUTED}]")
    else:
        for i, v in enumerate(versions):
            age_v = _age_str(_age_seconds(v.get("updated_at", "")))
            snap_id = v["id"]
            console.print()
            console.print(f"  [{AMBER}]v{len(versions) - i}[/{AMBER}]  [{MUTED}]{snap_id[-14:]}[/{MUTED}]  [{MUTED}]updated {age_v}[/{MUTED}]")
            console.print(f"  [{GRAY}]{v.get('content', '')[:120]}[/{GRAY}]")

    console.print()


@app.command(rich_help_panel="[bold #4ade80]Memory[/bold #4ade80]")
def clear(
    project: Optional[str] = typer.Option(None, "--project"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """[bold]Delete all[/bold] memory chunks for this project."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    if not yes:
        console.print(f"[{ORANGE}]⚠ This will delete all active memory chunks for project {project_id[:8]}.[/{ORANGE}]")
        confirmed = Confirm.ask("Continue?", default=False)
        if not confirmed:
            console.print(f"[{MUTED}]· Cancelled[/{MUTED}]")
            return

    with _spinner("Loading chunks") as status:
        db = _get_db(config, granum_dir)
        chunks = db.get_all_memory_chunks(project_id, include_deprecated=True)
        status.update(f"Deleting {len(chunks)} chunks")
        for chunk in chunks:
            db.soft_delete(chunk.id)
        status.update("Saving to disk")
        db.export_ndjson(project_id)

    console.print(f"[{GREEN}]✓[/{GREEN}] Cleared [{ORANGE}]{len(chunks)}[/{ORANGE}] chunk(s)")


@app.command(rich_help_panel="[bold #4ade80]Analysis[/bold #4ade80]")
def drift(
    project: Optional[str] = typer.Option(None, "--project"),
    all_types: bool = typer.Option(False, "--all", "-a", help="Check all chunk types, not just file_state"),
    threshold: int = typer.Option(60, "--days", help="Flag chunks older than N days as stale"),
):
    """Check for [bold]drifted or stale[/bold] memory chunks — verify stored facts against codebase."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]
    git_root = str(_git_root() or ".")

    with _spinner("Checking chunks against codebase") as _status:
        db = _get_db(config, granum_dir)
        types = None if all_types else ["file_state"]
        results = db.check_drift(project_id, git_root, age_threshold_days=threshold, types=types)

    if not results:
        label = "chunks" if all_types else "file_state chunks"
        console.print(f"[{MUTED}]· No {label} to check[/{MUTED}]")
        return

    flagged = [r for r in results if r["verdict"] in ("drifted", "stale")]
    clean   = [r for r in results if r["verdict"] == "ok"]

    _hdr(f"{len(results)} chunk{'s' if len(results) != 1 else ''} checked")

    if flagged:
        console.print(f"  [{RED}]⚠ potentially drifted ({len(flagged)})[/{RED}]\n")
        for r in flagged:
            icon = TYPE_ICONS.get(r["type"], "·")
            score_str = f"{r['score']:.0%}" if r["score"] is not None else "no terms"
            flags = []
            if r["age_days"] > threshold:
                flags.append(f"[{AMBER}]{r['age_days']}d old[/{AMBER}]")
            if r["score"] is not None and r["score"] < 0.6:
                flags.append(f"[{RED}]{score_str} verified[/{RED}]")
            flag_str = "  " + "  ".join(flags) if flags else ""
            console.print(f"  [{MUTED}]{r['id'][:8]}[/{MUTED}] [{ORANGE}]{icon} {r['title'][:50]}[/{ORANGE}]{flag_str}")
            if r["missing_paths"]:
                console.print(f"    [{GRAY}]missing paths: {', '.join(r['missing_paths'][:3])}[/{GRAY}]")
            if r["missing_idents"]:
                console.print(f"    [{GRAY}]unverified: {', '.join(r['missing_idents'][:4])}[/{GRAY}]")
        console.print()

    if clean:
        console.print(f"  [{GREEN}]✓ verified ({len(clean)})[/{GREEN}]")
        for r in clean:
            score_str = f"{r['score']:.0%}" if r["score"] is not None else "—"
            console.print(f"    [{MUTED}]{r['id'][:8]}[/{MUTED}] [{BODY}]{r['title'][:50]}[/{BODY}]  [{MUTED}]{score_str}[/{MUTED}]")
        console.print()

    if not flagged:
        console.print(f"  [{GREEN}]✓ all chunks verified against codebase[/{GREEN}]\n")


@app.command(rich_help_panel="[bold #4ade80]Analysis[/bold #4ade80]")
def summarize(
    project: Optional[str] = typer.Option(None, "--project"),
    hours: int = typer.Option(24, "--hours", "-h", help="Look back N hours"),
    save: bool = typer.Option(False, "--save", "-s", help="Store summary as a handoff chunk"),
):
    """Show [bold]recent memory changes[/bold] as a session handoff narrative."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    with _spinner(f"Loading changes in last {hours}h") as status:
        db = _get_db(config, granum_dir)
        chunks = db.get_recent_changes(project_id, since_hours=hours)

    if not chunks:
        console.print(f"[{MUTED}]· No memory changes in the last {hours}h[/{MUTED}]")
        return

    # Group by type
    by_type: dict[str, list] = {}
    for c in chunks:
        by_type.setdefault(c.type, []).append(c)

    _hdr(f"last {hours}h  ·  {len(chunks)} change{'s' if len(chunks) != 1 else ''}")

    type_order = ["decision", "constraint", "file_state", "preference"]
    for t in type_order:
        group = by_type.get(t, [])
        if not group:
            continue
        icon = TYPE_ICONS.get(t, "·")
        color = TYPE_COLORS.get(t, ORANGE)
        console.print(f"  [{color}]{icon} {t}[/{color}]  [{MUTED}]({len(group)})[/{MUTED}]")
        for c in group:
            age = _age_str(_age_seconds(c.updated_at))
            imp_str = f" [{MUTED}]imp {c.importance}[/{MUTED}]" if c.importance != 3 else ""
            console.print(f"    [{BODY}]{c.title[:60]}[/{BODY}]  [{MUTED}]{age}[/{MUTED}]{imp_str}")
        console.print()

    if save:
        # Build narrative from chunk titles grouped by type
        parts = []
        for t in type_order:
            group = by_type.get(t, [])
            if not group:
                continue
            titles = ", ".join(f'"{c.title}"' for c in group[:3])
            suffix = f" (and {len(group)-3} more)" if len(group) > 3 else ""
            parts.append(f"{t}: {titles}{suffix}")
        narrative = ". ".join(parts) + "."

        with _spinner("Saving handoff chunk") as status:
            db.save_handoff(project_id, narrative)
            db.export_ndjson(project_id)
        console.print(f"[{GREEN}]✓[/{GREEN}] Handoff saved as preference chunk")
    else:
        console.print(f"  [{MUTED}]· run with --save to store as a handoff chunk for next session[/{MUTED}]")


@app.command("export", rich_help_panel="[bold #4ade80]Data[/bold #4ade80]")
def export_cmd(project: Optional[str] = typer.Option(None, "--project")):
    """Export chunks to [bold].granum/chunks.ndjson[/bold]."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    with _spinner("Writing chunks.ndjson") as status:
        db = _get_db(config, granum_dir)
        db.export_ndjson(project_id)

    console.print(f"[{GREEN}]✓[/{GREEN}] Exported to [{ORANGE}].granum/chunks.ndjson[/{ORANGE}]")


@app.command("import", rich_help_panel="[bold #4ade80]Data[/bold #4ade80]")
def import_cmd(project: Optional[str] = typer.Option(None, "--project")):
    """Import chunks from [bold].granum/chunks.ndjson[/bold]."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)

    with _spinner("Reading chunks.ndjson") as status:
        db = _get_db(config, granum_dir)
        status.update("Importing to database")
        count = db.import_ndjson(force=True)

    console.print(f"[{GREEN}]✓[/{GREEN}] Imported [{ORANGE}]{count}[/{ORANGE}] chunk(s)")


specs_app = typer.Typer(
    help="Manage [bold #4ade80]spec paths[/bold #4ade80] — source files indexed as read-only context chunks.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(specs_app, name="specs", rich_help_panel="[bold #4ade80]Specs[/bold #4ade80]")


@specs_app.command("list")
def specs_list():
    """List configured [bold]spec paths[/bold]."""
    granum_dir = _find_granum_dir()
    config = _load_config(granum_dir)
    paths = config.get("spec_paths", [])
    if not paths:
        console.print(f"[{MUTED}]· No spec paths configured[/{MUTED}]")
        return

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style=f"bold {ORANGE}", border_style=GRAY)
    t.add_column("Path", min_width=30)
    t.add_column("Status", width=10)
    t.add_column("Files", width=6)
    for p in paths:
        full = Path.cwd() / p
        exists = full.exists()
        if not exists:
            status = f"[{AMBER}]⚠ missing[/{AMBER}]"
            file_count = f"[dim {MUTED}]—[/dim {MUTED}]"
        else:
            status = f"[{GREEN}]✓ ok[/{GREEN}]"
            if full.is_dir():
                n = len(list(full.rglob("*.md")))
                file_count = f"[{MUTED}]{n}[/{MUTED}]"
            else:
                file_count = f"[{MUTED}]1[/{MUTED}]"
        t.add_row(f"[{BODY}]{p}[/{BODY}]", status, file_count)
    console.print(t)
    console.print(f"[dim {MUTED}]{len(paths)} path(s)[/dim {MUTED}]")


@specs_app.command("add")
def specs_add(path: str = typer.Argument(..., help="File or directory to add as spec source")):
    """[bold]Add[/bold] a spec path and re-index it immediately."""
    granum_dir = _find_granum_dir()
    config = _load_config(granum_dir)
    paths = config.get("spec_paths", [])

    if path in paths:
        console.print(f"[{MUTED}]· Already in spec paths: {path}[/{MUTED}]")
        return

    full = Path.cwd() / path
    if not full.exists():
        console.print(f"[{AMBER}]⚠ Path not found: {path} (adding anyway)[/{AMBER}]")

    paths.append(path)
    config["spec_paths"] = paths
    _save_config(granum_dir, config)
    console.print(f"[{GREEN}]✓[/{GREEN}] Added [{ORANGE}]{path}[/{ORANGE}]")

    # Trigger re-index via IPC if server running, else direct
    with _spinner("Contacting MCP server") as status:
        result = _ipc_call(granum_dir, "reindex_specs", {})
        if result is None:
            # Server not running — index directly
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from mcp_server.server import _chunk_by_section
            status.update("Loading database")
            db = _get_db(config, granum_dir)
            project_id = config["project_id"]
            db.clear_spec_chunks(project_id)
            for sp in paths:
                sp_path = Path.cwd() / sp
                if not sp_path.exists():
                    continue
                files = list(sp_path.rglob("*.md")) if sp_path.is_dir() else [sp_path]
                for f in files:
                    try:
                        status.update(f"Chunking {f.name}")
                        text = f.read_text(errors="replace")
                        rel = str(f.relative_to(Path.cwd()))
                        status.update(f"Embedding {f.name}")
                        db.index_spec_file(project_id, rel, _chunk_by_section(text, rel))
                    except Exception:
                        pass
            console.print(f"[{MUTED}]· Server not running — indexed directly (re-index on next session)[/{MUTED}]")
        else:
            console.print(f"[{GREEN}]✓[/{GREEN}] Re-indexed via MCP server ({result.get('indexed', 0)} file(s))")


@specs_app.command("reindex")
def specs_reindex():
    """[bold]Re-index[/bold] all spec paths — useful after adding files mid-session."""
    granum_dir = _find_granum_dir()
    config = _load_config(granum_dir)

    with _spinner("Contacting MCP server") as status:
        result = _ipc_call(granum_dir, "reindex_specs", {})
        if result is not None:
            status.update(f"Re-indexing {len(config.get('spec_paths', []))} path(s)")

    if result is None:
        console.print(f"[{AMBER}]⚠ MCP server not running — specs will re-index on next session start[/{AMBER}]")
    else:
        console.print(f"[{GREEN}]✓[/{GREEN}] Re-indexed [{ORANGE}]{result.get('indexed', 0)}[/{ORANGE}] spec file(s)")


@specs_app.command("remove")
def specs_remove(path: str = typer.Argument(...)):
    """[bold]Remove[/bold] a spec path from config."""
    granum_dir = _find_granum_dir()
    config = _load_config(granum_dir)
    paths = config.get("spec_paths", [])

    if path not in paths:
        console.print(f"[{RED}]✗ Not in spec paths: {path}[/{RED}]")
        raise typer.Exit(1)

    paths.remove(path)
    config["spec_paths"] = paths
    _save_config(granum_dir, config)
    console.print(f"[{GREEN}]✓[/{GREEN}] Removed [{ORANGE}]{path}[/{ORANGE}]")
    console.print(f"[{MUTED}]· Spec chunks will be cleared on next session start[/{MUTED}]")


server_app = typer.Typer(
    help="[bold]MCP server[/bold] management. [dim]Debug only — Claude Code manages the server in normal usage.[/dim]",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(server_app, name="server", rich_help_panel="[bold #4ade80]Debug[/bold #4ade80]")


@server_app.command("status")
def server_status():
    """Show MCP server status."""
    pid_file = Path.cwd() / ".granum" / "server.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        # Check if process alive
        try:
            os.kill(int(pid), 0)
            console.print(f"[{GREEN}]MCP server: running[/{GREEN}]")
            console.print(f"[{MUTED}]PID:        {pid}[/{MUTED}]")
        except (ProcessLookupError, ValueError):
            console.print(f"[{AMBER}]MCP server: not running (stale PID file)[/{AMBER}]")
    else:
        console.print(f"[{MUTED}]MCP server: managed by Claude Code (normal)[/{MUTED}]")
        console.print(f"[{MUTED}]Run 'granum server start' only for debugging.[/{MUTED}]")


@server_app.command("start")
def server_start():
    """Start MCP server manually (debug only)."""
    plugin_dir = Path(__file__).parent.parent
    pid_file = plugin_dir / ".granum" / "server.pid"

    console.print(f"[{AMBER}]⚠ Debug mode — Claude Code manages the server in normal usage.[/{AMBER}]")
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=str(plugin_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid_file.parent.mkdir(exist_ok=True)
    pid_file.write_text(str(proc.pid))
    console.print(f"[{GREEN}]✓[/{GREEN}] Started (PID {proc.pid}). Logs: granum server logs")


@server_app.command("stop")
def server_stop():
    """Stop manually started MCP server."""
    plugin_dir = Path(__file__).parent.parent
    pid_file = plugin_dir / ".granum" / "server.pid"

    if not pid_file.exists():
        console.print(f"[{MUTED}]· No PID file found[/{MUTED}]")
        return

    pid = int(pid_file.read_text().strip())
    try:
        import signal
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink()
        console.print(f"[{GREEN}]✓[/{GREEN}] Stopped (PID {pid})")
    except ProcessLookupError:
        pid_file.unlink()
        console.print(f"[{MUTED}]· Process {pid} already stopped[/{MUTED}]")


@server_app.command("logs")
def server_logs():
    """Tail MCP server stderr log."""
    log_file = Path.cwd() / ".granum" / "server.log"
    if not log_file.exists():
        console.print(f"[{MUTED}]· No server log at .granum/server.log[/{MUTED}]")
        return
    subprocess.run(["tail", "-f", str(log_file)])


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_SHIMMER_BRIGHT = (0xd4, 0xea, 0xd9)   # very light mint — highlight peak
_SHIMMER_DIM = (0x6a, 0x9e, 0x78)     # muted sage — base
_SHIMMER_WIDTH = 5.0


def _shimmer_text(message: str, pos: float):
    from rich.text import Text
    t = Text()
    for i, ch in enumerate(message):
        blend = min(abs(i - pos) / _SHIMMER_WIDTH, 1.0)
        r = int(_SHIMMER_BRIGHT[0] + (_SHIMMER_DIM[0] - _SHIMMER_BRIGHT[0]) * blend)
        g = int(_SHIMMER_BRIGHT[1] + (_SHIMMER_DIM[1] - _SHIMMER_BRIGHT[1]) * blend)
        b = int(_SHIMMER_BRIGHT[2] + (_SHIMMER_DIM[2] - _SHIMMER_BRIGHT[2]) * blend)
        t.append(ch, style=f"#{r:02x}{g:02x}{b:02x}")
    return t


_ARC_FRAMES = ["◜", "◠", "◝", "◞", "◡", "◟"]


class _SpinnerCtx:
    """Arc + text unified shimmer via Live."""
    def __init__(self, message: str):
        self._message = message
        self._live = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _render(self, step: int):
        arc = _ARC_FRAMES[step % len(_ARC_FRAMES)]
        full = arc + "  " + self._message
        span = len(full) + int(_SHIMMER_WIDTH) * 2
        pos = (step % span) - _SHIMMER_WIDTH
        return _shimmer_text(full, pos)

    def __enter__(self):
        from rich.live import Live
        self._live = Live(self._render(0), console=console, refresh_per_second=20, transient=True)
        self._live.__enter__()
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def update(self, msg: str) -> None:
        self._message = msg

    def _animate(self) -> None:
        step = 0
        while not self._stop.wait(0.06):
            if self._live:
                self._live.update(self._render(step))
            step += 1

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._live:
            self._live.__exit__(*args)


def _spinner(message: str = "Loading") -> _SpinnerCtx:
    return _SpinnerCtx(message)


def _dir_size(path: Path) -> str:
    if not path.exists():
        return "0 B"
    total = path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _find_orphans(chunks, edges: list[dict]) -> list:
    """Chunks with no edges — likely isolated/stale."""
    connected: set[str] = set()
    for e in edges:
        connected.add(e["from_id"])
        connected.add(e["to_id"])
    return [c for c in chunks if c.id not in connected]


def _write_mcp_json(cwd: Path) -> None:
    mcp_path = cwd / ".mcp.json"
    existing: dict = {}
    if mcp_path.exists():
        try:
            existing = json.loads(mcp_path.read_text())
        except Exception:
            pass

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["granum"] = {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "mcp_server.server"],
        "env": {"GRANUM_CWD": str(cwd)},
        "alwaysLoad": True,
    }
    mcp_path.write_text(json.dumps(existing, indent=2) + "\n")
    console.print(f"[{GREEN}]✓[/{GREEN}] MCP server registered in [{ORANGE}].mcp.json[/{ORANGE}]")


def _write_claude_settings(cwd: Path, write_hooks: bool) -> None:
    claude_dir = cwd / ".claude"
    claude_dir.mkdir(exist_ok=True)
    settings_path = claude_dir / "settings.json"

    existing: dict = {}
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text())
        except Exception:
            pass

    hooks_dir = Path(__file__).parent.parent / "hooks"

    if write_hooks:
        existing.setdefault("hooks", {})
        hooks = existing["hooks"]

        def _set_hook(event: str, matcher: Optional[str], script: str) -> None:
            entry = {"hooks": [{"type": "command", "command": f"bash {hooks_dir / script}"}]}
            if matcher:
                entry["matcher"] = matcher
            existing_list = hooks.setdefault(event, [])
            # Remove any existing granum entry for this event+matcher to avoid dupes
            hooks[event] = [
                h for h in existing_list
                if not any(
                    "granum" in hook.get("command", "")
                    for hook in h.get("hooks", [])
                )
            ]
            hooks[event].append(entry)

        _set_hook("UserPromptSubmit", None, "granum-log.sh")
        _set_hook("SessionStart", "startup", "granum-coldstart.sh")
        _set_hook("SessionStart", "compact", "granum-reinject.sh")

        console.print(f"[{GREEN}]✓[/{GREEN}] Hooks registered in [{ORANGE}].claude/settings.json[/{ORANGE}]")
    else:
        console.print(f"[{AMBER}]⚠ Skipped hooks (bash not available)[/{AMBER}]")

    settings_path.write_text(json.dumps(existing, indent=2) + "\n")


def main():
    # Check Python version
    if sys.version_info < (3, 9):
        console.print(f"[{RED}]✗ Granum requires Python 3.9+. Found: Python {sys.version.split()[0]}[/{RED}]")
        sys.exit(1)
    app()


if __name__ == "__main__":
    main()
