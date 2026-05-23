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
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn  # Progress/SpinnerColumn used by timeline
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box


_ru.STYLE_COMMANDS_TABLE_FIRST_COLUMN = "#fe9785"   # light coral — command names (palette accent)
_ru.STYLE_OPTION = "#f27059"                        # coral — long flags
_ru.STYLE_SWITCH = "#f27059"
_ru.STYLE_NEGATIVE_OPTION = "#4a6d7c"               # teal — negative flags
_ru.STYLE_NEGATIVE_SWITCH = "#4a6d7c"
_ru.STYLE_METAVAR = "#4a6d7c"                       # teal — type hints
_ru.STYLE_OPTION_DEFAULT = "#4a6d7c"
_ru.STYLE_OPTION_ENVVAR = "#4a6d7c"
_ru.STYLE_OPTIONS_PANEL_BORDER = "#4a6d7c"          # teal border
_ru.STYLE_COMMANDS_PANEL_BORDER = "#4a6d7c"


app = typer.Typer(
    help=(
        "[bold #f27059]granum[/bold #f27059]  persistent semantic memory for Claude Code\n\n"
        "Granum stores decisions, preferences, constraints and file state as vector chunks "
        "that Claude retrieves automatically at the start of each turn."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# Brand colors
ORANGE = "#f27059"   # coral
GREEN = "#22c55e"   # green — success
AMBER = "#f59e0b"   # amber — warnings
RED = "#ef4444"     # red — errors
GRAY = "#4a6d7c"    # teal
BODY = "#c6d8d3"    # sage
MUTED = "#4a6d7c"   # teal

TYPE_ICONS = {
    "decision": "◆",
    "constraint": "▲",
    "preference": "★",
    "file_state": "▪",
    "spec": "◇",
}
STATUS_ICONS = {"active": "✓", "deprecated": "○", "deleted": "×"}


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
        db_path=granum_dir / "db",
        ndjson_path=granum_dir / "chunks.ndjson",
        stale_threshold_days=config.get("stale_threshold_days", 7),
    )


def _ipc_call(granum_dir: Path, method: str, params: dict) -> Optional[any]:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mcp_server.ipc import ipc_call
    return ipc_call(granum_dir, method, params)


def _ipc_query(granum_dir: Path, query: str, config: dict) -> Optional[list]:
    return _ipc_call(granum_dir, "query_context", {
        "query": query,
        "memory_limit": config.get("memory_retrieval_limit", 7),
        "spec_limit": config.get("spec_retrieval_limit", 3),
    })


def _ipc_chunks(granum_dir: Path, project_id: str, include_deprecated: bool = False) -> Optional[list]:
    return _ipc_call(granum_dir, "list_chunks", {
        "project_id": project_id,
        "include_deprecated": include_deprecated,
    })


def _ipc_spec_chunks(granum_dir: Path, project_id: str) -> Optional[list]:
    return _ipc_call(granum_dir, "list_spec_chunks", {"project_id": project_id})


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
        _status.update("Warming up database")
    db = _get_db(config, granum_dir)
    db.import_ndjson()
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


def _add_chunk_row(table: Table, chunk, stale_threshold: int) -> None:
    icon = TYPE_ICONS.get(chunk.type, "·")
    secs = _age_seconds(chunk.updated_at)
    age_days = _age_days_from_seconds(secs)
    stale_color = _stale_style(age_days, stale_threshold)

    age_label = _age_str(secs)
    if stale_color:
        badge = " ⚠ STALE" if age_days <= 14 else " ⚠ VERY STALE"
        age_cell = f"[{stale_color}]{age_label}{badge}[/{stale_color}]"
    else:
        age_cell = f"[{MUTED}]{age_label}[/{MUTED}]"

    if chunk.status == "deprecated":
        row_style = f"dim {GRAY}"
        status_cell = f"[{GRAY}]○ deprecated[/{GRAY}]"
        title_cell = f"[italic {GRAY}]{chunk.title}[/italic {GRAY}]"
    else:
        row_style = BODY
        status_cell = f"[{GREEN}]✓[/{GREEN}]"
        title_cell = f"[{BODY}]{chunk.title}[/{BODY}]"

    table.add_row(
        f"[{MUTED}]{chunk.id[:8]}[/{MUTED}]",
        f"[{ORANGE}]{icon}[/{ORANGE}] [{MUTED}]{chunk.type}[/{MUTED}]",
        title_cell,
        age_cell,
        f"[{ORANGE}]{chunk.importance}[/{ORANGE}]",
        status_cell,
    )


def _make_chunk_table(title: str = "") -> Table:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style=f"bold {ORANGE}",
        title=title or None,
        title_style=f"bold {ORANGE}",
    )
    table.add_column("ID", style=MUTED, width=9)
    table.add_column("Type", style=BODY, width=18)
    table.add_column("Title", style=BODY, min_width=30)
    table.add_column("Age", width=14)
    table.add_column("Imp", width=4)
    table.add_column("Status", width=14)
    return table


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

@app.command(rich_help_panel="[bold #f27059]Setup[/bold #f27059]")
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
        "openspec/specs/", "docs/", "AGENTS.md", "CLAUDE.md",
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
    gitignore.write_text("session.log\ntool_call_count\n")

    # Wire MCP server into .mcp.json
    _write_mcp_json(cwd)

    # Wire hooks into .claude/settings.json
    _write_claude_settings(cwd, bash_ok)

    console.print(f"\n[{GREEN}]✓[/{GREEN}] Config written to [{ORANGE}].granum/config.json[/{ORANGE}]")
    console.print(f"[{MUTED}]Project: {git_root} (branch: {branch})[/{MUTED}]")
    console.print(f"[{MUTED}]Project ID: {project_id}[/{MUTED}]\n")


@app.command("config", rich_help_panel="[bold #f27059]Setup[/bold #f27059]")
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


@app.command("list", rich_help_panel="[bold #f27059]Memory[/bold #f27059]")
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

    if chunks:
        table = _make_chunk_table()
        for chunk in chunks:
            _add_chunk_row(table, chunk, stale_threshold)
        console.print(table)
        console.print(f"[{MUTED}]{len(chunks)} memory chunk(s)[/{MUTED}]")

    if spec_chunks:
        spec_table = _make_chunk_table("Spec chunks (read-only, re-indexed each session)")
        for chunk in spec_chunks:
            _add_chunk_row(spec_table, chunk, stale_threshold)

        console.print(spec_table)
        console.print(f"[{MUTED}]{len(spec_chunks)} spec chunk(s)[/{MUTED}]")


@app.command(rich_help_panel="[bold #f27059]Memory[/bold #f27059]")
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

    def _recent_table() -> Table:
        t = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {ORANGE}")
        t.add_column("Type", width=18)
        t.add_column("Title", min_width=30)
        t.add_column("Updated", width=16)
        return t

    if chunks:
        table = _recent_table()
        for chunk in chunks:
            icon = TYPE_ICONS.get(chunk.type, "·")
            secs = _age_seconds(chunk.updated_at)
            table.add_row(
                f"[{ORANGE}]{icon}[/{ORANGE}] [{MUTED}]{chunk.type}[/{MUTED}]",
                f"[{BODY}]{chunk.title}[/{BODY}]",
                f"[{MUTED}]{_age_str(secs)}[/{MUTED}]",
            )
        console.print(table)

    if spec_chunks:
        spec_table = _recent_table()
        for chunk in spec_chunks:
            spec_table.add_row(
                f"[{ORANGE}]◇[/{ORANGE}] [{MUTED}]spec[/{MUTED}]",
                f"[{BODY}]{chunk.title}[/{BODY}]",
                f"[{MUTED}]{chunk.source or ''}[/{MUTED}]",
            )
        console.print(spec_table)
        console.print(f"[{MUTED}]{len(spec_chunks)} spec chunk(s) — re-indexed each session[/{MUTED}]")


@app.command(rich_help_panel="[bold #f27059]Memory[/bold #f27059]")
def stale(project: Optional[str] = typer.Option(None, "--project")):
    """List [bold]stale[/bold] chunks that need review, oldest first."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]
    threshold = config.get("stale_threshold_days", 7)

    with _spinner("Checking chunk freshness") as status:
        chunks = _get_chunks(granum_dir, project_id, config, _status=status)

    stale_chunks = [c for c in chunks if _age_days_from_seconds(_age_seconds(c.updated_at)) > threshold]
    stale_chunks.sort(key=lambda c: c.updated_at)

    if not stale_chunks:
        console.print(f"[{GREEN}]✓ No stale chunks[/{GREEN}]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {ORANGE}")
    table.add_column("Type", width=18)
    table.add_column("Title", min_width=30)
    table.add_column("Age", width=10)
    table.add_column("Imp", width=4)

    for chunk in stale_chunks:
        icon = TYPE_ICONS.get(chunk.type, "·")
        age_days = _age_days_from_seconds(_age_seconds(chunk.updated_at))
        color = RED if age_days > 14 else AMBER
        table.add_row(
            f"[{ORANGE}]{icon}[/{ORANGE}] [{MUTED}]{chunk.type}[/{MUTED}]",
            f"[{BODY}]{chunk.title}[/{BODY}]",
            f"[{color}]{age_days}d[/{color}]",
            f"[{ORANGE}]{chunk.importance}[/{ORANGE}]",
        )

    console.print(table)
    console.print(f"[{AMBER}]⚠ {len(stale_chunks)} stale chunk(s)[/{AMBER}]")


@app.command(rich_help_panel="[bold #f27059]Memory[/bold #f27059]")
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
            status.update("Loading database")
            db = _get_db(config, granum_dir)
            db.import_ndjson()
            status.update("Running similarity search")
            results = db.query_context(
                project_id=project_id,
                query=query,
                memory_limit=config.get("memory_retrieval_limit", 7),
                spec_limit=config.get("spec_retrieval_limit", 3),
                freshness_decay_days=config.get("freshness_decay_days", 90),
            )

    if not results:
        console.print(f"[{MUTED}]· No results[/{MUTED}]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {ORANGE}")
    table.add_column("Type", width=18)
    table.add_column("Title", min_width=28)
    table.add_column("Score", width=7)
    table.add_column("Sim", width=7)
    table.add_column("Age", width=10)

    for r in results:
        icon = TYPE_ICONS.get(r["type"], "·")
        stale = f" [{AMBER}]⚠[/{AMBER}]" if r.get("stale_warning") else ""
        table.add_row(
            f"[{ORANGE}]{icon}[/{ORANGE}] [{MUTED}]{r['type']}[/{MUTED}]",
            f"[{BODY}]{r['title']}[/{BODY}]{stale}",
            f"[{ORANGE}]{r['final_score']:.2f}[/{ORANGE}]",
            f"[{MUTED}]{r['similarity']:.2f}[/{MUTED}]",
            f"[{MUTED}]{r['age']}[/{MUTED}]",
        )

    console.print(table)


@app.command(rich_help_panel="[bold #f27059]Memory[/bold #f27059]")
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

    with _spinner("Loading database") as status:
        db = _get_db(config, granum_dir)
        db.import_ndjson()
        status.update("Deleting chunk")
        ok = db.soft_delete(chunk_id)
        if ok:
            status.update("Saving to disk")
            db.export_ndjson(config["project_id"])

    if ok:
        console.print(f"[{GREEN}]✓[/{GREEN}] Deleted [{MUTED}]{chunk_id[:12]}[/{MUTED}]")
    else:
        console.print(f"[{RED}]✗ Chunk not found: {chunk_id}[/{RED}]")


@app.command(rich_help_panel="[bold #f27059]Analysis[/bold #f27059]")
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

    db_size = _dir_size(granum_dir / "db")

    console.print(f"\n[{ORANGE}]─ Granum Stats ─────────────────────────[/{ORANGE}]")
    console.print(f"[{MUTED}]Project:[/{MUTED}]   [{BODY}]{_git_root() or Path.cwd()} (branch: {_git_branch()})[/{BODY}]")
    console.print(f"[{MUTED}]Spec paths:[/{MUTED}] [{BODY}]{', '.join(config.get('spec_paths', [])) or 'none'}[/{BODY}]")
    console.print(f"[{MUTED}]Memory:[/{MUTED}]    [{GREEN}]{len(active)} active[/{GREEN}], [{GRAY}]{len(deprecated)} deprecated[/{GRAY}]")
    console.print(f"[{MUTED}]Specs:[/{MUTED}]     [{BODY}]{len(spec_chunks)} chunk(s) indexed[/{BODY}]")
    console.print(f"[{MUTED}]Stale:[/{MUTED}]     [{AMBER}]{len(stale)}[/{AMBER}]")
    console.print(f"[{MUTED}]DB size:[/{MUTED}]   [{BODY}]{db_size}[/{BODY}]")
    console.print(f"[{MUTED}]Embedding:[/{MUTED}] [{BODY}]{config.get('embedding_model', 'all-MiniLM-L6-v2')}[/{BODY}]")
    console.print()


@app.command(rich_help_panel="[bold #f27059]Analysis[/bold #f27059]")
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
        status.update("Checking for stale chunks")
        active = [c for c in all_chunks if c.status == "active"]

    deprecated = [c for c in all_chunks if c.status == "deprecated"]
    stale = [c for c in active if _age_days_from_seconds(_age_seconds(c.updated_at)) > threshold]
    very_stale = [c for c in active if _age_days_from_seconds(_age_seconds(c.updated_at)) > 30]
    low_value = [c for c in active if c.importance <= 2 and _age_days_from_seconds(_age_seconds(c.updated_at)) > threshold]

    console.print(f"\n[{ORANGE}]─ Granum Audit ─────────────────────────[/{ORANGE}]")
    console.print(f"[{MUTED}]Project:[/{MUTED}]      [{BODY}]{_git_root() or Path.cwd()} (branch: {_git_branch()})[/{BODY}]")
    console.print(f"Active:       [{GREEN}]{len(active)} chunks[/{GREEN}]")
    console.print(f"Specs:        [{BODY}]{len(spec_chunks)} chunk(s) indexed[/{BODY}]")
    console.print(f"Deprecated:   [{GRAY}]{len(deprecated)} chunks[/{GRAY}]")
    console.print(f"Stale:        [{AMBER}]{len(stale)} chunks[/{AMBER}]  (>{threshold}d)")
    console.print(f"Very stale:   [{RED}]{len(very_stale)} chunk(s)[/{RED}]  (>30d)")
    console.print(f"Low value:    [{MUTED}]{len(low_value)} chunk(s)[/{MUTED}]  (importance 1-2, stale)")

    # Possible duplicates (same type, similar title prefix)
    duplicates = _find_possible_duplicates(active)
    if duplicates:
        console.print(f"\n[{AMBER}]Possible duplicates (similar titles, same type):[/{AMBER}]")
        for a, b in duplicates:
            console.print(f"  [{MUTED}]{a.id[:4]}[/{MUTED}]  [{BODY}]\"{a.title}\"[/{BODY}]")
            console.print(f"  [{MUTED}]{b.id[:4]}[/{MUTED}]  [{BODY}]\"{b.title}\"[/{BODY}]  [{MUTED}]— consider merging[/{MUTED}]")

    if low_value:
        console.print(f"\n[{MUTED}]Recommendation: run cleanup_context on {len(low_value)} low-value stale chunk(s).[/{MUTED}]")

    console.print()


@app.command(rich_help_panel="[bold #f27059]Analysis[/bold #f27059]")
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
        status.update("Mapping activity to calendar")

    # Count saves per day
    day_counts: dict[str, int] = {}
    for chunk in all_chunks:
        try:
            day_counts[chunk.updated_at[:10]] = day_counts.get(chunk.updated_at[:10], 0) + 1
        except Exception:
            pass

    today = date.today()
    max_count = max(day_counts.values()) if day_counts else 1

    def _cell(d: Optional[date]) -> str:
        if d is None:
            return "  "
        count = day_counts.get(str(d), 0)
        if count == 0:
            return f"[{MUTED}]░░[/{MUTED}]"
        intensity = count / max_count
        if intensity < 0.33:
            return f"[{MUTED}]▒▒[/{MUTED}]"
        elif intensity < 0.66:
            return f"[{AMBER}]▓▓[/{AMBER}]"
        else:
            return f"[{ORANGE}]██[/{ORANGE}]"

    console.print(f"\n  [{ORANGE}]Granum activity — {Path(_git_root() or '.').name} ({_git_branch()})[/{ORANGE}]")

    # Render each month
    for m_offset in range(months - 1, -1, -1):
        year = today.year
        month = today.month - m_offset
        while month <= 0:
            month += 12
            year -= 1

        month_name = date(year, month, 1).strftime("%B %Y")
        _, days_in_month = calendar.monthrange(year, month)
        first_dow = date(year, month, 1).weekday()  # 0=Mon

        console.print(f"\n  [{ORANGE}]{month_name}[/{ORANGE}]")
        console.print(f"  [{MUTED}]Mo Tu We Th Fr Sa Su[/{MUTED}]")

        # Build weeks
        cells: list[Optional[date]] = [None] * first_dow
        for day in range(1, days_in_month + 1):
            cells.append(date(year, month, day))
        # Pad to full weeks
        while len(cells) % 7 != 0:
            cells.append(None)

        for week_start in range(0, len(cells), 7):
            week = cells[week_start:week_start + 7]
            row = "  "
            for d in week:
                row += _cell(d) + " "
            console.print(row)

    total_saves = sum(day_counts.values())
    active_days = len(day_counts)
    console.print(f"\n  [{MUTED}]░ none  ▒ light  ▓ medium  █ heavy    {total_saves} save(s) across {active_days} active day(s)[/{MUTED}]\n")


@app.command(rich_help_panel="[bold #f27059]Memory[/bold #f27059]")
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

    with _spinner("Loading database") as status:
        db = _get_db(config, granum_dir)
        db.import_ndjson()
        chunks = db.get_all_memory_chunks(project_id, include_deprecated=True)
        status.update(f"Deleting {len(chunks)} chunks")
        for chunk in chunks:
            db.soft_delete(chunk.id)
        status.update("Saving to disk")
        db.export_ndjson(project_id)

    console.print(f"[{GREEN}]✓[/{GREEN}] Cleared [{ORANGE}]{len(chunks)}[/{ORANGE}] chunk(s)")


@app.command("export", rich_help_panel="[bold #f27059]Data[/bold #f27059]")
def export_cmd(project: Optional[str] = typer.Option(None, "--project")):
    """Export chunks to [bold].granum/chunks.ndjson[/bold]."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)
    project_id = config["project_id"]

    with _spinner("Loading database") as status:
        db = _get_db(config, granum_dir)
        db.import_ndjson()
        status.update("Writing chunks.ndjson")
        db.export_ndjson(project_id)

    console.print(f"[{GREEN}]✓[/{GREEN}] Exported to [{ORANGE}].granum/chunks.ndjson[/{ORANGE}]")


@app.command("import", rich_help_panel="[bold #f27059]Data[/bold #f27059]")
def import_cmd(project: Optional[str] = typer.Option(None, "--project")):
    """Import chunks from [bold].granum/chunks.ndjson[/bold]."""
    granum_dir = _find_granum_dir(Path(project) if project else None)
    config = _load_config(granum_dir)

    with _spinner("Reading chunks.ndjson") as status:
        db = _get_db(config, granum_dir)
        status.update("Importing to database")
        count = db.import_ndjson()

    console.print(f"[{GREEN}]✓[/{GREEN}] Imported [{ORANGE}]{count}[/{ORANGE}] chunk(s)")


specs_app = typer.Typer(
    help="Manage [bold #f27059]spec paths[/bold #f27059] — source files indexed as read-only context chunks.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(specs_app, name="specs", rich_help_panel="[bold #f27059]Specs[/bold #f27059]")


@specs_app.command("list")
def specs_list():
    """List configured [bold]spec paths[/bold]."""
    granum_dir = _find_granum_dir()
    config = _load_config(granum_dir)
    paths = config.get("spec_paths", [])
    if not paths:
        console.print(f"[{MUTED}]· No spec paths configured[/{MUTED}]")
        return
    for p in paths:
        exists = (Path.cwd() / p).exists()
        icon = f"[{GREEN}]✓[/{GREEN}]" if exists else f"[{AMBER}]⚠[/{AMBER}]"
        console.print(f"  {icon} [{BODY}]{p}[/{BODY}]")


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
app.add_typer(server_app, name="server", rich_help_panel="[bold #f27059]Debug[/bold #f27059]")


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

_SHIMMER_BRIGHT = (0xfe, 0x97, 0x85)   # light coral
_SHIMMER_DIM = (0xf2, 0x70, 0x59)     # coral
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
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _find_possible_duplicates(chunks) -> list[tuple]:
    seen: dict[str, list] = {}
    for chunk in chunks:
        key = chunk.type
        seen.setdefault(key, []).append(chunk)

    pairs = []
    for type_chunks in seen.values():
        for i, a in enumerate(type_chunks):
            for b in type_chunks[i + 1:]:
                # Simple heuristic: share 3+ words
                words_a = set(a.title.lower().split())
                words_b = set(b.title.lower().split())
                if len(words_a & words_b) >= 3:
                    pairs.append((a, b))
    return pairs[:5]  # cap at 5 shown


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
        _set_hook("Stop", None, "granum-compact.sh")
        _set_hook("SessionStart", "startup", "granum-coldstart.sh")
        _set_hook("SessionStart", "compact", "granum-reinject.sh")
        _set_hook("PostToolUse", "Edit|Write", "granum-spec-sync.sh")

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
