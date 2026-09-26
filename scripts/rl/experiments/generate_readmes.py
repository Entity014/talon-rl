"""Generate README.md files for experiment leaf folders.

Descriptions come from module docstrings so source files remain the single source
of truth. The experiment tree may be nested by responsibility; every directory
that directly contains experiment scripts receives its own README.
"""
import ast
import os
import pathlib
import re
import sys

D = pathlib.Path("scripts/rl/experiments")
RUN_RE = re.compile(r'["\'](?:runs/)?([A-Za-z0-9_]+-20\d\d-\d\d-\d\d)')


def facts(p):
    src = p.read_text()
    try:
        doc = ast.get_docstring(ast.parse(src))
    except SyntaxError:
        doc = None
    summary = ""
    if doc:
        para = []
        for line in doc.strip().splitlines():
            if not line.strip():
                break
            para.append(line.strip())
        summary = " ".join(para).replace("|", "\\|")
    tags = []
    if "AppLauncher" in src:
        tags.append("isaac")
    if re.search(r"\.backward\(\)|optim\.(Adam|SGD)", src):
        tags.append("trains")
    if re.search(r"^from rl\.core\.", src, re.M):
        tags.append("class")
    runs = sorted(set(RUN_RE.findall(src)))
    return summary, tags, runs, len(src.splitlines())


def nav_block(folder):
    """Build a local breadcrumb for an experiment README."""
    parts = []
    root = pathlib.Path("scripts/rl/experiments")
    parts.append(
        "[Experiments]("
        + os.path.relpath(root / "README.md", start=folder)
        + ")"
    )

    rel = folder.relative_to(root)
    current = root
    for name in rel.parts[:-1]:
        current = current / name
        readme = current / "README.md"
        if readme.exists():
            label = name.replace("_", " ").replace("-", " ").title()
            parts.append(
                f"[{label}]({os.path.relpath(readme, start=folder)})"
            )

    return ["<!-- nav:start -->", " · ".join(parts), "<!-- nav:end -->"]


def folder_readme(folder):
    files = sorted(p for p in folder.glob("*.py") if p.name != "__init__.py")
    if not files:
        return None
    rel = folder.relative_to(D)
    lines = [
        f"# `{rel}`",
        "",
        *nav_block(folder),
        "",
        f"{len(files)} scripts. One line each, taken from the file's own docstring — "
        "edit the docstring, not this file.",
        "",
        "| file | lines | tags | description |",
        "|---|---:|---|---|",
    ]
    undocumented = []
    runs = set()
    for p in files:
        summary, tags, r, n = facts(p)
        runs |= set(r)
        if not summary:
            undocumented.append(p.name)
        lines.append(
            f"| `{p.name}` | {n} | {' '.join(tags) or '—'} | "
            f"{summary or '**no docstring**'} |"
        )
    if runs:
        lines += ["", "## Run directories these touch", ""]
        lines += [f"- `runs/{r}`" for r in sorted(runs)]
    if undocumented:
        lines += [
            "",
            f"## Still undescribed ({len(undocumented)})",
            "",
            "These have no module docstring, so there is nothing to put in the table above.",
            "",
        ]
        lines += [f"- `{n}`" for n in undocumented]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    only = [a.strip("/") for a in sys.argv[1:] if not a.startswith("-")]
    folders = sorted(
        p for p in D.rglob("*")
        if p.is_dir() and p.name != "__pycache__" and any(
            x.suffix == ".py" and x.name != "__init__.py" for x in p.iterdir()
        )
    )
    for folder in folders:
        rel = str(folder.relative_to(D))
        if only and rel not in only and folder.name not in only:
            continue
        text = folder_readme(folder)
        if text is None:
            continue
        if apply:
            (folder / "README.md").write_text(text)
        else:
            print(text)
