"""Generate README.md files for experiment leaf folders.

Descriptions come from module docstrings so source files remain the single source
of truth. The experiment tree may be nested by responsibility; every directory
that directly contains experiment scripts receives its own README.
"""
import ast
import os
import pathlib
import re
import subprocess
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


# The top-level README's link bar, repeated on every page so each one reaches
# the same six entry points.
NAV_BAR = [
    ("Architecture", "docs/methods/architecture/teacher-architecture.md"),
    ("Train and run", "scripts/rl/README.md"),
    ("Experiments", "scripts/rl/experiments/README.md"),
    ("Research", "docs/README.md"),
    ("RL core", "scripts/rl/core/README.md"),
    ("Package", "talon_rl/README.md"),
]
CRUMB_LABELS = {
    ".": "TALON RL",
    "scripts/rl": "RL runner",
    "scripts/rl/core": "RL core",
    "docs": "Documentation",
}
NAV_RE = re.compile(r"<!-- nav:start -->.*?<!-- nav:end -->", re.S)


def crumb_label(folder):
    key = folder.as_posix()
    return CRUMB_LABELS.get(key) or folder.name.replace("_", " ").replace("-", " ").title()


def nav_block(page):
    """Link bar plus a breadcrumb from the repo root, for any page in the repo.

    Paths are repo-relative, so run from the repo root. Every crumb is a link,
    the last one to the page's own folder README.
    """
    page = pathlib.Path(page)
    folder = page.parent
    bar = " · ".join(
        f"[{label}]({os.path.relpath(target, start=folder)})" for label, target in NAV_BAR
    )
    chain = [pathlib.Path(".")] + [pathlib.Path(*folder.parts[:i + 1]) for i in range(len(folder.parts))]
    crumbs = []
    for d in chain:
        # The page's own folder always gets a crumb, even before its README exists.
        if d == folder or (d / "README.md").exists():
            crumbs.append(f"[{crumb_label(d)}]({os.path.relpath(d / 'README.md', start=folder)})")
    return ["<!-- nav:start -->", bar, "", " · ".join(crumbs), "<!-- nav:end -->"]


# A subfolder named at the start of a bullet or table row: "- name/", "- `name/`"
# or "| `name/` |". Already-linked entries start with "[" and never match.
CHILD_RE = re.compile(r"^(- |\| )(`?)([\w.-]+)/\2(?=[\s:|]|$)", re.M)


def link_children(page, text):
    """Turn subfolder names in a page's lists into links to their READMEs."""
    def link(m):
        lead, tick, name = m.groups()
        if not (page.parent / name / "README.md").exists():
            return m.group(0)
        return f"{lead}[{tick}{name}/{tick}]({name}/README.md)"
    # Split on code fences so folder trees inside ``` blocks stay untouched.
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    return "".join(p if p.startswith("```") else CHILD_RE.sub(link, p) for p in parts)


def rewrite_nav(page):
    """Rewrite the page's nav block and link its subfolder list; True if changed."""
    text = page.read_text()
    new = NAV_RE.sub(lambda _: "\n".join(nav_block(page)), text, count=1)
    new = link_children(page, new)
    if new != text:
        page.write_text(new)
    return new != text


def folder_readme(folder):
    files = sorted(p for p in folder.glob("*.py") if p.name != "__init__.py")
    if not files:
        return None
    rel = folder.relative_to(D)
    lines = [
        f"# `{rel}`",
        "",
        *nav_block(folder / "README.md"),
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
    if "--nav" in sys.argv:
        # Every hand-written page with a nav block, not just the generated ones.
        out = subprocess.run(["git", "ls-files", "-z", "*.md"], capture_output=True, text=True, check=True).stdout
        pages = [pathlib.Path(p) for p in out.split("\0") if p]
        changed = [p for p in pages if "<!-- nav:start -->" in p.read_text() and rewrite_nav(p)]
        print(f"rewrote nav in {len(changed)} pages")
        sys.exit(0)
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
