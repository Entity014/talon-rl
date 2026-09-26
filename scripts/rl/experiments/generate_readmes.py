"""Write one README.md per experiment folder, from what the files declare.

The description of a script is its module docstring, so the file stays the
single source of truth and the README cannot drift from it. Everything else —
whether it needs Isaac Sim, whether it trains, which run directories it touches
— is read from the source.
"""
import ast
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
    # the first paragraph, not the first line: a one-sentence summary often
    # wraps, and cutting it at the newline reads as a truncation
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


def folder_readme(folder):
    files = sorted(p for p in folder.glob("*.py") if p.name != "__init__.py")
    if not files:
        return None
    lines = [f"# `{folder.name}`", "",
             f"{len(files)} scripts. One line each, taken from the file's own docstring — "
             "edit the docstring, not this file.", "",
             "| file | lines | tags | description |",
             "|---|---:|---|---|"]
    undocumented = []
    runs = set()
    for p in files:
        summary, tags, r, n = facts(p)
        runs |= set(r)
        if not summary:
            undocumented.append(p.name)
        lines.append(f"| `{p.name}` | {n} | {' '.join(tags) or '—'} | "
                     f"{summary or '**no docstring**'} |")
    if runs:
        lines += ["", "## Run directories these touch", ""]
        lines += [f"- `runs/{r}`" for r in sorted(runs)]
    if undocumented:
        lines += ["", f"## Still undescribed ({len(undocumented)})", "",
                  "These have no module docstring, so there is nothing to put in the table above.",
                  ""]
        lines += [f"- `{n}`" for n in undocumented]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for folder in sorted(x for x in D.iterdir() if x.is_dir() and x.name != "__pycache__"):
        if only and folder.name not in only:
            continue
        text = folder_readme(folder)
        if text is None:
            continue
        if apply:
            (folder / "README.md").write_text(text)
        else:
            print(text)
