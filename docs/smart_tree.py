#!/usr/bin/env python3
"""
smart-tree.py — tree that respects .gitignore intelligently.

Recurses into non-ignored directories as usual. For ignored directories,
prints a single summary line with counts and a few sample filenames, so
you see that the directory exists and what's in it without drowning in
thousands of data files.

Usage:
    python smart-tree.py [path]       # default: .
    python smart-tree.py -s 5 path    # show 5 sample files per ignored dir
    python smart-tree.py -L 3 path    # max depth 3 (ignored dirs still summarized)
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    r = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"error: {start} is not inside a git repo", file=sys.stderr)
        sys.exit(1)
    return Path(r.stdout.strip())


def check_ignored_batch(rel_paths, repo_root: Path) -> set:
    """Return subset of rel_paths (as strings) that git considers ignored."""
    if not rel_paths:
        return set()
    stdin = "\n".join(str(p) for p in rel_paths) + "\n"
    r = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "--stdin"],
        input=stdin, capture_output=True, text=True,
    )
    # returncode 0 = some ignored, 1 = none ignored, other = error
    if r.returncode not in (0, 1):
        return set()
    return set(r.stdout.splitlines())


def summarize_dir(path: Path, sample: int) -> str:
    try:
        entries = list(path.iterdir())
    except OSError:
        return "<unreadable>"
    files = [e for e in entries if e.is_file()]
    dirs = [e for e in entries if e.is_dir()]
    bits = []
    if dirs:
        bits.append(f"{len(dirs)} subdir{'s' if len(dirs) != 1 else ''}")
    if files:
        bits.append(f"{len(files)} file{'s' if len(files) != 1 else ''}")
    s = ", ".join(bits) if bits else "empty"
    if files:
        picks = random.sample(files, min(sample, len(files)))
        names = ", ".join(p.name for p in picks)
        more = ", ..." if len(files) > sample else ""
        s += f"  e.g. {names}{more}"
    return s


def walk(path: Path, repo_root: Path, prefix: str, depth: int,
         max_depth: int, sample: int):
    try:
        entries = sorted(
            path.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except OSError:
        return
    entries = [e for e in entries if e.name != ".git"]

    rels = [e.relative_to(repo_root) for e in entries]
    ignored = check_ignored_batch(rels, repo_root)

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        rel = str(entry.relative_to(repo_root))
        ext = "    " if is_last else "│   "

        if rel in ignored:
            if entry.is_dir():
                print(f"{prefix}{connector}{entry.name}/  [ignored — {summarize_dir(entry, sample)}]")
            else:
                print(f"{prefix}{connector}{entry.name}  [ignored]")
        else:
            if entry.is_dir():
                print(f"{prefix}{connector}{entry.name}/")
                if max_depth is None or depth + 1 < max_depth:
                    walk(entry, repo_root, prefix + ext, depth + 1,
                         max_depth, sample)
                else:
                    # hit depth limit but show count
                    try:
                        n = len(list(entry.iterdir()))
                        print(f"{prefix}{ext}└── ... ({n} entries)")
                    except OSError:
                        pass
            else:
                print(f"{prefix}{connector}{entry.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("-s", "--sample", type=int, default=3,
                    help="sample filenames shown per ignored dir (default 3)")
    ap.add_argument("-L", "--max-depth", type=int, default=None,
                    help="max recursion depth (ignored dirs still summarized)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for reproducible sampling")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    root = Path(args.path).resolve()
    repo_root = find_repo_root(root)

    print(f"{root.name}/")
    walk(root, repo_root, "", 0, args.max_depth, args.sample)


if __name__ == "__main__":
    main()