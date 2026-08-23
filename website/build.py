#!/usr/bin/env python3
"""Build the virtio-accel.org website.

This script is the single entry point for producing the public site. It:

  1. Curates the hand-authored narrative docs from ``docs/`` (excluding the
     internal ``docs/agents/`` and ``docs/plans/`` trees) and the repository
     ``README.md`` into the mdBook ``src/`` tree.
  2. Rewrites cross-links so that links to curated docs stay relative while
     links to repository files that are not part of the site (source files,
     the C header, conformance artifacts, licenses, etc.) point at GitHub.
  3. Generates ``src/SUMMARY.md``.
  4. Builds rustdoc for the whole workspace and copies it under ``book/api/``.
  5. Runs ``mdbook build``.

The generated ``src/`` and ``book/`` trees are git-ignored; only the curated
inputs, the theme, and this script are checked in.

Usage::

    python3 website/build.py [--skip-rustdoc] [--skip-mdbook]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEBSITE = REPO_ROOT / "website"
SRC = WEBSITE / "src"
BOOK = WEBSITE / "book"
CONTENT = WEBSITE / "content"
DOCS = REPO_ROOT / "docs"
TARGET_DOC = REPO_ROOT / "target" / "doc"

GITHUB_BASE = "https://github.com/MicroPerceptron/virtio-accel/blob/main"

# Repo-relative source path -> site-relative destination path. Order matters:
# it becomes the order of the generated SUMMARY.md.
CURATED: list[tuple[str, str]] = [
    ("README.md", "getting-started.md"),
    ("docs/specification.md", "docs/specification.md"),
    ("docs/wire-abi.md", "docs/wire-abi.md"),
    ("docs/virtqueue.md", "docs/virtqueue.md"),
    ("docs/architecture.md", "docs/architecture.md"),
    ("docs/threat-model.md", "docs/threat-model.md"),
    ("docs/portability.md", "docs/portability.md"),
    ("docs/performance.md", "docs/performance.md"),
    ("docs/public-api.md", "docs/public-api.md"),
    ("docs/release-policy.md", "docs/release-policy.md"),
    ("docs/backend-implementer-guide.md", "docs/backend-implementer-guide.md"),
    ("docs/hexagon-operator-matrix.md", "docs/hexagon-operator-matrix.md"),
    ("docs/releases/v1.0.md", "docs/releases/v1.0.md"),
]

# Hand-authored pages that live in website/content/ and are copied verbatim.
CONTENT_PAGES: list[tuple[str, str]] = [
    ("index.md", "index.md"),
    ("api.md", "api.md"),
]

LINK_RE = re.compile(r"(\[[^\]]*\]\()([^)\s]+)(\))")


def curated_dest_map() -> dict[str, str]:
    """Map repo-relative source path -> site-relative destination path."""
    return {src: dst for src, dst in CURATED}


def is_curated(repo_rel: str) -> bool:
    return repo_rel in curated_dest_map()


def resolve_repo_path(source_repo_rel: str, target: str) -> str:
    """Resolve a link target against the source file's repo-relative location."""
    source_dir = Path(source_repo_rel).parent
    resolved = (source_dir / target).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return target


def rewrite_link(source_repo_rel: str, dest_site_rel: str, target: str) -> str:
    """Rewrite a single link target for the site tree."""
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return target

    anchor = ""
    if "#" in target:
        target, anchor = target.split("#", 1)
        anchor = "#" + anchor

    if not target:
        return anchor

    repo_rel = resolve_repo_path(source_repo_rel, target)

    if is_curated(repo_rel):
        dest = curated_dest_map()[repo_rel]
        dest_dir = Path(dest_site_rel).parent
        new_target = os.path.relpath(dest, dest_dir)
        return new_target + anchor

    if (REPO_ROOT / repo_rel).exists():
        return f"{GITHUB_BASE}/{repo_rel}{anchor}"

    return target + anchor


def rewrite_markdown(source_repo_rel: str, dest_site_rel: str, text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix, target, suffix = match.groups()
        return prefix + rewrite_link(source_repo_rel, dest_site_rel, target) + suffix

    return LINK_RE.sub(repl, text)


def clean_src() -> None:
    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir(parents=True)


def copy_curated() -> None:
    for source_repo_rel, dest_site_rel in CURATED:
        source = REPO_ROOT / source_repo_rel
        dest = SRC / dest_site_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        text = rewrite_markdown(source_repo_rel, dest_site_rel, text)
        dest.write_text(text, encoding="utf-8")


def copy_content() -> None:
    for source_rel, dest_site_rel in CONTENT_PAGES:
        source = CONTENT / source_rel
        dest = SRC / dest_site_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)


def write_summary() -> None:
    lines = [
        "# Summary",
        "",
        "- [Home](index.md)",
        "- [API reference](api.md)",
        "- [Getting started](getting-started.md)",
        "",
        "- [Protocol 1.0](docs/specification.md)",
        "  - [Wire ABI](docs/wire-abi.md)",
        "  - [Command virtqueue](docs/virtqueue.md)",
        "  - [Threat model](docs/threat-model.md)",
        "  - [Release notes](docs/releases/v1.0.md)",
        "",
        "- [Implementation](docs/architecture.md)",
        "  - [Portability](docs/portability.md)",
        "  - [Performance budgets](docs/performance.md)",
        "  - [Public API policy](docs/public-api.md)",
        "  - [Backend implementer guide](docs/backend-implementer-guide.md)",
        "  - [Hexagon operator matrix](docs/hexagon-operator-matrix.md)",
        "",
        "- [Governance](docs/release-policy.md)",
        "",
    ]
    (SRC / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def build_rustdoc() -> None:
    subprocess.run(
        ["cargo", "doc", "--workspace", "--all-features", "--no-deps"],
        cwd=REPO_ROOT,
        check=True,
    )
    api_dir = BOOK / "api"
    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TARGET_DOC, api_dir, dirs_exist_ok=True)
    write_api_index(api_dir)


def write_api_index(api_dir: Path) -> None:
    """Generate a top-level index.html for the rustdoc tree.

    ``cargo doc --workspace`` does not emit a top-level index.html, so we build
    a small one that links to every crate's documentation.
    """
    crates = sorted(
        p.name
        for p in api_dir.iterdir()
        if p.is_dir() and (p / "index.html").exists()
    )
    rows = "\n".join(
        f'<li><a href="{name}/index.html">{name.replace("_", "-")}</a></li>'
        for name in crates
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>virtio-accel API reference</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 48rem;
       margin: 3rem auto; padding: 0 1.5rem; color: #0f172a; }}
h1 {{ font-size: 2rem; }}
ul {{ list-style: none; padding: 0; }}
li {{ margin: 0.5rem 0; }}
a {{ color: #0ea5e9; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>virtio-accel API reference</h1>
<p>rustdoc for every crate in the workspace, built with all features enabled.</p>
<ul>
{rows}
</ul>
</body>
</html>
"""
    (api_dir / "index.html").write_text(html, encoding="utf-8")


def build_mdbook() -> None:
    subprocess.run(["mdbook", "build"], cwd=WEBSITE, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rustdoc", action="store_true")
    parser.add_argument("--skip-mdbook", action="store_true")
    args = parser.parse_args()

    clean_src()
    copy_content()
    copy_curated()
    write_summary()

    if not args.skip_mdbook:
        build_mdbook()

    if not args.skip_rustdoc:
        build_rustdoc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
