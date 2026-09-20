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
  6. Overlays the hand-written landing page (``content/index.html``) onto
     ``book/index.html``. The landing page is a standalone document rather than
     an mdBook chapter, so it is not constrained by the book's chrome.

The result in ``book/`` is a plain static tree, served by nginx on the droplet.
Nothing in it is host-specific.

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
ASSETS = REPO_ROOT / "assets"
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
    ("docs/device-support-matrix.md", "docs/device-support-matrix.md"),
    ("docs/performance.md", "docs/performance.md"),
    ("docs/public-api.md", "docs/public-api.md"),
    ("docs/release-policy.md", "docs/release-policy.md"),
    ("docs/backend-implementer-guide.md", "docs/backend-implementer-guide.md"),
    ("docs/hexagon-operator-matrix.md", "docs/hexagon-operator-matrix.md"),
    ("docs/releases/v1.0.md", "docs/releases/v1.0.md"),
]

# Hand-authored mdBook chapters that live in website/content/ and are copied
# verbatim into the book source.
CONTENT_PAGES: list[tuple[str, str]] = [
    ("api.md", "api.md"),
]

# Standalone documents copied straight into the built site, bypassing mdBook.
# ``index.html`` overwrites the copy mdBook emits for the first chapter, which
# is also written under its own name and so is not lost.
STANDALONE_PAGES: list[tuple[Path, str]] = [
    (CONTENT / "index.html", "index.html"),
    (WEBSITE / "theme" / "favicon.svg", "favicon.svg"),
    (WEBSITE / "theme" / "logo-mark.svg", "logo-mark.svg"),
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
    resolved = (REPO_ROOT / source_dir / target).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
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
        # Site links are URLs even when the build runs on Windows, where relpath uses `\\`.
        new_target = os.path.relpath(dest, dest_dir).replace(os.sep, "/")
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
    """Write the mdBook table of contents.

    ``getting-started.md`` is a prefix chapter, so it is the book's first page
    and mdBook also emits it as ``book/index.html`` -- which the standalone
    landing page then overwrites.
    """
    lines = [
        "# Summary",
        "",
        "[Getting started](getting-started.md)",
        "",
        "# Protocol 1.0",
        "",
        "- [Specification](docs/specification.md)",
        "- [Wire ABI](docs/wire-abi.md)",
        "- [Command virtqueue](docs/virtqueue.md)",
        "- [Threat model](docs/threat-model.md)",
        "- [Release notes](docs/releases/v1.0.md)",
        "",
        "# Implementation",
        "",
        "- [Architecture](docs/architecture.md)",
        "- [Portability](docs/portability.md)",
        "- [Device support matrix](docs/device-support-matrix.md)",
        "- [Performance budgets](docs/performance.md)",
        "- [Backend implementer guide](docs/backend-implementer-guide.md)",
        "- [Hexagon operator matrix](docs/hexagon-operator-matrix.md)",
        "",
        "# Reference",
        "",
        "- [API reference](api.md)",
        "- [Public API policy](docs/public-api.md)",
        "- [Release policy](docs/release-policy.md)",
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
        f'<li><a href="{name}/index.html"><b>{name.replace("_", "-")}</b>'
        f'<span>{name}/index.html</span></a></li>'
        for name in crates
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>virtio-accel API reference</title>
<link rel="icon" href="../favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&amp;family=JetBrains+Mono:wght@400;500;700&amp;display=swap">
<style>
:root {{
  --bg: oklch(0.170 0.012 265);
  --raised: oklch(0.195 0.012 265);
  --rule: oklch(0.280 0.014 265);
  --fg: oklch(0.925 0.008 265);
  --dim: oklch(0.600 0.012 265);
  --accent: oklch(0.800 0.160 165);
  --sans: "Archivo", ui-sans-serif, system-ui, -apple-system, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: var(--sans); line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}}
a {{ color: var(--accent); text-decoration: none; }}
.wrap {{ max-width: 56rem; margin: 0 auto; padding: 0 clamp(20px, 5vw, 40px) 72px; }}
header {{ border-bottom: 1px solid var(--rule); margin-bottom: 40px; }}
.bar {{
  max-width: 56rem; margin: 0 auto;
  padding: 0 clamp(20px, 5vw, 40px); height: 62px;
  display: flex; align-items: center; gap: 10px;
}}
.bar a {{ display: flex; align-items: center; gap: 10px; color: var(--fg); }}
.mark {{
  display: block; flex: none; width: 23px; height: 20px;
  background: var(--accent);
  -webkit-mask: url("../logo-mark.svg") no-repeat center / contain;
  mask: url("../logo-mark.svg") no-repeat center / contain;
}}
.bar b {{ font-family: var(--mono); font-size: 14px; letter-spacing: -0.01em; }}
h1 {{ font-size: clamp(28px, 4vw, 38px); letter-spacing: -0.03em; margin: 40px 0 12px; }}
.lede {{ color: var(--dim); margin: 0 0 36px; max-width: 40rem; }}
ul {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }}
li a {{
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 16px; flex-wrap: wrap;
  border: 1px solid var(--rule); border-radius: 8px;
  background: var(--raised); padding: 14px 18px; color: var(--fg);
}}
li a:hover {{ border-color: var(--accent); }}
li b {{ font-family: var(--mono); font-size: 14px; font-weight: 700; }}
li span {{ font-family: var(--mono); font-size: 11.5px; color: var(--dim); }}
</style>
</head>
<body>
<header>
  <div class="bar">
    <a href="../index.html" aria-label="virtio-accel home">
      <i class="mark" aria-hidden="true"></i>
      <b>virtio-accel</b>
    </a>
  </div>
</header>
<div class="wrap">
<h1>API reference</h1>
<p class="lede">rustdoc for every crate in the workspace, built with all features enabled.
See the <a href="../api.html">crate reference</a> for tiers and roles, or the
<a href="../docs/public-api.html">public API policy</a> for what is guaranteed stable.</p>
<ul>
{rows}
</ul>
</div>
</body>
</html>
"""
    (api_dir / "index.html").write_text(html, encoding="utf-8")


def build_mdbook() -> None:
    subprocess.run(["mdbook", "build"], cwd=WEBSITE, check=True)


def copy_standalone() -> None:
    """Copy the standalone landing page and favicon into the built site.

    ``mdbook build`` recreates ``book/`` from scratch, so this must run after
    it to survive.
    """
    BOOK.mkdir(parents=True, exist_ok=True)
    for source, dest_rel in STANDALONE_PAGES:
        dest = BOOK / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)


def copy_assets() -> None:
    """Copy the repository's image and video assets into the built site.

    The README embeds its logo and demo stills with raw HTML (``<img src>``,
    ``<source srcset>``) rather than Markdown links, which the link rewriter in
    this file never sees -- it only matches ``[text](target)``. Rather than
    teach the rewriter HTML, the assets are published at the same relative path
    the README already uses, so those references resolve as-is.

    The whole tree is copied rather than only what is currently referenced: it
    is a few megabytes, and it means a newly-referenced asset needs no change
    here. ``check.py`` is what catches a reference with no file behind it.

    ``mdbook build`` recreates ``book/`` from scratch, so this must run after it.
    """
    if not ASSETS.is_dir():
        return
    shutil.copytree(ASSETS, BOOK / "assets", dirs_exist_ok=True)


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

    copy_standalone()
    copy_assets()

    return 0


if __name__ == "__main__":
    sys.exit(main())
