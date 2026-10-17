"""Builds the site's guide pages from README.md, so the README stays the one place they are written.

README.md is cut at its ``## `` headings and each page below takes the sections it names. Links are then made to
work on the site: an anchor that is now on another page points at that page, a picture in docs/ loses its
``docs/`` prefix, and a link to a file in the repository (a test, an example, LICENSE) goes to it on GitHub.
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown.extensions.toc import slugify
from mkdocs.structure.files import File

REPO = "https://github.com/JeremiahM37/robowright/blob/main/"

# page -> README ## sections, in order ("" is the text before the first ## heading)
PAGES = {
    "index.md": ["", "Why"],
    "quickstart.md": ["Quickstart (60 seconds)"],
    "install.md": ["Install"],
    "tour.md": ["A tour"],
    "robots.md": ["Robots", "Legged robots"],
    "matrix.md": ["The matrix: every robot on every engine"],
    "how-it-works.md": ["How it works", "Extending robowright"],
    "limitations.md": ["Limitations", "Roadmap", "License"],
}

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6}) (.+?)\s*$")


def _sections(text: str) -> dict[str, list[str]]:
    """README's ## sections by title, each with its own lines (fenced code is never taken for a heading)."""
    out: dict[str, list[str]] = {"": []}
    current, fenced = "", False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            current = line[3:].strip()
            out[current] = []
        out[current].append(line)
    return out


def _anchors(lines: list[str]) -> list[str]:
    fenced, found = False, []
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(line)):
            found.append(slugify(m.group(2), "-"))
    return found


def _intro(lines: list[str]) -> list[str]:
    """The README's opening without its in-page navigation row (the site has its own)."""
    text = "\n".join(lines)
    text = re.sub(r'<p align="center">\s*<a href="#quickstart.*?</p>\n', "", text, flags=re.S)
    return ["# robowright { .rw-hidden }", "", *text.splitlines()]


def _relink(text: str, page: str, where: dict[str, str]) -> str:
    def target(url: str, html: bool = False) -> str:
        if url.startswith(("http://", "https://", "mailto:")):
            return url
        path, _, anchor = url.partition("#")
        if not path:  # an anchor in the README: on this page, or on the page it moved to
            other = where.get(anchor)
            return url if other in (None, page) else f"{other}#{anchor}"
        if path.startswith("docs/"):  # raw HTML is not rewritten by MkDocs: from a page's own directory, up one
            up = "../" if html and page != "index.md" else ""
            return up + path[len("docs/") :] + (f"#{anchor}" if anchor else "")
        return REPO + url

    text = re.sub(r"\]\(([^)\s]+)\)", lambda m: f"]({target(m.group(1))})", text)
    return re.sub(r'((?:href|src)=")([^"]+)(")', lambda m: m.group(1) + target(m.group(2), html=True) + m.group(3), text)


def pages(readme: str) -> dict[str, str]:
    sections = _sections(readme)
    missing = [s for names in PAGES.values() for s in names if s not in sections]
    if missing:
        raise ValueError(f"README.md has no ## section {missing}: update PAGES in docs/hooks/readme_pages.py")
    where = {a: page for page, names in PAGES.items() for s in names for a in _anchors(sections[s])}
    out = {}
    for page, names in PAGES.items():
        lines: list[str] = []
        for s in names:
            body = _intro(sections[s]) if s == "" else sections[s]
            if page != "index.md" and len(names) == 1:  # the page's one section is its title
                body = ["# " + body[0][3:], *body[1:]]
            lines += [*body, ""]
        out[page] = _relink("\n".join(lines), page, where)
    return out


def on_page_markdown(markdown, page, config, files):
    """A page written in docs/ links to the repository as ``../path``, which works on GitHub: on the site it goes there."""
    return re.sub(r"\]\(\.\./([^)\s]+)\)", lambda m: f"]({REPO}{m.group(1)})", markdown)


def on_files(files, config):
    readme = (Path(config["config_file_path"]).parent / "README.md").read_text(encoding="utf-8")
    for page, content in pages(readme).items():
        files.append(File.generated(config, page, content=content))
    return files
