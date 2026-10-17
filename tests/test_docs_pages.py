"""The documentation site's guide pages, cut from README.md by docs/hooks/readme_pages.py (needs the docs extra)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs")
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("readme_pages", ROOT / "docs/hooks/readme_pages.py")
readme_pages = importlib.util.module_from_spec(spec)
sys.modules["readme_pages"] = readme_pages
spec.loader.exec_module(readme_pages)

README = """<p align="center"><img src="docs/logo.svg"></p>
<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
</p>

Intro, see [Limitations](#limitations).

## Quickstart

```python
## not a heading: a comment in code
```
Next: the [tour](#a-tour), the [example](examples/test_x.py) and [hardware](docs/hardware.md#wiring).

## A tour

### Traces

<p><img src="docs/viewer.png"></p> Back to [Quickstart](#quickstart), or [Traces](#traces) here.

## Limitations

Pre-alpha.
"""


@pytest.fixture
def pages(monkeypatch):
    monkeypatch.setattr(
        readme_pages, "PAGES", {"index.md": [""], "quickstart.md": ["Quickstart"], "tour.md": ["A tour"], "limitations.md": ["Limitations"]}
    )
    return readme_pages.pages(README)


def test_each_page_takes_its_sections_and_code_is_not_cut_at_a_comment(pages):
    assert "## not a heading" in pages["quickstart.md"] and pages["quickstart.md"].startswith("# Quickstart")
    assert "### Traces" in pages["tour.md"] and "Pre-alpha" in pages["limitations.md"]
    assert '<a href="#quickstart">' not in pages["index.md"], "the README's own navigation row stays on GitHub"


def test_links_follow_what_they_pointed_at(pages):
    assert "](limitations.md#limitations)" in pages["index.md"]
    assert "](tour.md#a-tour)" in pages["quickstart.md"] and "](quickstart.md#quickstart)" in pages["tour.md"]
    assert "](#traces)" in pages["tour.md"], "an anchor on its own page stays as it is"
    assert "](https://github.com/JeremiahM37/robowright/blob/main/examples/test_x.py)" in pages["quickstart.md"]
    assert "](hardware.md#wiring)" in pages["quickstart.md"]


def test_pictures_in_raw_html_are_found_from_each_page(pages):
    assert 'src="logo.svg"' in pages["index.md"]  # the site's root
    assert 'src="../viewer.png"' in pages["tour.md"]  # /tour/, one down


def test_a_section_the_readme_lost_fails_the_build(monkeypatch):
    monkeypatch.setattr(readme_pages, "PAGES", {"robots.md": ["Robots"]})
    with pytest.raises(ValueError, match="no ## section"):
        readme_pages.pages(README)


def test_the_real_readme_has_every_section_the_site_takes():
    out = readme_pages.pages((ROOT / "README.md").read_text(encoding="utf-8"))
    assert set(out) == set(readme_pages.PAGES)
