"""Tests for the MkDocs documentation site configuration and content."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docs.yml"


# ── mkdocs.yml ────────────────────────────────────────────────────────────────

def test_mkdocs_yml_exists():
    assert MKDOCS_YML.exists(), "mkdocs.yml not found at repo root"


def test_mkdocs_yml_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    content = yaml.safe_load(MKDOCS_YML.read_text())
    assert isinstance(content, dict)


def test_mkdocs_theme_is_material():
    yaml = pytest.importorskip("yaml")
    content = yaml.safe_load(MKDOCS_YML.read_text())
    assert content["theme"]["name"] == "material"


def test_mkdocs_has_three_palette_entries():
    """Palette must have system, light, and dark entries."""
    yaml = pytest.importorskip("yaml")
    content = yaml.safe_load(MKDOCS_YML.read_text())
    palettes = content["theme"]["palette"]
    assert len(palettes) == 3
    media_values = [p.get("media", "") for p in palettes]
    assert any("prefers-color-scheme)" in m for m in media_values), "missing system entry"
    assert any("light" in m for m in media_values), "missing light entry"
    assert any("dark" in m for m in media_values), "missing dark entry"


def test_mkdocs_nav_pages_exist():
    """Every page referenced in the nav must exist under docs/."""
    yaml = pytest.importorskip("yaml")
    content = yaml.safe_load(MKDOCS_YML.read_text())

    def _collect_pages(nav):
        """Recursively collect all page paths from the nav structure."""
        pages = []
        for item in nav:
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        pages.append(v)
                    elif isinstance(v, list):
                        pages.extend(_collect_pages(v))
        return pages

    missing = []
    for page in _collect_pages(content.get("nav", [])):
        if not (DOCS_DIR / page).exists():
            missing.append(page)

    assert not missing, f"Nav references missing docs pages: {missing}"


def test_mkdocs_repo_url_points_to_correct_repo():
    yaml = pytest.importorskip("yaml")
    content = yaml.safe_load(MKDOCS_YML.read_text())
    repo_url = content.get("repo_url", "")
    assert "scivis-embedding-workbench" in repo_url
    assert "bams-ai-data-exploration" not in repo_url


# ── docs/ content ─────────────────────────────────────────────────────────────

def test_docs_index_exists():
    assert (DOCS_DIR / "index.md").exists()


def test_docs_examples_exists():
    assert (DOCS_DIR / "examples.md").exists()


def test_docs_index_has_copernicus_attribution():
    text = (DOCS_DIR / "index.md").read_text()
    assert "Copernicus" in text, "ERA5 attribution missing from index.md"


def test_docs_index_has_ibtracs_attribution():
    text = (DOCS_DIR / "index.md").read_text()
    assert "IBTrACS" in text, "IBTrACS attribution missing from index.md"


def test_docs_no_stale_repo_references():
    """No doc page should reference the old bams-ai-data-exploration repo."""
    stale = "bams-ai-data-exploration"
    offenders = [
        p for p in DOCS_DIR.rglob("*.md")
        if stale in p.read_text()
    ]
    assert not offenders, f"Stale repo reference found in: {[str(p) for p in offenders]}"


def test_docs_pipeline_pages_exist():
    expected = [
        "pipeline/data-processing.md",
        "pipeline/prepare-data.md",
        "pipeline/embeddings.md",
        "pipeline/dashboard.md",
    ]
    missing = [p for p in expected if not (DOCS_DIR / p).exists()]
    assert not missing, f"Missing pipeline docs: {missing}"


# ── GitHub Actions workflow ───────────────────────────────────────────────────

def test_docs_workflow_exists():
    assert WORKFLOW.exists(), ".github/workflows/docs.yml not found"


def test_docs_workflow_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    content = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(content, dict)


def test_docs_workflow_triggers_on_main():
    # PyYAML parses the YAML key `on` as Python True, not the string "on"
    yaml = pytest.importorskip("yaml")
    content = yaml.safe_load(WORKFLOW.read_text())
    on_block = content.get(True, content.get("on", {}))
    branches = on_block.get("push", {}).get("branches", [])
    assert "main" in branches, "Workflow should trigger on push to main"


def test_docs_workflow_has_deploy_step():
    text = WORKFLOW.read_text()
    assert "mkdocs gh-deploy" in text, "Workflow must contain mkdocs gh-deploy step"
