"""Tests for model_registry.json integrity and get_model_info()."""
import json
from pathlib import Path

import pytest

from helpers.embedding_experiment import get_model_info, pick_num_sub_vectors

REGISTRY_PATH = (
    Path(__file__).parents[2]
    / "notebooks"
    / "02-generate-embeddings"
    / "helpers"
    / "model_registry.json"
)
HELPERS_DIR = REGISTRY_PATH.parent

REQUIRED_KEYS = {"script", "default_model", "description", "default_batch", "default_workers", "default_scan_batch"}


@pytest.fixture(scope="module")
def registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def test_registry_is_valid_json():
    """Registry file parses as valid JSON."""
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_registry_not_empty(registry):
    assert len(registry) > 0


@pytest.mark.parametrize("family", ["dinov3", "dinov3_sat_rect", "openclip"])
def test_required_keys_present(registry, family):
    """Each known family has all required keys."""
    assert family in registry, f"Family '{family}' missing from registry"
    missing = REQUIRED_KEYS - registry[family].keys()
    assert not missing, f"'{family}' is missing keys: {missing}"


@pytest.mark.parametrize("family", ["dinov3", "dinov3_sat_rect", "openclip"])
def test_script_file_exists(registry, family):
    """Every registered script resolves to a real .py file on disk."""
    script = registry[family]["script"]
    path = HELPERS_DIR / script
    assert path.exists(), f"Script file not found: {path}"
    assert path.suffix == ".py"


def test_openclip_has_default_pretrained(registry):
    """openclip entry must have default_pretrained (required for CLI)."""
    assert "default_pretrained" in registry["openclip"]
    assert registry["openclip"]["default_pretrained"]


def test_default_batch_is_positive_int(registry):
    for family, info in registry.items():
        assert isinstance(info["default_batch"], int) and info["default_batch"] > 0, \
            f"'{family}.default_batch' must be a positive int"


def test_default_workers_is_positive_int(registry):
    for family, info in registry.items():
        assert isinstance(info["default_workers"], int) and info["default_workers"] > 0, \
            f"'{family}.default_workers' must be a positive int"


# ── get_model_info() ──────────────────────────────────────────────────────────

def test_get_model_info_returns_script_path():
    """get_model_info injects script_path pointing to an existing file."""
    info = get_model_info("dinov3")
    assert "script_path" in info
    assert Path(info["script_path"]).exists()


def test_get_model_info_openclip():
    info = get_model_info("openclip")
    assert info["default_model"] == "ViT-L-14"
    assert info["default_pretrained"] == "laion2b_s32b_b82k"
    assert Path(info["script_path"]).exists()


def test_get_model_info_dinov3_sat_rect():
    """Satellite family resolves to the v5 rectangular script."""
    info = get_model_info("dinov3_sat_rect")
    assert info["script"] == "v5_dino_embeddings_lancedb.py"
    assert Path(info["script_path"]).exists()


def test_dinov3_sat_rect_model_name_is_tagged(registry):
    """The .sat493m tag is load-bearing.

    Without it, timm loads the LVD-1689M web weights AND ImageNet normalization
    instead of the satellite weights and their (0.430, 0.411, 0.296) mean.
    """
    assert registry["dinov3_sat_rect"]["default_model"].endswith(".sat493m")


def test_get_model_info_unknown_raises():
    """Unknown family raises KeyError with a helpful message."""
    with pytest.raises(KeyError, match="Unknown model family"):
        get_model_info("nonexistent_family")


def test_get_model_info_error_lists_available():
    """KeyError message lists available families."""
    with pytest.raises(KeyError) as exc_info:
        get_model_info("bad_family")
    assert "dinov3" in str(exc_info.value)
    assert "openclip" in str(exc_info.value)


# ── pick_num_sub_vectors() ────────────────────────────────────────────────────

@pytest.mark.parametrize(("embedding_dim", "expected"), [(768, 96), (1024, 128), (384, 48)])
def test_pick_num_sub_vectors_known_dims(embedding_dim, expected):
    """768 keeps the previously hardcoded 96; ViT-L's 1024 gets 128."""
    assert pick_num_sub_vectors(embedding_dim) == expected


@pytest.mark.parametrize("embedding_dim", [256, 384, 512, 768, 1024, 1280, 1536, 4096])
def test_pick_num_sub_vectors_always_divides(embedding_dim):
    """LanceDB requires embedding_dim % num_sub_vectors == 0 for IVF-PQ."""
    n = pick_num_sub_vectors(embedding_dim)
    assert n > 0
    assert embedding_dim % n == 0


def test_pick_num_sub_vectors_handles_indivisible_target():
    """A dim whose /8 target isn't a divisor falls back to a real divisor."""
    n = pick_num_sub_vectors(100)  # target 12 (not a divisor of 100)
    assert 100 % n == 0


@pytest.mark.parametrize("bad", [0, -768, 768.0, "768"])
def test_pick_num_sub_vectors_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        pick_num_sub_vectors(bad)
