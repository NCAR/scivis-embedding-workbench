"""Tests for the projection pipeline: make_projection.py + write_projection_table.py.

Pure functions only -- no GPU, no LanceDB, no network. The parts that need CUDA
are three thin adapter methods on `CudaBackend`, which by construction cannot run
here; everything else in both scripts is exercised below.

The modules are loaded by file path rather than imported as `helpers.*`. Two
notebook directories in this repo ship a package called `helpers`, only one of
which can be live in a given interpreter, and `tests/helpers/` already claims
that name for notebooks/02-generate-embeddings.
"""

import ast
import getpass
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

REPO_ROOT = Path(__file__).parent.parent
HELPERS = REPO_ROOT / "notebooks" / "05-latent-space-exploration" / "helpers"


def _load(name: str):
    """Load one helper module by path, under a name that cannot collide."""
    path = HELPERS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage1 = _load("make_projection")
stage2 = _load("write_projection_table")


# ── stage 1: reading is width-agnostic ───────────────────────────────────────

@pytest.mark.parametrize("dim", [768, 1024, 3])
def test_vector_width_read_from_arrow(dim):
    """The embedding width comes from the data, never from a constant."""
    values = np.arange(5 * dim, dtype=np.float32)
    col = pa.FixedSizeListArray.from_arrays(pa.array(values), dim)
    assert stage1.vector_width(col) == dim


@pytest.mark.parametrize("dim", [768, 1024])
def test_to_matrix_roundtrip(dim):
    rows = 7
    values = np.arange(rows * dim, dtype=np.float32)
    col = pa.FixedSizeListArray.from_arrays(pa.array(values), dim)
    X = stage1.to_matrix(col, dim)
    assert X.shape == (rows, dim)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X, values.reshape(rows, dim))


def test_to_matrix_is_writable():
    """pynndescent's numba kernels reject read-only arrays outright."""
    col = pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(12, np.float32)), 4)
    assert stage1.to_matrix(col, 4).flags.writeable


def test_to_matrix_rejects_ragged_rows():
    """A wrong width would reshape cleanly and misalign every subsequent row."""
    col = pa.array([[1.0, 2.0], [3.0]], type=pa.list_(pa.float32()))
    with pytest.raises(ValueError, match="ragged"):
        stage1.to_matrix(col, 2)


def test_vector_width_rejects_non_vector_column():
    with pytest.raises(TypeError):
        stage1.vector_width(pa.array([1, 2, 3]))


# ── stage 1: fingerprint ─────────────────────────────────────────────────────

def test_fingerprint_is_order_sensitive():
    a = pa.array(["p1", "p2", "p3"])
    b = pa.array(["p1", "p3", "p2"])
    assert stage1.fingerprint(a) != stage1.fingerprint(b)


def test_fingerprint_ignores_arrow_layout():
    """Chunking must not change the digest.

    The whole point is to compare an id column read alongside the embeddings
    against the same column read alongside the identity columns, which produces
    a differently-laid-out but logically identical array.
    """
    flat = pa.array(["p1", "p2", "p3", "p4"])
    chunked = pa.chunked_array([["p1", "p2"], ["p3", "p4"]])
    assert stage1.fingerprint(flat) == stage1.fingerprint(chunked)


# ── stage 1: subsampling and scale-derived defaults ──────────────────────────

def test_sample_offsets_returns_none_when_not_subsampling():
    assert stage1.sample_offsets(100, None, 0) is None
    assert stage1.sample_offsets(100, 500, 0) is None


def test_sample_offsets_are_sorted_and_unique():
    off = stage1.sample_offsets(10_000, 250, seed=1)
    assert len(off) == 250
    assert len(np.unique(off)) == 250
    assert (np.diff(off) > 0).all()
    assert off.max() < 10_000


@pytest.mark.parametrize("n_rows", [50_000, 1_000_000])
def test_scale_derived_defaults_are_sane(n_rows):
    """A constant tuned at 1M is wrong at 50k, so the defaults track N."""
    mcs = stage1.default_min_cluster_size(n_rows)
    ms = stage1.default_min_samples(mcs)
    k = stage1.default_kmeans_k(n_rows)
    assert 25 <= mcs <= 5000
    assert 5 <= ms <= 100
    assert ms < mcs
    assert 8 <= k <= 64


def test_min_cluster_size_grows_with_table():
    assert (stage1.default_min_cluster_size(1_000_000)
            > stage1.default_min_cluster_size(50_000))


# ── stage 2: cluster folding ─────────────────────────────────────────────────

def test_fold_rare_clusters_leaves_small_runs_alone():
    labels = np.array([0, 0, 1, 1, 2, -1], dtype=np.int32)
    np.testing.assert_array_equal(stage2.fold_rare_clusters(labels, cap=64), labels)


def test_fold_rare_clusters_respects_cap_and_keeps_biggest():
    # 200 clusters, cluster id 0 by far the largest.
    labels = np.concatenate([
        np.zeros(1000, dtype=np.int32),
        np.repeat(np.arange(1, 200, dtype=np.int32), 3),
    ])
    folded = stage2.fold_rare_clusters(labels, cap=64)
    distinct = set(np.unique(folded).tolist())
    assert len(distinct) <= 64, "must fit datashader's category budget"
    assert 0 in distinct, "the largest cluster must survive"
    assert stage2.OTHER_CLUSTER in distinct, "the tail must be folded, not dropped"
    assert len(folded) == len(labels), "folding must not drop rows"


def test_fold_rare_clusters_keeps_noise_distinct_from_other():
    """`-1` (density noise) and OTHER (too small to colour) are different claims."""
    labels = np.concatenate([
        np.full(50, -1, dtype=np.int32),
        np.repeat(np.arange(0, 100, dtype=np.int32), 5),
    ])
    folded = stage2.fold_rare_clusters(labels, cap=64)
    assert (folded == -1).sum() == 50
    assert stage2.OTHER_CLUSTER in set(np.unique(folded).tolist())
    assert len(set(np.unique(folded).tolist())) <= 64


def test_fold_rare_clusters_handles_all_noise():
    labels = np.full(10, -1, dtype=np.int32)
    np.testing.assert_array_equal(stage2.fold_rare_clusters(labels), labels)


def test_fold_default_cap_comes_from_the_viewer():
    """Imported, not restated: raising the cap in one place must move both."""
    assert stage2.fold_rare_clusters.__defaults__[0] == stage2._data.MAX_CATEGORIES


# ── stage 2: embeddings unpack from array shape ──────────────────────────────

@pytest.mark.parametrize("n_cluster_components", [5, 10, 30])
def test_z_columns_derived_from_array_width(n_cluster_components):
    """Re-running with a different --n-components-cluster needs no code change."""
    rows = 4
    df = pd.DataFrame({"patch_index": np.arange(rows)})
    view = np.zeros((rows, 2), np.float32)
    z = np.arange(rows * n_cluster_components, dtype=np.float32).reshape(rows, -1)

    out = stage2.unpack_embeddings(df, view, z)
    z_cols = [c for c in out.columns if c.startswith("z")]
    assert z_cols == [f"z{i}" for i in range(n_cluster_components)]
    np.testing.assert_array_equal(out[f"z{n_cluster_components - 1}"], z[:, -1])


def test_view_unpacks_to_x_y_and_extras():
    rows = 3
    df = pd.DataFrame({"patch_index": np.arange(rows)})
    view = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    out = stage2.unpack_embeddings(df, view, np.zeros((rows, 2), np.float32))

    np.testing.assert_array_equal(out["x"], view[:, 0])
    np.testing.assert_array_equal(out["y"], view[:, 1])
    np.testing.assert_array_equal(out["view2"], view[:, 2])


def test_two_d_view_makes_no_extra_columns():
    rows = 3
    df = pd.DataFrame({"patch_index": np.arange(rows)})
    out = stage2.unpack_embeddings(
        df, np.zeros((rows, 2), np.float32), np.zeros((rows, 2), np.float32)
    )
    assert not [c for c in out.columns if c.startswith("view")]


# ── stage 2: the join rules ──────────────────────────────────────────────────

SOURCE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("image_blob", pa.large_binary()),
    ("embedding", pa.list_(pa.float32())),
    ("filename", pa.large_string()),
    ("storm_lats", pa.large_string()),
    ("max_wind_kts", pa.float32()),
    ("hurricane_present", pa.bool_()),
])


def test_rule1_skips_blobs_and_vectors():
    """`image_blob` across a million patch rows would be ruinous to join."""
    keep = stage2.scalar_columns(SOURCE_SCHEMA, skip=("image_blob",))
    assert "image_blob" not in keep
    assert "embedding" not in keep
    assert {"max_wind_kts", "hurricane_present"} <= set(keep)


def test_rule1_skips_strings_but_keeps_the_join_key():
    """Strings can never be colour-bys, so joining them is pure cost.

    `infer_color_roles` classifies only bools and numerics, so no string column
    reaches the dropdown whatever its cardinality. Measured on this dataset,
    `filename` and the storm-detail strings were 23% of the finished table.
    """
    keep = stage2.scalar_columns(SOURCE_SCHEMA, skip=("image_blob",))
    assert "filename" not in keep
    assert "storm_lats" not in keep
    assert "id" in keep, "the join key must survive"


def test_rule1_keep_strings_opts_back_in():
    keep = stage2.scalar_columns(SOURCE_SCHEMA, skip=("image_blob",), keep_strings=True)
    assert {"filename", "storm_lats", "id"} <= set(keep)
    assert "image_blob" not in keep


def test_rule2_expands_timestamps_by_dtype_not_by_name():
    """A dataset whose time column is `valid_time` must work unchanged."""
    df = pd.DataFrame({
        "valid_time": pd.to_datetime(["2017-09-06 06:00", "2018-01-02 13:00"]),
        "not_a_time": [1, 2],
    })
    out = stage2.expand_timestamps(df.copy())
    for part in ("year", "month", "day", "hour", "dayofyear"):
        assert f"valid_time_{part}" in out.columns
    assert not [c for c in out.columns if c.startswith("not_a_time_")]
    assert out["valid_time_year"].tolist() == [2017, 2018]
    assert out["valid_time_hour"].tolist() == [6, 13]


def test_rule3_uses_the_configs_own_grid():
    """Rectangular grids are the normal case here (ERA5 is 16x56)."""
    config = {"attention_spatial_h": "16", "attention_spatial_w": "56"}
    df = pd.DataFrame({"patch_index": [0, 55, 56, 57]})
    out, grid = stage2.add_patch_geometry(df, config)

    assert grid == (16, 56)
    assert out["patch_row"].tolist() == [0, 0, 1, 1]
    assert out["patch_col"].tolist() == [0, 55, 0, 1]


def test_rule3_falls_back_to_image_dimensions():
    """Configs written before the attention keys existed still work."""
    config = {"image_h": "256", "image_w": "896", "patch_size": "16"}
    _, grid = stage2.add_patch_geometry(pd.DataFrame({"patch_index": [0]}), config)
    assert grid == (16, 56)


def test_rule3_skips_gracefully_when_grid_is_unknowable():
    df = pd.DataFrame({"patch_index": [0, 1]})
    out, grid = stage2.add_patch_geometry(df, {})
    assert grid is None
    assert "patch_row" not in out.columns


def test_rule4_skipped_without_an_extent():
    """`get_spatial_extent` returns None rather than guessing; so does this."""
    df = pd.DataFrame({"patch_row": [0], "patch_col": [0]})
    out = stage2.add_latlon(df.copy(), (16, 56), None)
    assert "lat" not in out.columns and "lon" not in out.columns

    out = stage2.add_latlon(df.copy(), None, {"lat_min": 0, "lat_max": 1,
                                              "lon_min": 0, "lon_max": 1})
    assert "lat" not in out.columns


def test_rule4_row_zero_is_north_and_longitude_is_normalised():
    extent = {"lat_min": 15.0, "lat_max": 35.0, "lon_min": 260.0, "lon_max": 330.0}
    df = pd.DataFrame({"patch_row": [0, 15], "patch_col": [0, 0]})
    out = stage2.add_latlon(df, (16, 56), extent)

    assert out["lat"][0] > out["lat"][1], "row 0 must be the northern edge"
    assert -180 <= out["lon"][0] <= 180, "longitude must be normalised to -180..180"


def test_rule5_drops_constant_columns_and_classifies_clusters():
    """`hour` is all-zero in a 24h experiment: one flat colour reads as a bug."""
    rows = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=rows),
        "y": rng.normal(size=rows),
        "cluster": rng.integers(0, 5, rows),
        "cluster_top": rng.integers(0, 5, rows),
        "lat": rng.normal(size=rows),
        "constant_hour": np.zeros(rows, dtype=np.int8),
        "z0": rng.normal(size=rows),
        "z9": rng.normal(size=rows),
    })
    roles = stage2.colour_roles(df, extra_categorical=("cluster_top",))
    everything = roles["categorical"] + roles["continuous"]

    assert "constant_hour" not in everything, "a single-valued column paints one colour"
    assert "x" not in everything and "y" not in everything
    assert "z0" not in everything and "z9" not in everything, \
        "clustering axes are stored for re-clustering, not for colouring"
    assert "cluster_top" in roles["categorical"]
    assert "lat" in roles["continuous"]


# ── guardrails ───────────────────────────────────────────────────────────────

PY_SCRIPTS = ["make_projection.py", "write_projection_table.py"]
ABSOLUTE_PREFIXES = ("/glade/", "/Users/", "/home/")


def _code_constants(name: str):
    """String and numeric literals in real code, excluding docstrings.

    Substring-matching the raw source is too blunt: it flags prose that
    *mentions* 768, and usage examples in a module docstring. What matters is
    whether a literal is baked into an expression.
    """
    tree = ast.parse((HELPERS / name).read_text())
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    strings, numbers = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, str):
            if node.value not in docstrings:
                strings.append(node.value)
        elif isinstance(node.value, int) and not isinstance(node.value, bool):
            numbers.append(node.value)
    return strings, numbers


@pytest.mark.parametrize("name", PY_SCRIPTS)
def test_no_absolute_paths_in_code(name):
    """These must run on anyone's machine and on Casper without editing.

    notebooks/04-benchmarking hardcodes a /glade PROJECT_ROOT with a
    commented-out /Users alternative; this pipeline does not repeat that. Paths
    inside docs and usage examples are fine -- what must not exist is a literal
    baked into an expression.
    """
    strings, _ = _code_constants(name)
    offenders = [
        s for s in strings if any(s.startswith(p) for p in ABSOLUTE_PREFIXES)
    ]
    assert not offenders, f"{name} hardcodes absolute path(s): {offenders}"


def test_pbs_has_no_absolute_paths_outside_comments():
    source = (HELPERS / "run_projection.pbs").read_text()
    code = [
        line for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    offenders = [
        line.strip() for line in code
        if any(p in line for p in ABSOLUTE_PREFIXES)
    ]
    assert not offenders, f"run_projection.pbs hardcodes paths: {offenders}"


@pytest.mark.parametrize("name", PY_SCRIPTS)
def test_no_hardcoded_embedding_width(name):
    """768 and 1024 must never be literals: the width comes from the Arrow type."""
    _, numbers = _code_constants(name)
    for literal in (768, 1024):
        assert literal not in numbers, f"{name} hardcodes an embedding width: {literal}"


@pytest.mark.parametrize("name", PY_SCRIPTS + ["run_projection.pbs"])
def test_no_personal_username_anywhere(name):
    """Catches a personal path smuggled in through a comment or usage example.

    Comments may show `/glade/work/$USER/...`; they may not show a real account.
    """
    user = getpass.getuser()
    if user in ("runner", "root"):  # CI users; the check would be meaningless
        pytest.skip("generic CI username")
    assert user not in (HELPERS / name).read_text(), \
        f"{name} mentions the current username; use $USER or a placeholder"


def test_max_cluster_size_reaches_both_backends():
    """A ceiling that only worked on one backend would silently change results
    between a local test run and the Casper job."""
    import inspect

    for backend in (stage1.CpuBackend, stage1.CudaBackend):
        params = inspect.signature(backend.cluster).parameters
        assert "max_cluster_size" in params, f"{backend.__name__}.cluster ignores the ceiling"


def test_max_cluster_size_defaults_to_no_ceiling():
    args = stage1.build_parser().parse_args(["--experiment", str(REPO_ROOT)])
    assert args.max_cluster_size is None


def test_max_cluster_size_breaks_a_dominant_cluster():
    """The measured behaviour this flag exists for.

    Two overlapping blobs plus a small distant one. Excess-of-mass finds the
    overlapping pair more "stable" merged than split and returns 96% of the
    points as a single cluster -- the same degenerate answer seen on the real
    patch embeddings. A ceiling forces it to descend into the sub-clusters.

    The separation matters: at 1.5 sigma eom splits the pair on its own and the
    ceiling never binds, so this fixture would silently stop testing anything.
    """
    rng = np.random.default_rng(0)
    Z = np.vstack([
        rng.normal(0.0, 0.30, size=(1500, 4)),
        rng.normal(0.8, 0.30, size=(1500, 4)),
        rng.normal(9.0, 0.30, size=(120, 4)),
    ])
    uncapped = stage1.CpuBackend("euclidean", 0, False).cluster(Z, 60, 5, "eom", None)[0]
    capped = stage1.CpuBackend("euclidean", 0, False).cluster(Z, 60, 5, "eom", 2000)[0]

    def largest(labels):
        labels = np.asarray(labels)
        return np.bincount(labels[labels >= 0]).max() / labels.size

    assert largest(uncapped) > 0.9, "fixture no longer reproduces eom dominance"
    assert largest(capped) < 0.5, "the ceiling must split the dominant cluster"
    assert len(set(capped[capped >= 0].tolist())) > len(set(uncapped[uncapped >= 0].tolist()))


def test_stage1_refuses_a_one_dimensional_view():
    """The viewer reads columns named x and y, so V < 2 cannot work."""
    args = stage1.build_parser().parse_args(
        ["--experiment", str(REPO_ROOT), "--n-components-view", "1"]
    )
    assert args.n_components_view < stage1.MIN_VIEW_COMPONENTS


def test_protected_tables_cannot_be_overwritten():
    for name in ("config", "patch_embeddings", "image_embeddings"):
        assert name in stage2.PROTECTED_TABLES
