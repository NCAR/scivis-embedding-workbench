import marimo

__generated_with = "0.10.0"


app = marimo.App()


# ---------------------------------------------------------------------------
# Cell 1 — Imports
# ---------------------------------------------------------------------------
@app.cell
def _():
    import dask
    import glob
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    from PIL import Image
    from tqdm import tqdm

    return (
        Image, Path, ProcessPoolExecutor, ThreadPoolExecutor, as_completed,
        dask, glob, np, pd, plt, tqdm, xr,
    )


# ---------------------------------------------------------------------------
# Cell 2 — Data-loading config
# DEFAULT_CHUNKS["time"] = 24 aligns with BATCH_SIZE=24 so each batch
# request maps to exactly one dask chunk — no wasted I/O or rechunking.
# ---------------------------------------------------------------------------
@app.cell
def _(ThreadPoolExecutor, glob, pd, xr):
    DATA_ROOT = "/glade/campaign/collections/gdex/data/d633000/e5.oper.an.sfc"
    DEFAULT_CHUNKS = {"time": 24, "latitude": -1, "longitude": -1}

    PATTERNS = {
        "MSL":     f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_151_msl.ll025sc.*.nc",
        "TCWV":    f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_137_tcwv.ll025sc.*.nc",
        "VAR_10U": f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_165_10u.ll025sc.*.nc",
        "VAR_10V": f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_166_10v.ll025sc.*.nc",
    }

    # ---------- helpers ----------
    def _months(start, end):
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        return [p.strftime("%Y%m") for p in pd.period_range(s, e, freq="M")]

    def _files(pattern, start, end):
        fs = []
        for ym in _months(start, end):
            fs += glob.glob(pattern.format(ym))
        fs = sorted(fs)
        if not fs:
            raise FileNotFoundError(pattern)
        return fs

    def _preproc(lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        def _pp(ds):
            # ERA5 latitude is descending (90 → -90)
            if (lat_min is not None) and (lat_max is not None):
                ds = ds.sel(latitude=slice(lat_max, lat_min))
            # ERA5 longitude is native 0..360 (no wrapping)
            if (lon_min is not None) and (lon_max is not None):
                ds = ds.sel(longitude=slice(lon_min, lon_max))
            return ds
        return _pp

    def _open_var(name, pattern, start, end, *, lat_min=None, lat_max=None,
                  lon_min=None, lon_max=None):
        ds = xr.open_mfdataset(
            _files(pattern, start, end),
            combine="by_coords", engine="h5netcdf", parallel=True,
            chunks=DEFAULT_CHUNKS,
            preprocess=_preproc(lat_min, lat_max, lon_min, lon_max),
            data_vars="minimal", coords="minimal", compat="override", join="override",
        ).sortby("time").sel(time=slice(pd.to_datetime(start), pd.to_datetime(end)))
        var = list(ds.data_vars)[0] if len(ds.data_vars) == 1 else name
        if var not in ds.data_vars:
            raise ValueError(f"{name}: available {list(ds.data_vars)}")
        return ds[[var]].rename({var: name})

    def open_four(start, end, *, lat_min=None, lat_max=None,
                  lon_min=None, lon_max=None, n_io_threads=4):
        """Open all 4 ERA5 variables in parallel (I/O-bound → ThreadPoolExecutor)."""
        def _open(item):
            name, pat = item
            return _open_var(name, pat, start, end,
                             lat_min=lat_min, lat_max=lat_max,
                             lon_min=lon_min, lon_max=lon_max)

        with ThreadPoolExecutor(max_workers=n_io_threads) as ex:
            parts = list(ex.map(_open, PATTERNS.items()))

        return xr.merge(parts, compat="override", join="override")

    return DATA_ROOT, DEFAULT_CHUNKS, PATTERNS, open_four


# ---------------------------------------------------------------------------
# Cell 3 — Pipeline config (all magic numbers in one place)
# ---------------------------------------------------------------------------
@app.cell
def _(Path):
    # --- domain ---
    LAT_MIN, LAT_MAX = 15.0, 35.0
    LON_MIN, LON_MAX = 260.0, 330.0
    START, END = "2017-06-01", "2017-12-01"

    # --- channel encoding ---
    R_RANGE     = (-20.0, 20.0)   # MSLP anomaly [hPa]
    G_RANGE     = (0.0,   35.0)   # 10m wind speed [m/s]
    B_RANGE     = (20.0,  70.0)   # TCWV [kg m^-2]
    TCWV_NONLIN = "sqrt"          # None | "sqrt" | "log"

    # --- output ---
    OUT_ROOT     = Path("./openclip_ready_896x256")
    IMG_WIDTH    = 896
    IMG_HEIGHT   = 256
    JPEG_QUALITY = 90

    # --- parallelism ---
    N_WORKERS    = 32   # dask threaded-scheduler workers AND ProcessPoolExecutor size
    N_IO_THREADS = 4    # threads for opening the 4 ERA5 variables concurrently (I/O-bound)
    BATCH_SIZE   = 24   # timesteps per dask.compute() call (= 1 day of hourly data)

    return (
        B_RANGE, BATCH_SIZE, END, G_RANGE, IMG_HEIGHT, IMG_WIDTH, JPEG_QUALITY,
        LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_IO_THREADS, N_WORKERS,
        OUT_ROOT, R_RANGE, START, TCWV_NONLIN,
    )


# ---------------------------------------------------------------------------
# Cell 4 — Dask scheduler config
# Uses the built-in threaded scheduler (no dask.distributed needed).
# HDF5/h5netcdf releases the GIL during reads, so 32 threads run in parallel.
# ---------------------------------------------------------------------------
@app.cell
def _(N_WORKERS, dask):
    dask.config.set(scheduler="threads", num_workers=N_WORKERS)
    print(f"Dask: threaded scheduler, {N_WORKERS} workers")
    return ()


# ---------------------------------------------------------------------------
# Cell 5 — Channel-encoding helpers + precompute_daily_channels
# Private helpers (_get, _rescale01, _to_uint8_img) must live in the same
# cell as any function that calls them — marimo treats _-prefixed names as
# cell-private and will not export them across cell boundaries.
# ---------------------------------------------------------------------------
@app.cell
def _(B_RANGE, G_RANGE, Image, R_RANGE, TCWV_NONLIN, np, xr):
    def _get(ds, names):
        for n in names:
            if n in ds:
                return ds[n]
        raise KeyError(f"Vars {list(ds.data_vars)} do not include any of {names}")

    def _rescale01(da, vmin, vmax, eps=1e-6, nonlin=None):
        x = (da - vmin) / (vmax - vmin + eps)
        x = x.clip(0, 1)
        if nonlin == "sqrt":
            x = xr.apply_ufunc(np.sqrt, x, dask="allowed")
        elif nonlin == "log":
            x = xr.apply_ufunc(lambda y: np.log1p(9 * y) / np.log(10), x, dask="allowed")
        return x

    def _to_uint8_img(arr01, width=896, height=256):
        arr01 = np.clip(np.asarray(arr01), 0, 1)
        if arr01.ndim == 2:
            arr01 = np.stack([arr01, arr01, arr01], axis=-1)
        img = Image.fromarray((arr01 * 255).astype(np.uint8))
        return img.resize((width, height), resample=Image.BILINEAR)

    def precompute_daily_channels(ds, mean_msl):
        """
        Compute R, G, B channel DataArrays (dask-backed, lazy).

        R = inverted MSLP anomaly  (lows = bright red)
        G = 10m wind speed magnitude
        B = TCWV (optionally nonlinear)

        Note: no rechunking to time:1 — save_batch_parallel uses batched
        dask.compute() aligned to DEFAULT_CHUNKS["time"] = BATCH_SIZE = 24.
        """
        msl  = _get(ds, ["msl", "MSL", "mslp", "msl_hPa"])
        u10  = _get(ds, ["u10", "VAR_10U"])
        v10  = _get(ds, ["v10", "VAR_10V"])
        tcwv = _get(ds, ["tcwv", "TCWV"])

        if msl.attrs.get("units", "").lower() in ("pa", "pascal", "pascals"):
            msl = msl / 100.0
            msl.attrs["units"] = "hPa"

        ws10 = xr.apply_ufunc(np.hypot, u10, v10, dask="parallelized",
                               output_dtypes=[u10.dtype])

        clim         = _get(mean_msl, ["MSL", "msl", "msl_mean", "msl_hPa"])
        doy_index    = msl["time"].dt.dayofyear
        clim_on_time = clim.sel(doy=doy_index).drop_vars("doy")
        msl          = msl.transpose("time", "latitude", "longitude")
        clim_on_time = clim_on_time.transpose("time", "latitude", "longitude")
        msl_anom     = msl - clim_on_time

        R = 1.0 - _rescale01(msl_anom, *R_RANGE)
        G = _rescale01(ws10, *G_RANGE)
        B = _rescale01(tcwv, *B_RANGE, nonlin=TCWV_NONLIN)

        return R, G, B

    return _to_uint8_img, precompute_daily_channels


# ---------------------------------------------------------------------------
# Cell 8 — save_batch_parallel
#
# Pipeline per batch:
#   1. dask.compute(R_batch, G_batch, B_batch)  — one optimized graph traversal
#      using N_WORKERS threads (h5netcdf releases GIL during reads)
#   2. ProcessPoolExecutor.submit(_save_rgb_worker, ...)  — PIL resize + JPEG
#      encode runs in real processes (no GIL), 24 frames/batch in parallel
#
# _ensure_dirs is defined here (not a separate cell) because it starts with _
# and marimo treats such names as cell-private.
# ---------------------------------------------------------------------------
@app.cell
def _(
    BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, JPEG_QUALITY, N_WORKERS, OUT_ROOT,
    Path, ProcessPoolExecutor, as_completed, dask, pd, tqdm,
):
    # Workers and GPU utilities live in e5_helpers.py so ProcessPoolExecutor
    # can pickle them — marimo compiles cells into isolated files, making
    # notebook-level definitions unpicklable.
    from e5_helpers import (
        batch_resize_torch as _batch_resize_torch,
        pick_device        as _pick_device,
        save_jpeg_worker   as _save_jpeg_worker,
    )

    def _ensure_dirs(root):
        root = Path(root)
        for sub in ("msl", "wmax", "tcwv", "rgb"):
            (root / sub).mkdir(parents=True, exist_ok=True)

    def save_batch_parallel(
        R, G, B,
        start, end,
        out_root=OUT_ROOT,
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
        quality=JPEG_QUALITY,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS,
    ):
        out_root = Path(out_root)
        _ensure_dirs(out_root)
        rgb_dir = out_root / "rgb"

        device  = _pick_device()
        times   = R.sel(time=slice(start, end)).time.values
        n_total = len(times)
        print(f"Device: {device} | {n_total} frames | batch={batch_size} | workers={n_workers}")

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            with tqdm(total=n_total, desc="Saving RGB frames", unit="frame") as pbar:
                for i in range(0, n_total, batch_size):
                    batch_times = times[i : i + batch_size]

                    # --- Step 1: compute all three channel arrays in one dask call ---
                    R_np, G_np, B_np = dask.compute(
                        R.sel(time=batch_times),
                        G.sel(time=batch_times),
                        B.sel(time=batch_times),
                    )

                    # --- Step 2: batch resize on GPU (one kernel for all frames) ---
                    # resized_batch: numpy uint8 (B, height, width, 3)
                    resized_batch = _batch_resize_torch(
                        R_np.values, G_np.values, B_np.values,
                        width, height, device,
                    )

                    # --- Step 3: submit JPEG encode to process pool ---
                    futures = [
                        pool.submit(
                            _save_jpeg_worker,
                            (
                                resized_batch[j],
                                str(rgb_dir / f"{pd.Timestamp(t).strftime('%Y%m%d_%H')}_rgb.jpeg"),
                                quality,
                            ),
                        )
                        for j, t in enumerate(batch_times)
                    ]

                    # --- Step 4: collect results and update progress bar ---
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"\n[skip] {e}")
                        pbar.update(1)

        print(f"\nDone: {n_total} images -> {rgb_dir}")

    return (save_batch_parallel,)


# ---------------------------------------------------------------------------
# Cell 10 — Visualization helper (unchanged)
# ---------------------------------------------------------------------------
@app.cell
def _(plt):
    def show_rgb_and_channels(rgb_da, parts=None):
        """Show the combined RGB plus three individual channels in a 2×2 grid."""
        arr = rgb_da.values  # (lat, lon, 3) in [0,1]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
        axes = axes.ravel()

        axes[0].imshow(
            arr, origin="upper",
            extent=[float(rgb_da.longitude.min()), float(rgb_da.longitude.max()),
                    float(rgb_da.latitude.min()),  float(rgb_da.latitude.max())],
            aspect="auto",
        )
        axes[0].set_title("RGB composite")

        cmap_list = ["Reds", "Greens", "Blues"]
        titles    = ["R: -MSLP anomaly", "G: max ws10", "B: TCWV"]
        for i in range(3):
            axes[i + 1].imshow(
                arr[:, :, i], origin="upper", cmap=cmap_list[i],
                extent=[float(rgb_da.longitude.min()), float(rgb_da.longitude.max()),
                        float(rgb_da.latitude.min()),  float(rgb_da.latitude.max())],
                aspect="auto",
            )
            axes[i + 1].set_title(titles[i])

        for ax in axes:
            ax.set_box_aspect(1)
            ax.set_xlabel("Longitude (0–360)")
            ax.set_ylabel("Latitude")

        plt.show()

    return (show_rgb_and_channels,)


# ---------------------------------------------------------------------------
# Cell 11 — Open dataset
# ---------------------------------------------------------------------------
@app.cell
def _(END, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_IO_THREADS, START, open_four):
    ds = open_four(
        START, END,
        lat_min=LAT_MIN, lat_max=LAT_MAX,
        lon_min=LON_MIN, lon_max=LON_MAX,
        n_io_threads=N_IO_THREADS,
    )
    return (ds,)


# ---------------------------------------------------------------------------
# Cell 12 — Inspect dataset
# ---------------------------------------------------------------------------
@app.cell
def _(ds):
    ds
    return


# ---------------------------------------------------------------------------
# Cell 13 — Load DOY climatology (MSLP)
# ---------------------------------------------------------------------------
@app.cell
def _(xr):
    file_msl = "/glade/work/ncheruku/research/era5_climatology/era5_climatology_1991_2020/MSL_Mean_1991_2020.zarr"
    mean_msl = xr.open_dataset(file_msl, engine="zarr")
    return file_msl, mean_msl


# ---------------------------------------------------------------------------
# Cell 14 — Precompute channel DataArrays (lazy, dask-backed)
# ---------------------------------------------------------------------------
@app.cell
def _(ds, mean_msl, precompute_daily_channels):
    R, G, B = precompute_daily_channels(ds, mean_msl)
    return B, G, R


# ---------------------------------------------------------------------------
# Cell 15 — Run the parallel save
# ---------------------------------------------------------------------------
@app.cell
def _(B, END, G, R, START, save_batch_parallel):
    save_batch_parallel(R, G, B, start=START, end=END)
    return


if __name__ == "__main__":
    app.run()
