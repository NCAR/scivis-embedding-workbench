"""
Standalone ERA5 RGB image generation script.

Extracts the full pipeline from the marimo notebook (e5_channels.py) into a
plain Python script suitable for HPC batch submission. The marimo notebook
remains for interactive exploration and parameter tuning.

Usage
-----
python e5_batch.py \\
    --start 2017-06-01 --end 2017-12-01 \\
    --lat-min 15 --lat-max 35 --lon-min 260 --lon-max 330 \\
    --out-root ./openclip_ready_896x256 \\
    --n-workers 32 --batch-size 24

PBS job script
--------------
#!/bin/bash
#PBS -N era5_rgb
#PBS -l select=1:ncpus=32:ngpus=1:mem=128GB
#PBS -l walltime=04:00:00
#PBS -q casper
#PBS -j oe

cd $PBS_O_WORKDIR
module load cuda
python e5_batch.py \\
    --start 2017-06-01 --end 2017-12-01 \\
    --n-workers 32 --batch-size 24

Notes
-----
- GPU resize uses torch.nn.functional.interpolate (CUDA if available, else CPU)
- CuPy zero-copy path activated automatically if cupy is installed:
    pip install cupy-cuda12x   # CUDA 12.x
    pip install cupy-cuda11x   # CUDA 11.x  (check: nvidia-smi | head -1)
- Dask threaded scheduler uses N_WORKERS threads; HDF5/h5netcdf releases
  the GIL during reads so all threads run concurrently
"""
import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from e5_helpers import batch_resize_torch, pick_device, save_jpeg_worker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT     = "/glade/campaign/collections/gdex/data/d633000/e5.oper.an.sfc"
DEFAULT_CHUNKS = {"time": 24, "latitude": -1, "longitude": -1}
CLIM_PATH     = ("/glade/work/ncheruku/research/era5_climatology"
                 "/era5_climatology_1991_2020/MSL_Mean_1991_2020.zarr")

PATTERNS = {
    "MSL":     f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_151_msl.ll025sc.*.nc",
    "TCWV":    f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_137_tcwv.ll025sc.*.nc",
    "VAR_10U": f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_165_10u.ll025sc.*.nc",
    "VAR_10V": f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_166_10v.ll025sc.*.nc",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _months(start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return [p.strftime("%Y%m") for p in pd.period_range(s, e, freq="M")]


def _files(pattern, start, end):
    fs = []
    for ym in _months(start, end):
        fs += glob.glob(pattern.format(ym))
    fs = sorted(fs)
    if not fs:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return fs


def _preproc(lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    def _pp(ds):
        if lat_min is not None and lat_max is not None:
            ds = ds.sel(latitude=slice(lat_max, lat_min))  # ERA5 lat is descending
        if lon_min is not None and lon_max is not None:
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


# ---------------------------------------------------------------------------
# Channel computation
# ---------------------------------------------------------------------------
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


def precompute_daily_channels(ds, mean_msl, *, r_range, g_range, b_range,
                               tcwv_nonlin="sqrt"):
    """
    Compute R, G, B channel DataArrays (dask-backed, lazy).

    R = inverted MSLP anomaly  (lows = bright red)
    G = 10m wind speed magnitude
    B = TCWV (optionally nonlinear)
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

    R = 1.0 - _rescale01(msl_anom, *r_range)
    G = _rescale01(ws10, *g_range)
    B = _rescale01(tcwv, *b_range, nonlin=tcwv_nonlin)

    return R, G, B


# ---------------------------------------------------------------------------
# Save pipeline
# ---------------------------------------------------------------------------
def _ensure_dirs(root):
    root = Path(root)
    for sub in ("msl", "wmax", "tcwv", "rgb"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def save_batch_parallel(R, G, B, start, end, *, out_root, width, height,
                        quality, batch_size, n_workers, device):
    out_root = Path(out_root)
    _ensure_dirs(out_root)
    rgb_dir = out_root / "rgb"

    times   = R.sel(time=slice(start, end)).time.values
    n_total = len(times)
    print(f"Device: {device} | {n_total} frames | batch={batch_size} | workers={n_workers}",
          flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        with tqdm(total=n_total, desc="Saving RGB frames", unit="frame",
                  file=sys.stdout) as pbar:
            for i in range(0, n_total, batch_size):
                batch_times = times[i : i + batch_size]

                # --- Step 1: compute all three channel arrays in one dask call ---
                R_np, G_np, B_np = dask.compute(
                    R.sel(time=batch_times),
                    G.sel(time=batch_times),
                    B.sel(time=batch_times),
                )

                # --- Step 2: batch resize on GPU (one kernel for all frames) ---
                resized_batch = batch_resize_torch(
                    R_np.values, G_np.values, B_np.values,
                    width, height, device,
                )  # numpy uint8 (B, height, width, 3)

                # --- Step 3: submit JPEG encode to process pool ---
                futures = [
                    pool.submit(
                        save_jpeg_worker,
                        (
                            resized_batch[j],
                            str(rgb_dir / f"{pd.Timestamp(t).strftime('%Y%m%d_%H')}_rgb.jpeg"),
                            quality,
                        ),
                    )
                    for j, t in enumerate(batch_times)
                ]

                # --- Step 4: collect results ---
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"\n[skip] {e}", flush=True)
                    pbar.update(1)

    print(f"\nDone: {n_total} images -> {rgb_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ERA5 RGB composite JPEG images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Time range
    p.add_argument("--start",    default="2017-06-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",      default="2017-12-01", help="End date (YYYY-MM-DD)")
    # Spatial domain
    p.add_argument("--lat-min",  type=float, default=15.0)
    p.add_argument("--lat-max",  type=float, default=35.0)
    p.add_argument("--lon-min",  type=float, default=260.0, help="0..360 convention")
    p.add_argument("--lon-max",  type=float, default=330.0, help="0..360 convention")
    # Channel encoding
    p.add_argument("--r-range",  type=float, nargs=2, default=[-20.0, 20.0],
                   metavar=("RMIN", "RMAX"), help="MSLP anomaly range [hPa]")
    p.add_argument("--g-range",  type=float, nargs=2, default=[0.0, 35.0],
                   metavar=("GMIN", "GMAX"), help="Wind speed range [m/s]")
    p.add_argument("--b-range",  type=float, nargs=2, default=[20.0, 70.0],
                   metavar=("BMIN", "BMAX"), help="TCWV range [kg/m^2]")
    p.add_argument("--tcwv-nonlin", default="sqrt", choices=["sqrt", "log", "none"],
                   help="Nonlinear transform for TCWV channel")
    # Output
    p.add_argument("--out-root", default="./openclip_ready_896x256")
    p.add_argument("--width",    type=int, default=896, help="Output image width px")
    p.add_argument("--height",   type=int, default=256, help="Output image height px")
    p.add_argument("--quality",  type=int, default=90,  help="JPEG quality (1-95)")
    # Parallelism
    p.add_argument("--n-workers",    type=int, default=32,
                   help="Dask threaded-scheduler workers + ProcessPoolExecutor size")
    p.add_argument("--n-io-threads", type=int, default=4,
                   help="Threads for opening 4 ERA5 variables concurrently")
    p.add_argument("--batch-size",   type=int, default=24,
                   help="Timesteps per dask.compute() call")
    # Climatology
    p.add_argument("--clim-path", default=CLIM_PATH,
                   help="Path to MSL climatology Zarr store")
    return p.parse_args()


def main():
    args = parse_args()

    tcwv_nonlin = None if args.tcwv_nonlin == "none" else args.tcwv_nonlin

    # --- configure dask ---
    dask.config.set(scheduler="threads", num_workers=args.n_workers)
    print(f"Dask: threaded scheduler, {args.n_workers} workers", flush=True)

    # --- pick compute device ---
    device = pick_device()

    # --- load ERA5 data ---
    print(f"Opening ERA5 data: {args.start} → {args.end}", flush=True)
    ds = open_four(
        args.start, args.end,
        lat_min=args.lat_min, lat_max=args.lat_max,
        lon_min=args.lon_min, lon_max=args.lon_max,
        n_io_threads=args.n_io_threads,
    )

    # --- load climatology ---
    print(f"Opening climatology: {args.clim_path}", flush=True)
    mean_msl = xr.open_dataset(args.clim_path, engine="zarr")

    # --- build lazy channel DataArrays ---
    R, G, B = precompute_daily_channels(
        ds, mean_msl,
        r_range=tuple(args.r_range),
        g_range=tuple(args.g_range),
        b_range=tuple(args.b_range),
        tcwv_nonlin=tcwv_nonlin,
    )

    # --- run the save pipeline ---
    save_batch_parallel(
        R, G, B,
        args.start, args.end,
        out_root=args.out_root,
        width=args.width,
        height=args.height,
        quality=args.quality,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        device=device,
    )


if __name__ == "__main__":
    main()
