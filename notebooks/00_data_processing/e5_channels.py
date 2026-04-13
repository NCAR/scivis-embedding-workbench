import marimo

__generated_with = "0.10.0"
app = marimo.App()


@app.cell
def _():
    import glob
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt
    from PIL import Image

    return (
        Image, Path, ThreadPoolExecutor, as_completed,
        glob, np, pd, plt, threading, xr,
    )


@app.cell
def _(glob, pd, xr):
    # ---------- config ----------
    DATA_ROOT = "/glade/campaign/collections/gdex/data/d633000/e5.oper.an.sfc"
    DEFAULT_CHUNKS = {"time": 120, "latitude": -1, "longitude": -1}

    PATTERNS = {
        "MSL":    f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_151_msl.ll025sc.*.nc",
        "TCWV":   f"{DATA_ROOT}" + "/{}/e5.oper.an.sfc.128_137_tcwv.ll025sc.*.nc",
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
            # ERA5 latitude is descending (90→-90)
            if (lat_min is not None) and (lat_max is not None):
                ds = ds.sel(latitude=slice(lat_max, lat_min))
            # ERA5 longitude is native 0..360 (no wrapping)
            if (lon_min is not None) and (lon_max is not None):
                ds = ds.sel(longitude=slice(lon_min, lon_max))
            return ds
        return _pp

    def _open_var(name, pattern, start, end, *, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
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

    def open_four(start, end, *, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        parts = []
        for name, pat in PATTERNS.items():
            parts.append(_open_var(name, pat, start, end,
                                   lat_min=lat_min, lat_max=lat_max,
                                   lon_min=lon_min, lon_max=lon_max))
        return xr.merge(parts, compat="override", join="override")

    return DATA_ROOT, DEFAULT_CHUNKS, PATTERNS, open_four


@app.cell
def _(open_four):
    # Atlantic box in native 0..360 longitudes (≈ -110..-10 in -180..180)
    LAT_MIN, LAT_MAX = 15.0, 35.0
    LON_MIN, LON_MAX = 260.0, 330.0

    ds = open_four(
        "2017-06-01", "2017-12-01",
        lat_min=LAT_MIN, lat_max=LAT_MAX,
        lon_min=LON_MIN, lon_max=LON_MAX,
    )
    return LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, ds


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(plt):
    def show_rgb_and_channels(rgb_da, parts=None):
        """
        Show the combined RGB plus the three individual channels in a 2x2 grid.
        Each subplot is square, and the image is stretched to fill (no empty space).
        """
        arr = rgb_da.values  # (lat, lon, 3) in [0,1]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
        axes = axes.ravel()

        # --- RGB composite ---
        axes[0].imshow(
            arr, origin="upper",
            extent=[float(rgb_da.longitude.min()), float(rgb_da.longitude.max()),
                    float(rgb_da.latitude.min()),  float(rgb_da.latitude.max())],
            aspect="auto",
        )
        axes[0].set_title("RGB composite")

        # --- individual channels ---
        cmap_list = ["Reds", "Greens", "Blues"]
        titles = ["R: -MSLP anomaly", "G: max ws10", "B: TCWV"]

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


@app.cell
def _(Path):
    # ---------------- image pipeline config ----------------
    R_RANGE = (-20.0, 20.0)   # MSLP anomaly [hPa]
    G_RANGE = (0.0, 35.0)     # 10m wind daily max [m/s]
    B_RANGE = (20.0, 70.0)    # TCWV daily mean [kg m^-2]; set to (10,60) for mid-lats
    TCWV_NONLIN = "sqrt"      # None | "sqrt" | "log"

    OUT_ROOT = Path("./openclip_ready_512")
    OUT_ROOT.mkdir(exist_ok=True)
    (OUT_ROOT / "red").mkdir(exist_ok=True)
    (OUT_ROOT / "green").mkdir(exist_ok=True)
    (OUT_ROOT / "blue").mkdir(exist_ok=True)
    (OUT_ROOT / "rgb").mkdir(exist_ok=True)

    SAVE_SIZE = 512
    JPEG_QUALITY = 90
    N_WORKERS = 8  # threads for writing JPEGs (I/O bound)

    return B_RANGE, G_RANGE, JPEG_QUALITY, N_WORKERS, OUT_ROOT, R_RANGE, SAVE_SIZE, TCWV_NONLIN


@app.cell
def _(
    B_RANGE, G_RANGE, Image, JPEG_QUALITY, N_WORKERS, OUT_ROOT,
    Path, R_RANGE, SAVE_SIZE, TCWV_NONLIN,
    ThreadPoolExecutor, as_completed, np, pd, threading, xr,
):
    # ---------------- helpers ----------------
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

    def _to_uint8_img(arr01, size=512):
        arr01 = np.asarray(arr01)
        arr01 = np.clip(arr01, 0, 1)
        if arr01.ndim == 2:
            arr01 = np.stack([arr01, arr01, arr01], axis=-1)
        img = Image.fromarray((arr01 * 255).astype(np.uint8))
        return img.resize((size, size), resample=Image.BILINEAR)

    # ---------------- core: precompute once for whole period ----------------
    def precompute_daily_channels(ds, mean_msl):
        """
        ds: Dataset with msl, u10, v10, tcwv (time, latitude, longitude), native 0–360 lon okay
        mean_msl: Dataset with climatology on 'doy' (1..366) and same lat/lon
        Returns daily DataArrays (time, lat, lon): R*, G*, B* in 0..1
        """
        msl  = _get(ds, ["msl", "MSL", "mslp", "msl_hPa"])
        u10  = _get(ds, ["u10", "VAR_10U"])
        v10  = _get(ds, ["v10", "VAR_10V"])
        tcwv = _get(ds, ["tcwv", "TCWV"])

        if msl.attrs.get("units", "").lower() in ("pa", "pascal", "pascals"):
            msl = msl / 100.0
            msl.attrs["units"] = "hPa"

        ws10 = xr.apply_ufunc(np.hypot, u10, v10, dask="parallelized", output_dtypes=[u10.dtype])

        msl_daymean  = msl.resample(time="1D").mean()
        ws10_daymax  = ws10.resample(time="1D").max()
        tcwv_daymean = tcwv.resample(time="1D").mean()

        clim = _get(mean_msl, ["MSL", "msl", "msl_mean", "msl_hPa"])
        doy_index = msl_daymean["time"].dt.dayofyear
        clim_on_time = clim.sel(doy=doy_index).drop_vars("doy")
        msl_daymean  = msl_daymean.transpose("time", "latitude", "longitude")
        clim_on_time = clim_on_time.transpose("time", "latitude", "longitude")
        msl_anom = msl_daymean - clim_on_time

        R = 1.0 - _rescale01(msl_anom, *R_RANGE)                    # lows bright
        G = _rescale01(ws10_daymax, *G_RANGE)                        # intensity
        B = _rescale01(tcwv_daymean, *B_RANGE, nonlin=TCWV_NONLIN)  # moist footprint

        R = R.chunk({"time": 1})
        G = G.chunk({"time": 1})
        B = B.chunk({"time": 1})

        return R, G, B

    def _ensure_dirs(root):
        root = Path(root)
        (root / "msl").mkdir(parents=True, exist_ok=True)
        (root / "wmax").mkdir(parents=True, exist_ok=True)
        (root / "tcwv").mkdir(parents=True, exist_ok=True)
        (root / "rgb").mkdir(parents=True, exist_ok=True)

    # ---------------- saver: iterate only to write files ----------------
    def save_range_fast(R, G, B, start, end, out_root=OUT_ROOT, size=SAVE_SIZE, quality=JPEG_QUALITY):
        out_root = Path(out_root)
        _ensure_dirs(out_root)
        dates = pd.date_range(start=start, end=end, freq="D")
        total = len(dates)
        counter = 0
        lock = threading.Lock()

        def _save_one(ts):
            nonlocal counter
            ymd = pd.to_datetime(ts).strftime("%Y%m%d")
            try:
                r = R.sel(time=[ts]).isel(time=0)
                g = G.sel(time=[ts]).isel(time=0)
                b = B.sel(time=[ts]).isel(time=0)
                rgb = xr.concat([r, g, b], dim="channel").transpose("latitude", "longitude", "channel")
                img_rgb = _to_uint8_img(rgb.values, size=size)
                img_rgb.save(out_root / "rgb" / f"{ymd}_rgb.jpeg", "JPEG", quality=quality, optimize=True)
            except Exception as e:
                print(f"[skip] {ymd}: {e}")
            finally:
                with lock:
                    counter += 1
                    print(f"\r[{counter}/{total}] processed {ymd}", end="", flush=True)

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(_save_one, t) for t in dates]
            for f in as_completed(futures):
                f.result()
        print(f"\n✅ Finished writing {total} images to {out_root}/rgb/")

    def save_range_fast_single(R, start, end, out_root=OUT_ROOT, size=SAVE_SIZE, quality=JPEG_QUALITY):
        out_root = Path(out_root)
        _ensure_dirs(out_root)
        dates = pd.date_range(start=start, end=end, freq="D")

        def _save_one(ts):
            ymd = pd.to_datetime(ts).strftime("%Y%m%d")
            try:
                r = R.sel(time=[ts]).isel(time=0).values
                H, W = r.shape
                rgb = np.zeros((H, W, 3), dtype=np.float32)
                rgb[..., 0] = r
                img_rgb = _to_uint8_img(rgb, size=size)
                img_rgb.save(out_root / "rgb" / f"{ymd}_rgb.jpeg", "JPEG", quality=quality, optimize=True)
            except Exception as e:
                print(f"[skip] {ymd}: {e}")

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(_save_one, t) for t in dates]
            for f in futures:
                f.result()

    return precompute_daily_channels, save_range_fast, save_range_fast_single


@app.cell
def _(xr):
    # DOY climatology (Zarr) for MSLP
    file_msl = "/glade/work/ncheruku/research/era5_climatology/era5_climatology_1991_2020/MSL_Mean_1991_2020.zarr"
    mean_msl = xr.open_dataset(file_msl, engine="zarr")
    return file_msl, mean_msl


@app.cell
def _(ds, mean_msl, precompute_daily_channels):
    R, G, B = precompute_daily_channels(ds, mean_msl)
    return B, G, R


@app.cell
def _(B, G, R, save_range_fast):
    save_range_fast(R, G, B, start="2017-06-01", end="2017-12-01")
    return


@app.cell
def _():
    # save_range_fast_single(R, start="2017-06-01", end="2017-11-30")
    return


if __name__ == "__main__":
    app.run()
