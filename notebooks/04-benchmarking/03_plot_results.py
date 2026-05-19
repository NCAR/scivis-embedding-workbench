import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # ANN Benchmark Results

    Interactive plots for `results/search_latency.csv` produced by
    `02_benchmark_ann_latency.py`.

    **Panels**
    - **Latency & Recall** — mean query latency (ms) with p95 / p99 shaded envelopes
      and mean Recall@10 on a secondary axis, plotted against index size (log scale).
    - **Memory (RSS)** — process RAM before and after the timed query loop, plus the
      raw float32 index footprint and a 16 GiB workstation reference line.
    - **QPS** — queries per second per experiment.
    - **Summary table** — all numeric columns from the CSV.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import pandas as pd

    return Path, np, pd, plt, ticker


@app.cell
def _(Path):
    RESULTS_DIR = Path(__file__).parent / "results"
    CSV_PATH    = RESULTS_DIR / "search_latency.csv"
    return (CSV_PATH,)


@app.cell
def _(CSV_PATH, mo, pd):
    if not CSV_PATH.exists():
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"**`{CSV_PATH}` not found.**\n\n"
                    "Run `02_benchmark_ann_latency.py` first to generate results."
                ),
                kind="warn",
            ),
        )

    df = pd.read_csv(CSV_PATH)

    # Guarantee ascending patch count order (smallest → largest index)
    df = df.sort_values("n_patches").reset_index(drop=True)

    mo.callout(
        mo.md(
            f"Loaded **{len(df)} experiments** from `{CSV_PATH.name}`.  "
            f"Resolutions: {', '.join(df['resolution'].tolist())}"
        ),
        kind="success",
    )
    return (df,)


@app.cell
def _(df, mo):
    _display_cols = [
        "experiment", "resolution", "n_patches", "db_size_gib",
        "num_partitions", "nprobes", "refine_factor",
        "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
        "qps", "mean_recall", "rss_before_gib", "rss_after_gib",
    ]
    _present = [c for c in _display_cols if c in df.columns]

    mo.md("### Summary Table"), mo.ui.table(
        df[_present].rename(columns={
            "n_patches":       "Patches",
            "db_size_gib":     "DB (GiB)",
            "num_partitions":  "Partitions",
            "nprobes":         "nprobes",
            "refine_factor":   "Refine",
            "mean_ms":         "Mean (ms)",
            "p50_ms":          "p50 (ms)",
            "p95_ms":          "p95 (ms)",
            "p99_ms":          "p99 (ms)",
            "max_ms":          "Max (ms)",
            "qps":             "QPS",
            "mean_recall":     "Recall@10",
            "rss_before_gib":  "RSS Before (GiB)",
            "rss_after_gib":   "RSS After (GiB)",
        }),
        selection=None,
    )
    return


@app.cell
def _(mo):
    show_p95_band  = mo.ui.checkbox(value=True,  label="p95 envelope")
    show_p99_band  = mo.ui.checkbox(value=True,  label="p99 envelope")
    show_labels    = mo.ui.checkbox(value=True,  label="Resolution labels")
    show_nprobes   = mo.ui.checkbox(value=True,  label="nprobes annotations")
    show_rss_panel = mo.ui.checkbox(value=True,  label="Show RSS panel")
    show_qps_panel = mo.ui.checkbox(value=True,  label="Show QPS panel")

    mo.md("### Display Options"), mo.hstack(
        [show_p95_band, show_p99_band, show_labels,
         show_nprobes, show_rss_panel, show_qps_panel],
        gap=2,
    )
    return (
        show_labels,
        show_nprobes,
        show_p95_band,
        show_p99_band,
        show_qps_panel,
        show_rss_panel,
    )


@app.cell
def _(
    df,
    mo,
    np,
    plt,
    show_labels,
    show_nprobes,
    show_p95_band,
    show_p99_band,
    show_qps_panel,
    show_rss_panel,
    ticker,
):
    # ── Colour palette ────────────────────────────────────────────────────────
    LATENCY_COLOR = "#2166ac"
    P95_COLOR     = "#2166ac"
    P99_COLOR     = "#b2182b"
    RECALL_COLOR  = "#1a9641"
    RSS_COLOR     = "#762a83"
    QPS_COLOR     = "#e08214"

    def _fmt_patches(n, _=None):
        return f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n / 1e3:.0f}K"

    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        10,
        "axes.labelsize":   11,
        "axes.titlesize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "axes.spines.top":  False,
        "figure.dpi":       150,
    })

    x_vals    = df["n_patches"].to_numpy(dtype=float)
    means     = df["mean_ms"].to_numpy()
    p50s      = df["p50_ms"].to_numpy()
    p95s      = df["p95_ms"].to_numpy()
    p99s      = df["p99_ms"].to_numpy()
    recalls   = df["mean_recall"].to_numpy()
    freqs     = df["resolution"].tolist()
    nprobes_l = df["nprobes"].tolist()

    rss_before = df["rss_before_gib"].to_numpy() if "rss_before_gib" in df.columns else np.full_like(means, np.nan)
    rss_after  = df["rss_after_gib"].to_numpy()  if "rss_after_gib"  in df.columns else np.full_like(means, np.nan)
    raw_gib    = x_vals * 768 * 4 / 1024**3
    qps        = df["qps"].to_numpy() if "qps" in df.columns else np.full_like(means, np.nan)

    n_panels = 1 + int(show_rss_panel.value) + int(show_qps_panel.value)
    height_ratios = [3] + ([1.6] if show_rss_panel.value else []) + ([1.4] if show_qps_panel.value else [])

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(7.5, 3.5 * n_panels),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.42},
        squeeze=False,
    )
    ax_lat = axes[0, 0]
    panel_idx = 1

    # ── Latency envelopes ─────────────────────────────────────────────────────
    if show_p99_band.value:
        ax_lat.fill_between(x_vals, p95s, p99s, alpha=0.18, color=P99_COLOR,
                            label="p99 envelope")
    if show_p95_band.value:
        ax_lat.fill_between(x_vals, means, p95s, alpha=0.28, color=P95_COLOR,
                            label="p95 envelope")
    ax_lat.plot(x_vals, means, "o-", color=LATENCY_COLOR, lw=2, ms=7,
                zorder=5, label="Mean latency")

    if show_labels.value:
        for xv, mv, freq in zip(x_vals, means, freqs):
            ax_lat.annotate(
                freq, xy=(xv, mv),
                textcoords="offset points", xytext=(-6, 9),
                ha="center", fontsize=8.5, color="#333", fontweight="bold",
            )
    if show_nprobes.value:
        for xv, mv, k in zip(x_vals, means, nprobes_l):
            ax_lat.annotate(
                f"k={k}", xy=(xv, mv),
                textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=7.5, color="#555",
            )

    ax_lat.set_xscale("log")
    ax_lat.set_xticks(x_vals)
    ax_lat.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_patches))
    ax_lat.xaxis.set_minor_locator(ticker.NullLocator())
    ax_lat.set_xlabel("Patches in Index")
    ax_lat.set_ylabel("Query Latency (ms)", color=LATENCY_COLOR)
    ax_lat.tick_params(axis="y", colors=LATENCY_COLOR)
    ax_lat.yaxis.label.set_color(LATENCY_COLOR)
    ax_lat.spines["left"].set_color(LATENCY_COLOR)
    ax_lat.set_ylim(bottom=0)
    ax_lat.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)

    # Recall on secondary y-axis
    ax_rec = ax_lat.twinx()
    ax_rec.plot(x_vals, recalls, "s--", color=RECALL_COLOR, lw=1.8, ms=7,
                zorder=6, label="Mean Recall@10")
    ax_rec.set_ylim(0.0, 1.05)
    ax_rec.set_ylabel("Mean Recall@10", color=RECALL_COLOR)
    ax_rec.tick_params(axis="y", colors=RECALL_COLOR)
    ax_rec.yaxis.label.set_color(RECALL_COLOR)
    ax_rec.spines["right"].set_color(RECALL_COLOR)
    ax_rec.spines["top"].set_visible(False)
    ax_rec.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )

    lines_lat, labs_lat = ax_lat.get_legend_handles_labels()
    lines_rec, labs_rec = ax_rec.get_legend_handles_labels()
    ax_lat.legend(lines_lat + lines_rec, labs_lat + labs_rec,
                  loc="upper left", framealpha=0.88)
    ax_lat.set_title(
        "IVF-PQ ANN Search: Latency & Recall@10 vs. Index Size\n"
        r"(cosine, $k$=10, nprobes = 5 % of partitions, 1 000 queries per experiment)",
        pad=10,
    )

    # ── RSS panel (optional) ──────────────────────────────────────────────────
    if show_rss_panel.value:
        ax_rss = axes[panel_idx, 0]
        panel_idx += 1

        ax_rss.fill_between(x_vals, rss_before, rss_after, alpha=0.25,
                            color=RSS_COLOR, label="RSS range (before → after)")
        ax_rss.plot(x_vals, rss_after,  "^-",  color=RSS_COLOR, lw=2, ms=7,
                    zorder=5, label="Peak RSS (after queries)")
        ax_rss.plot(x_vals, rss_before, "v--", color=RSS_COLOR, lw=1.4, ms=6,
                    alpha=0.6, zorder=4, label="Baseline RSS (before queries)")
        ax_rss.plot(x_vals, raw_gib,    ":",   color="#aaaaaa", lw=1.5,
                    zorder=3, label="Raw float32 size")
        ax_rss.axhline(16, color="#d6604d", lw=1.2, ls="--", zorder=2)
        ax_rss.text(x_vals[-1], 16, "  16 GiB RAM",
                    va="bottom", fontsize=8, color="#d6604d")

        ax_rss.set_xscale("log")
        ax_rss.set_xticks(x_vals)
        ax_rss.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_patches))
        ax_rss.xaxis.set_minor_locator(ticker.NullLocator())
        ax_rss.set_xlabel("Patches in Index")
        ax_rss.set_ylabel("Memory (GiB)", color=RSS_COLOR)
        ax_rss.tick_params(axis="y", colors=RSS_COLOR)
        ax_rss.yaxis.label.set_color(RSS_COLOR)
        ax_rss.spines["left"].set_color(RSS_COLOR)
        ax_rss.spines["top"].set_visible(False)
        ax_rss.set_ylim(bottom=0)
        ax_rss.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
        ax_rss.legend(loc="upper left", framealpha=0.88)
        ax_rss.set_title("Process RSS vs. Index Size  (mmap out-of-core access)", pad=6)

    # ── QPS panel (optional) ──────────────────────────────────────────────────
    if show_qps_panel.value:
        ax_qps = axes[panel_idx, 0]

        bar_x = np.arange(len(freqs))
        bars  = ax_qps.bar(bar_x, qps, color=QPS_COLOR, alpha=0.8, width=0.55,
                           zorder=3)
        ax_qps.bar_label(bars, fmt="%.1f", padding=3, fontsize=8.5)
        ax_qps.set_xticks(bar_x)
        ax_qps.set_xticklabels(
            [f"{f}\n{_fmt_patches(p)}" for f, p in zip(freqs, x_vals.astype(int))],
            fontsize=9,
        )
        ax_qps.set_ylabel("Queries / second", color=QPS_COLOR)
        ax_qps.tick_params(axis="y", colors=QPS_COLOR)
        ax_qps.yaxis.label.set_color(QPS_COLOR)
        ax_qps.spines["left"].set_color(QPS_COLOR)
        ax_qps.spines["top"].set_visible(False)
        ax_qps.set_ylim(bottom=0, top=max(qps) * 1.2)
        ax_qps.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
        ax_qps.set_title("Throughput (QPS) per Experiment", pad=6)

    fig.tight_layout()
    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
