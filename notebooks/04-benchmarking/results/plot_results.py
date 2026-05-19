import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # ANN Benchmark Results

    **Dataset** — ERA5 2016–2018 · DINOv3 ViT-B/16 · 896 patches/image · 768-dim embeddings
    **Index** — IVF-PQ · cosine · 96 sub-vectors · nprobes = 5 % of partitions · refine factor = 50
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, pd


@app.cell
def _(Path, mo, pd):
    _csv = Path(__file__).parent / "search_latency.csv"
    if not _csv.exists():
        mo.stop(True, mo.callout(mo.md(f"**{_csv.name} not found.**"), kind="warn"))

    df = pd.read_csv(_csv).sort_values("n_patches").reset_index(drop=True)

    # Arrays
    x         = df["n_patches"].to_numpy(dtype=float)
    means     = df["mean_ms"].to_numpy()
    p95s      = df["p95_ms"].to_numpy()
    p99s      = df["p99_ms"].to_numpy()
    recalls   = df["mean_recall"].to_numpy() * 100   # percent
    qps       = df["qps"].to_numpy()
    rss_pre   = df["rss_before_gib"].to_numpy()
    rss_post  = df["rss_after_gib"].to_numpy()
    db_gib    = df["db_size_gib"].to_numpy()
    freqs     = df["resolution"].tolist()
    nprobes_l = df["nprobes"].tolist()
    raw_gib   = x * 768 * 4 / 1024**3

    x_tick = [f"{v/1e6:.2f} M" if v >= 1e6 else f"{v/1e3:.0f} K" for v in x]

    # Shared style
    FONT   = "Inter, system-ui, sans-serif"
    BG     = "#f8fafc"
    GRID   = "#e2e8f0"
    INK    = "#1e293b"
    BLUE   = "#3b82f6"
    GREEN  = "#10b981"
    AMBER  = "#f59e0b"
    VIOLET = "#8b5cf6"
    SLATE  = "#94a3b8"

    def base_layout(title):
        return dict(
            template="plotly_white",
            title=dict(text=title, x=0.5, xanchor="center",
                       font=dict(size=15, color=INK, family=FONT)),
            font=dict(family=FONT, size=12, color=INK),
            paper_bgcolor="#ffffff",
            plot_bgcolor=BG,
            margin=dict(l=60, r=40, t=60, b=60),
            height=380,
        )

    def base_xaxis(log=True):
        d = dict(showgrid=True, gridcolor=GRID, zeroline=False,
                 linecolor="#cbd5e1", tickfont=dict(size=11),
                 title=dict(text="Patches in index", font=dict(size=12)))
        if log:
            d.update(type="log", tickvals=x.tolist(), ticktext=x_tick)
        return d

    def base_yaxis(label):
        return dict(showgrid=True, gridcolor=GRID, zeroline=False,
                    linecolor="#cbd5e1", tickfont=dict(size=11),
                    title=dict(text=label, font=dict(size=12)))

    mo.callout(mo.md(
        f"**{len(df)} experiments** — " +
        " · ".join(f"`{r}` ({p/1e6:.1f} M)" for r, p in zip(freqs, x))
    ), kind="success")
    return (
        AMBER,
        BLUE,
        GREEN,
        SLATE,
        VIOLET,
        base_layout,
        base_xaxis,
        base_yaxis,
        df,
        freqs,
        means,
        nprobes_l,
        p95s,
        p99s,
        qps,
        raw_gib,
        recalls,
        rss_post,
        rss_pre,
        x,
    )


@app.cell
def _(
    BLUE,
    Path,
    base_layout,
    base_xaxis,
    base_yaxis,
    freqs,
    go,
    means,
    mo,
    nprobes_l,
    p95s,
    p99s,
    x,
):
    _fig = go.Figure()

    # p99 upper bound (invisible, anchor for fill)
    _fig.add_trace(go.Scatter(
        x=x, y=p99s, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    # p95–p99 band
    _fig.add_trace(go.Scatter(
        x=x, y=p95s, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(239,68,68,0.12)",
        name="p95–p99", hoverinfo="skip",
    ))
    # mean–p95 band
    _fig.add_trace(go.Scatter(
        x=x, y=p95s, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    _fig.add_trace(go.Scatter(
        x=x, y=means, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(59,130,246,0.13)",
        name="mean–p95", hoverinfo="skip",
    ))
    # Mean line
    _fig.add_trace(go.Scatter(
        x=x, y=means, mode="lines+markers+text",
        name="Mean latency",
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=9, color=BLUE, line=dict(color="white", width=2)),
        text=[f"<b>{f}</b>  k={k}" for f, k in zip(freqs, nprobes_l)],
        textposition="top center",
        textfont=dict(size=10, color="#334155"),
        hovertemplate="%{text}<br>Mean: %{y:.1f} ms<extra></extra>",
    ))

    _fig.update_layout(
        **base_layout("Query Latency vs. Index Size"),
        xaxis=base_xaxis(),
        yaxis=dict(**base_yaxis("Latency (ms)"), rangemode="tozero"),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )

    _out = Path(__file__).parent
    _fig.write_html(str(_out / "fig1_latency.html"))
    mo.vstack([mo.md("### Query Latency"), _fig])
    return


@app.cell
def _(
    GREEN,
    Path,
    base_layout,
    base_xaxis,
    base_yaxis,
    freqs,
    go,
    mo,
    recalls,
    x,
):
    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=x, y=recalls, mode="lines+markers+text",
        name="Recall@10",
        line=dict(color=GREEN, width=2.5),
        marker=dict(size=9, color=GREEN, line=dict(color="white", width=2)),
        text=[f"<b>{f}</b>" for f in freqs],
        textposition="top center",
        textfont=dict(size=10, color="#334155"),
        hovertemplate="%{text}<br>Recall@10: %{y:.1f}%%<extra></extra>",
    ))

    _fig.add_hline(y=100, line=dict(color="#94a3b8", width=1, dash="dot"))

    _fig.update_layout(
        **base_layout("Recall@10 vs. Index Size"),
        xaxis=base_xaxis(),
        yaxis=dict(**base_yaxis("Recall@10 (%)"), range=[0, 110]),
    )

    _out = Path(__file__).parent
    _fig.write_html(str(_out / "fig2_recall.html"))
    mo.vstack([mo.md("### Recall@10"), _fig])
    return


@app.cell
def _(AMBER, Path, base_layout, base_xaxis, base_yaxis, freqs, go, mo, qps, x):
    _fig = go.Figure()

    _fig.add_trace(go.Bar(
        x=x, y=qps,
        marker=dict(color=AMBER, line=dict(width=0)),
        text=[f"{v:.1f}" for v in qps],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="%{text} q/s<extra></extra>",
        customdata=freqs,
    ))

    _fig.update_layout(
        **base_layout("Throughput (QPS) vs. Index Size"),
        xaxis={**base_xaxis(), "ticktext": freqs},
        yaxis=dict(**base_yaxis("Queries / second"), rangemode="tozero"),
        bargap=0.4,
    )

    _out = Path(__file__).parent
    _fig.write_html(str(_out / "fig3_qps.html"))
    mo.vstack([mo.md("### Throughput (QPS)"), _fig])
    return


@app.cell
def _(
    Path,
    SLATE,
    VIOLET,
    base_layout,
    base_xaxis,
    base_yaxis,
    freqs,
    go,
    mo,
    raw_gib,
    rss_post,
    rss_pre,
    x,
):
    _fig = go.Figure()

    # ── Raw float32 area (background reference) ───────────────────────────────
    # Filled to zero so it reads as "how big the index would be if fully loaded"
    _fig.add_trace(go.Scatter(
        x=x, y=raw_gib,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(148,163,184,0.08)",
        line=dict(color=SLATE, width=1, dash="dot"),
        name="Raw float32 index (uncompressed)",
        hovertemplate="<b>%{x}</b><br>Uncompressed: %{y:.1f} GB<extra></extra>",
    ))

    # ── Shaded band between RSS before and after ──────────────────────────────
    # Correct fill="tonexty" approach: draw lower bound first, then upper bound
    _fig.add_trace(go.Scatter(
        x=x, y=rss_pre,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    _fig.add_trace(go.Scatter(
        x=x, y=rss_post,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(139,92,246,0.15)",
        name="RSS growth during queries", hoverinfo="skip",
    ))

    # ── RSS baseline (before queries) ─────────────────────────────────────────
    _fig.add_trace(go.Scatter(
        x=x, y=rss_pre,
        mode="lines+markers",
        line=dict(color=VIOLET, width=1.8, dash="dash"),
        marker=dict(size=7, color=VIOLET, line=dict(color="white", width=1.5)),
        name="RSS before queries",
        hovertemplate="<b>%{customdata}</b><br>Baseline RSS: %{y:.2f} GB<extra></extra>",
        customdata=freqs,
    ))

    # ── Peak RSS (after queries) with resolution labels ───────────────────────
    _delta = [f"+{(post - pre):.2f} GB" for pre, post in zip(rss_pre, rss_post)]
    _fig.add_trace(go.Scatter(
        x=x, y=rss_post,
        mode="lines+markers+text",
        line=dict(color=VIOLET, width=2.5),
        marker=dict(size=9, color=VIOLET, line=dict(color="white", width=2)),
        text=[f"<b>{f}</b>" for f in freqs],
        textposition="top center",
        textfont=dict(size=10, color="#4c1d95"),
        name="Peak RSS after queries",
        customdata=list(zip(freqs, _delta)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Peak RSS: %{y:.2f} GB<br>"
            "Growth: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))

    # ── 16 GiB workstation reference ──────────────────────────────────────────
    _fig.add_hline(
        y=16,
        line=dict(color="#f87171", width=1.2, dash="dash"),
        annotation_text="16 GB (typical workstation RAM)",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#f87171"),
    )

    _fig.update_layout(
        **{**base_layout("IVF-PQ Search Memory: Only Probed Partitions Load into RAM"), "height": 420},
        xaxis=base_xaxis(),
        yaxis=dict(**base_yaxis("Process RSS in Gigabytes (GB)"), rangemode="tozero"),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1),

    )

    _out = Path(__file__).parent
    _fig.write_html(str(_out / "fig4_rss.html"))
    mo.vstack([mo.md("### Memory (RSS)"), _fig])
    return


@app.cell
def _(means, p95s, p99s, raw_gib, recalls, rss_post, rss_pre, x):
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots
    import marimo as _mo
    from pathlib import Path as _Path

    # ── Cell-Private Constants (Prevents Marimo Redefinition Errors) ──────────
    _SLATE = "#64748b"
    _VIOLET = "#8b5cf6"
    _GREEN = "#059669"
    _BLUE = "#2563eb"
    _RED = "#dc2626"
    _GRID = "#e2e8f0"
    _INK = "#0f172a"
    _BG = "#f8fafc"
    _FONT = "Helvetica" 

    # Create a 2-row subplot, enabling a secondary y-axis on the bottom row
    _fig = _make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08, 
        row_heights=[0.45, 0.55],
        specs=[[{"secondary_y": False}], 
               [{"secondary_y": True}]]
    )

    # ── Format X-Axis Labels Consistently to Max 2 Decimals ───────────────────
    # .2f confines to 2 decimals, rstrip cleans up trailing zeros for whole numbers
    _x_tick_fmt = [f"{val / 1e6:.2f}".rstrip('0').rstrip('.') for val in x.tolist()]

    _xs = dict(
        showgrid=True, gridcolor=_GRID, zeroline=False, linecolor="#cbd5e1",
        type="log", tickvals=x.tolist(), ticktext=_x_tick_fmt, tickfont=dict(size=10),
    )

    # ── Data Labels ───────────────────────────────────────────────────────────
    _text_rss = [f"{v:.1f}" for v in rss_post]
    _text_recall = [f"{v:.1f}" for v in recalls]
    _text_latency = [f"{v:.0f}" for v in means]

    # ── Row 1: RSS ────────────────────────────────────────────────────────────
    _fig.add_trace(_go.Scatter(
        x=x, y=raw_gib, mode="lines", fill="tozeroy",
        fillcolor="rgba(148,163,184,0.08)", line=dict(color=_SLATE, width=1.5, dash="dot"),
        name="Raw float32",
    ), row=1, col=1)

    _fig.add_trace(_go.Scatter(
        x=x, y=rss_pre, mode="lines", line=dict(width=0), showlegend=False,
    ), row=1, col=1)

    _fig.add_trace(_go.Scatter(
        x=x, y=rss_post, mode="lines", fill="tonexty", fillcolor="rgba(139,92,246,0.15)",
        line=dict(width=0), name="RSS growth",
    ), row=1, col=1)

    _fig.add_trace(_go.Scatter(
        x=x, y=rss_pre, mode="lines+markers", line=dict(color=_VIOLET, width=1.5, dash="dash"),
        marker=dict(size=5, color=_VIOLET, line=dict(color="white", width=1)), name="RSS before",
    ), row=1, col=1)

    _fig.add_trace(_go.Scatter(
        x=x, y=rss_post, mode="lines+markers+text", line=dict(color=_VIOLET, width=2),
        marker=dict(size=6, color=_VIOLET, line=dict(color="white", width=1)), name="Peak RSS",
        text=_text_rss, textposition="top left", textfont=dict(size=8, color=_VIOLET)
    ), row=1, col=1)

    _fig.add_hline(y=16, row=1, col=1, line=dict(color=_RED, width=1.5, dash="dash"))

    # ── Row 2: Recall (Left Axis) & Latency (Right Axis) ──────────────────────
    _fig.add_trace(_go.Scatter(
        x=x, y=recalls, mode="lines+markers+text", line=dict(color=_GREEN, width=2),
        marker=dict(size=6, color=_GREEN, line=dict(color="white", width=1)), name="Recall@10",
        text=_text_recall, textposition="bottom right", textfont=dict(size=8, color=_GREEN)
    ), row=2, col=1, secondary_y=False)

    _fig.add_hline(y=100, row=2, col=1, secondary_y=False, line=dict(color="#94a3b8", width=1, dash="dot"))

    _fig.add_trace(_go.Scatter(
        x=x, y=p99s, mode="lines", line=dict(width=0), showlegend=False,
    ), row=2, col=1, secondary_y=True)

    _fig.add_trace(_go.Scatter(
        x=x, y=p95s, mode="lines", fill="tonexty", fillcolor="rgba(239,68,68,0.12)",
        line=dict(width=0), name="p95–p99",
    ), row=2, col=1, secondary_y=True)

    _fig.add_trace(_go.Scatter(
        x=x, y=p95s, mode="lines", line=dict(width=0), showlegend=False,
    ), row=2, col=1, secondary_y=True)

    _fig.add_trace(_go.Scatter(
        x=x, y=means, mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,0.13)",
        line=dict(width=0), name="mean–p95",
    ), row=2, col=1, secondary_y=True)

    _fig.add_trace(_go.Scatter(
        x=x, y=means, mode="lines+markers+text", line=dict(color=_BLUE, width=2),
        marker=dict(size=6, color=_BLUE, line=dict(color="white", width=1)), name="Mean latency",
        text=_text_latency, textposition="top center", textfont=dict(size=8, color=_BLUE)
    ), row=2, col=1, secondary_y=True)


    # ── Layout and Formatting ─────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        font=dict(family=_FONT, size=11, color=_INK),
        paper_bgcolor="#ffffff", plot_bgcolor=_BG,

        width=400,
        height=350,
        margin=dict(l=40, r=40, t=65, b=40),

        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0)",
        ),

        xaxis=dict(**_xs, showticklabels=False),
        xaxis2=dict(**_xs, title=dict(text="Number of Vectors (Millions)", font=dict(size=11))),
    )

    _fig.update_yaxes(title_text="RSS (GB)", tickfont=dict(size=10), title_font=dict(size=11), rangemode="tozero", row=1, col=1)
    _fig.update_yaxes(title_text="Recall (%)", tickfont=dict(size=10, color=_GREEN), title_font=dict(size=11, color=_GREEN), range=[0, 110], row=2, col=1, secondary_y=False)

    # Fixed Latency Axis to 150ms to prevent line overlap with Recall
    _fig.update_yaxes(title_text="Latency (ms)", tickfont=dict(size=10, color=_BLUE), title_font=dict(size=11, color=_BLUE), range=[0, 150], row=2, col=1, secondary_y=True)

    # ── Export & Render ───────────────────────────────────────────────────────
    _out = _Path(__file__).parent
    _fig.write_html(str(_out / "fig_combined.html"))
    try:
        _fig.write_image(str(_out / "fig_combined.pdf"))
        _fig.write_image(str(_out / "fig_combined.png"), scale=3)
        _note = "_Saved → `fig_combined.html` · `.pdf` · `.png`_"
    except Exception:
        _note = "_Saved → `fig_combined.html`_"

    _mo.vstack([_mo.md("### Figure Export"), _fig, _mo.md(_note)])

    # ── Export ────────────────────────────────────────────────────────────────
    # Use .cwd() so it saves in the exact same folder as your Marimo notebook!
    _out = _Path.cwd()

    print(f"Saving figures to: {_out}") # This will print the exact folder path

    try:
        _fig.write_html(str(_out / "fig_combined.html"))
        _fig.write_image(str(_out / "fig_combined.pdf"))
        _fig.write_image(str(_out / "fig_combined.png"), scale=6)
        _fig.write_image(str(_out / "fig_combined.eps"))
        print("Export successful!")
    except Exception as e:
        print(f"Image export failed: {e}")

    # ── Render in Marimo ──────────────────────────────────────────────────────
    _fig
    return


@app.cell
def _(means, p95s, p99s, raw_gib, recalls, rss_post, rss_pre, x):
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots
    import marimo as _mo
    from pathlib import Path as _Path

    # ── Cell-Private Constants (Prevents Marimo Redefinition Errors) ──────────
    _SLATE = "#64748b"
    _VIOLET = "#8b5cf6"
    _GREEN = "#059669"
    _BLUE = "#2563eb"
    _RED = "#dc2626"
    _GRID = "#e2e8f0"
    _INK = "#0f172a"
    _BG = "#f8fafc"
    _FONT = "Helvetica" 

    # Create a 2-row subplot, enabling a secondary y-axis on the bottom row
    _fig = _make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08, 
        row_heights=[0.45, 0.55],
        specs=[[{"secondary_y": False}], 
               [{"secondary_y": True}]]
    )

    # ── Format X-Axis Labels Consistently to Max 2 Decimals ───────────────────
    # .2f confines to 2 decimals, rstrip cleans up trailing zeros for whole numbers
    _x_tick_fmt = [f"{val / 1e6:.2f}".rstrip('0').rstrip('.') for val in x.tolist()]

    _xs = dict(
        showgrid=True, gridcolor=_GRID, zeroline=False, linecolor="#cbd5e1",
        type="log", tickvals=x.tolist(), ticktext=_x_tick_fmt, tickfont=dict(size=10),
    )

    # ── Data Labels ───────────────────────────────────────────────────────────
    _text_rss = [f"{v:.1f}" for v in rss_post]
    _text_recall = [f"{v:.1f}" for v in recalls]
    _text_latency = [f"{v:.0f}" for v in means]

    # ── Row 1: RSS ────────────────────────────────────────────────────────────
    _fig.add_trace(_go.Scatter(
        x=x, y=raw_gib, mode="lines", fill="tozeroy",
        fillcolor="rgba(148,163,184,0.08)", line=dict(color=_SLATE, width=1.5, dash="dot"),
        name="Raw float32",
    ), row=1, col=1)

    _fig.add_trace(_go.Scatter(
        x=x, y=rss_pre, mode="lines", line=dict(width=0), showlegend=False,
    ), row=1, col=1)

    _fig.add_trace(_go.Scatter(
        x=x, y=rss_post, mode="lines", fill="tonexty", fillcolor="rgba(139,92,246,0.15)",
        line=dict(width=0), name="RSS growth",
    ), row=1, col=1)

    _fig.add_trace(_go.Scatter(
        x=x, y=rss_pre, mode="lines+markers", line=dict(color=_VIOLET, width=1.5, dash="dash"),
        marker=dict(size=5, color=_VIOLET, line=dict(color="white", width=1)), name="RSS before",
    ), row=1, col=1)

    _fig.add_trace(_go.Scatter(
        x=x, y=rss_post, mode="lines+markers+text", line=dict(color=_VIOLET, width=2),
        marker=dict(size=6, color=_VIOLET, line=dict(color="white", width=1)), name="Peak RSS",
        text=_text_rss, textposition="top left", textfont=dict(size=8, color=_VIOLET)
    ), row=1, col=1)

    _fig.add_hline(y=16, row=1, col=1, line=dict(color=_RED, width=1.5, dash="dash"))

    # ── Row 2: Recall (Left Axis) & Latency (Right Axis) ──────────────────────
    _fig.add_trace(_go.Scatter(
        x=x, y=recalls, mode="lines+markers+text", line=dict(color=_GREEN, width=2),
        marker=dict(size=6, color=_GREEN, line=dict(color="white", width=1)), name="Recall@10",
        text=_text_recall, textposition="bottom right", textfont=dict(size=8, color=_GREEN)
    ), row=2, col=1, secondary_y=False)

    _fig.add_hline(y=100, row=2, col=1, secondary_y=False, line=dict(color="#94a3b8", width=1, dash="dot"))

    _fig.add_trace(_go.Scatter(
        x=x, y=p99s, mode="lines", line=dict(width=0), showlegend=False,
    ), row=2, col=1, secondary_y=True)

    _fig.add_trace(_go.Scatter(
        x=x, y=p95s, mode="lines", fill="tonexty", fillcolor="rgba(239,68,68,0.12)",
        line=dict(width=0), name="p95–p99",
    ), row=2, col=1, secondary_y=True)

    _fig.add_trace(_go.Scatter(
        x=x, y=p95s, mode="lines", line=dict(width=0), showlegend=False,
    ), row=2, col=1, secondary_y=True)

    _fig.add_trace(_go.Scatter(
        x=x, y=means, mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,0.13)",
        line=dict(width=0), name="mean–p95",
    ), row=2, col=1, secondary_y=True)

    _fig.add_trace(_go.Scatter(
        x=x, y=means, mode="lines+markers+text", line=dict(color=_BLUE, width=2),
        marker=dict(size=6, color=_BLUE, line=dict(color="white", width=1)), name="Mean latency",
        text=_text_latency, textposition="top center", textfont=dict(size=8, color=_BLUE)
    ), row=2, col=1, secondary_y=True)


    # ── Layout and Formatting ─────────────────────────────────────────────────
    _fig.update_layout(
        template="plotly_white",
        font=dict(family=_FONT, size=11, color=_INK),
        paper_bgcolor="#ffffff", plot_bgcolor=_BG,

        width=400,
        height=350,
        # Increased the right margin (r=130) to shrink the subplots and make room for the legend
        margin=dict(l=40, r=130, t=65, b=40),

        legend=dict(
            orientation="v",       # Stacked vertically
            yanchor="top", y=1.0,  # Align top of legend with top of plotting area
            xanchor="left", x=1.1, # Push to the right, outside the secondary Y-axis
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0)",
        ),

        xaxis=dict(**_xs, showticklabels=False),
        xaxis2=dict(**_xs, title=dict(text="Number of Vectors (Millions)", font=dict(size=11))),
    )

    _fig.update_yaxes(title_text="RSS (GB)", tickfont=dict(size=10), title_font=dict(size=11), rangemode="tozero", row=1, col=1)
    _fig.update_yaxes(title_text="Recall (%)", tickfont=dict(size=10, color=_GREEN), title_font=dict(size=11, color=_GREEN), range=[0, 110], row=2, col=1, secondary_y=False)

    # Fixed Latency Axis to 150ms to prevent line overlap with Recall
    _fig.update_yaxes(title_text="Latency (ms)", tickfont=dict(size=10, color=_BLUE), title_font=dict(size=11, color=_BLUE), range=[0, 150], row=2, col=1, secondary_y=True)

    # ── Export & Render ───────────────────────────────────────────────────────
    _out = _Path(__file__).parent
    _fig.write_html(str(_out / "fig_combined.html"))
    try:
        _fig.write_image(str(_out / "fig_combined.pdf"))
        _fig.write_image(str(_out / "fig_combined.png"), scale=3)
        _note = "_Saved → `fig_combined.html` · `.pdf` · `.png`_"
    except Exception:
        _note = "_Saved → `fig_combined.html`_"

    _mo.vstack([_mo.md("### Figure Export"), _fig, _mo.md(_note)])

    # ── Export ────────────────────────────────────────────────────────────────
    # Use .cwd() so it saves in the exact same folder as your Marimo notebook!
    _out = _Path.cwd()

    print(f"Saving figures to: {_out}") # This will print the exact folder path

    try:
        _fig.write_html(str(_out / "fig_combined.html"))
        _fig.write_image(str(_out / "fig_combined.pdf"))
        _fig.write_image(str(_out / "fig_combined.png"), scale=6)
        _fig.write_image(str(_out / "fig_combined.eps"))
        print("Export successful!")
    except Exception as e:
        print(f"Image export failed: {e}")

    # ── Render in Marimo ──────────────────────────────────────────────────────
    _fig
    return


@app.cell
def _(df, mo):
    _cols = ["resolution", "n_patches", "db_size_gib", "num_partitions",
             "nprobes", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
             "qps", "mean_recall", "rss_before_gib", "rss_after_gib"]
    _present = [c for c in _cols if c in df.columns]
    mo.vstack([
        mo.md("### Summary"),
        mo.ui.table(df[_present].rename(columns={
            "resolution": "Res.", "n_patches": "Patches", "db_size_gib": "DB (GiB)",
            "num_partitions": "Parts.", "nprobes": "nprobes",
            "mean_ms": "Mean (ms)", "p50_ms": "p50", "p95_ms": "p95",
            "p99_ms": "p99", "qps": "QPS", "mean_recall": "Recall@10",
            "rss_before_gib": "RSS₀ GiB", "rss_after_gib": "RSS₁ GiB",
        }), selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
