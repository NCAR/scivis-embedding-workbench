import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notebook DAG

    Marimo notebooks are **reactive**: each cell is a node in a directed acyclic graph (DAG).
    When you change an input, only the cells that depend on it re-run — automatically, in the
    correct order, no matter how the cells are arranged on screen.

    Try the controls below, then **drag cells into a different order** in the editor to see
    that the reactive wiring stays intact.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    freq_slider = mo.ui.slider(1, 10, value=3, step=1, label="Frequency")
    amplitude_slider = mo.ui.slider(0.1, 2.0, value=1.0, step=0.1, label="Amplitude")
    reset_button = mo.ui.button(label="Reset to defaults", kind="neutral")
    mo.vstack([
        mo.md("### Controls"),
        freq_slider,
        amplitude_slider,
        reset_button,
    ])
    return amplitude_slider, freq_slider, reset_button


@app.cell
def _(amplitude_slider, freq_slider, mo, np, reset_button):
    # Reading reset_button.value creates a reactive dependency — clicking re-runs this cell
    _ = reset_button.value
    freq = freq_slider.value
    amp = amplitude_slider.value
    x = np.linspace(0, 2 * np.pi, 500)
    y = amp * np.sin(freq * x)
    mo.md(f"**Current values →** frequency: `{freq}`, amplitude: `{amp:.1f}`")
    return amp, freq, x, y


@app.cell
def _(amp, freq, plt, x, y):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, linewidth=2, color="#4C8BF5")
    ax.set_title(f"sin wave  |  freq={freq}  amp={amp:.1f}", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
        **How the DAG works here:**
        - **Controls cell** defines `freq_slider`, `amplitude_slider`, `reset_button`
        - **Derived values cell** reads those → computes `x`, `y`
        - **Plot cell** reads `x`, `y`, `freq`, `amp` → renders the chart

        Moving any of these cells up or down in the editor doesn't change execution order —
        Marimo always follows the dependency graph.
        """),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
