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

    return mo, plt


@app.function
def make_wave(freq: int, amp: float, n_points: int = 500):
    """Return (x, y) arrays for a sine wave with given frequency and amplitude."""
    import numpy as np
    x = np.linspace(0, 2 * np.pi, n_points)
    y = amp * np.sin(freq * x)
    return x, y


@app.cell
def _(mo):
    reset_button = mo.ui.button(label="Reset to defaults", kind="neutral")
    reset_button
    return (reset_button,)


@app.cell
def _(mo, reset_button):
    # Reading reset_button.value creates a reactive dependency — every click
    # invalidates this cell, which recreates the sliders at their default values.
    _ = reset_button.value
    freq_slider = mo.ui.slider(1, 10, value=3, step=1, label="Frequency")
    amplitude_slider = mo.ui.slider(0.1, 2.0, value=1.0, step=0.1, label="Amplitude")
    mo.vstack([
        mo.md("### Controls"),
        freq_slider,
        amplitude_slider,
    ])
    return amplitude_slider, freq_slider


@app.cell
def _(amplitude_slider, freq_slider, mo):
    freq = freq_slider.value
    amp = amplitude_slider.value
    # make_wave is defined with @app.function above — used here just like any function
    x, y = make_wave(freq, amp)
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
    return


if __name__ == "__main__":
    app.run()
