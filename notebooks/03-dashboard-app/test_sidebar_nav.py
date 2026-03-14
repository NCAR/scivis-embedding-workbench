import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.sidebar(
        mo.vstack([
            mo.md("## SciVis Workbench"),
            mo.nav_menu(
                {
                    "#overview": "🏠 Overview",
                    "#embeddings": "🔍 Embeddings",
                    "#map": "🗺️ Map",
                },
                orientation="vertical",
            ),
            mo.md("---"),
            mo.md("*Shared sidebar across pages*"),
        ])
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # 🏠 Overview
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("This is a test notebook for `mo.sidebar` and `mo.nav_menu`. "
              "The sidebar on the left is visible in both edit and app mode."),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Navigation
    """)
    return


@app.cell
def _(mo):
    page = mo.ui.radio(
        options=["Overview", "Embeddings", "Map"],
        value="Overview",
        label="Active page",
        inline=True,
    )
    page
    return (page,)


@app.cell
def _(mo, page):
    pages = {
        "Overview": mo.vstack([
            mo.md("### Overview Page"),
            mo.md("Summary stats and dataset info would go here."),
        ]),
        "Embeddings": mo.vstack([
            mo.md("### Embeddings Page"),
            mo.md("Embedding explorer and experiment selector would go here."),
        ]),
        "Map": mo.vstack([
            mo.md("### Map Page"),
            mo.md("Spatial extent map and patch grid would go here."),
        ]),
    }
    mo.callout(pages[page.value], kind="neutral")
    return


if __name__ == "__main__":
    app.run()
