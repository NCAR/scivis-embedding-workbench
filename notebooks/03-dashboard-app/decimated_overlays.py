import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import hvplot.pandas
    import holoviews as hv
    from holoviews.operation.datashader import dynspread, datashade

    # Assuming 'df' is your existing dataframe with 'x', 'y', and 'label'
    # If not, let's create a dummy one for this cell to be runnable
    if 'df' not in globals():
        df = pd.DataFrame({
            'x': np.random.randn(50000),
            'y': np.random.randn(50000),
            'cluster': np.random.choice(['A', 'B', 'C'], 50000),
            'metadata': [f"Point_{i}" for i in range(50000)]
        })

    # Slider to control the decimated overlay size
    sample_slider = mo.ui.slider(
        start=100, 
        stop=5000, 
        step=100, 
        value=1000, 
        label="Interactive Sample Size"
    )
    sample_slider
    return df, hv, mo, sample_slider


@app.cell
def _(df, sample_slider):
    # 1. Background: All 50k points via Datashader + Dynspread
    # Update your plot code with these options
    bg_layer = df.hvplot.scatter(
        x='x', y='y', by='cluster',
        datashade=True, 
        dynspread=True
    ).opts(
        width=800, 
        height=500, 
        tools=['box_select', 'lasso_select', 'hover'], # Add these explicitly
        active_tools=['box_select']                   # Make it the default
    )

    # 2. Foreground: Decimated sample for hover/tooltips
    df_sample = df.sample(sample_slider.value)
    fg_layer = df_sample.hvplot.scatter(
        x='x', y='y',
        hover_cols=['metadata', 'cluster'],
        alpha=0.4,
        size=5,
        color='white', 
        use_index=False
    )

    # 3. Combine them 
    hybrid_plot = (bg_layer * fg_layer)

    # Simply output the plot object; Marimo handles the rest.
    hybrid_plot
    return (fg_layer,)


@app.cell
def _(fg_layer, hv, mo):
    # Create a pointer to the bounds of your selection
    selection_stream = hv.streams.BoundsXY(source=fg_layer)

    mo.md(
        f"""
        **Current Selection Bounds (for LanceDB query):**

        X-Range: {selection_stream.bounds[0] if selection_stream.bounds else 'None'} to {selection_stream.bounds[2] if selection_stream.bounds else 'None'}

        Y-Range: {selection_stream.bounds[1] if selection_stream.bounds else 'None'} to {selection_stream.bounds[3] if selection_stream.bounds else 'None'}
        """
    )
    return


if __name__ == "__main__":
    app.run()
