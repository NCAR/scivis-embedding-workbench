import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import pandas as pd
    import numpy as np

    return mo, np, pd, px


@app.cell
def _(mo):
    # 1. Slider to test different data sizes (10k up to 1 Million)
    n_points_slider = mo.ui.slider(
        start=10000, 
        stop=1000000, 
        step=40000, 
        value=50000, 
        label="Number of Points:"
    )

    # 2. Dropdown for coloring
    color_options = {
        "Categorical (Groups)": "category",
        "Continuous (0-100)": "score",
        "Solid Color (None)": None
    }
    color_dropdown = mo.ui.dropdown(
        options=color_options,
        value="Categorical (Groups)",
        label="Color points by:"
    )

    # Display the controls
    mo.md(f"""
    ### Plotly WebGL Stress Test
    Adjust the controls below to test browser rendering and SSH-tunnel network lag.

    {mo.hstack([n_points_slider, color_dropdown])}
    """)
    return color_dropdown, n_points_slider


@app.cell
def _(mo, n_points_slider, np, pd):
    # Read the current value of the slider
    n = n_points_slider.value

    # Generate the synthetic data
    df = pd.DataFrame({
        "x": np.random.randn(n),
        "y": np.random.randn(n),
        "category": np.random.choice(["Group A", "Group B", "Group C"], n),
        "score": np.random.uniform(0, 100, n),
        "point_id": np.arange(n) # Crucial for our thin-frontend strategy
    })

    mo.md(f"*Generated synthetic dataset with **{n:,}** rows.*")
    return df, n


@app.cell
def _(color_dropdown, df, mo, n, px):
    # Get the selected color column (or None)
    selected_color = color_dropdown.value

    # Build the WebGL plot with Opacity added
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=selected_color,
        custom_data=["point_id"], 
        render_mode="webgl",
        opacity=0.5, # 1. TRANSPARENCY: Adjust this between 0.1 and 0.5
        title=f"Rendering {n:,} points with WebGL"
    )

    # 2 & 3. SHRINK AND REMOVE BORDERS
    fig.update_traces(
        marker=dict(
            size=3,              # Make dots smaller
            line=dict(width=0)   # Remove the dark outlines
        )
    )

    # A small optimization: disable animations
    fig.update_layout(
        transition_duration=0,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    # Wrap in Marimo UI for interactivity
    interactive_scatter = mo.ui.plotly(fig)

    # Display the plot
    interactive_scatter
    return (interactive_scatter,)


@app.cell
def _(interactive_scatter, mo):
    # Fetch the currently selected points from the UI element
    selected_data = interactive_scatter.value

    # Check if there is data, and assign the markdown to a variable
    if selected_data:
        num_selected = len(selected_data)
    
        # Let's peek at the first point to see how Plotly structures the data
        first_point = selected_data[0]
    
        output_text = mo.md(f"""
        ### Selection active!
        You successfully lassoed **{num_selected:,}** points.
    
        *Here is the raw data payload Plotly sends back for a single point:*
        ```python
        {first_point}
        ```
        """)
    else:
        output_text = mo.md("### Waiting for selection...\n*Draw a box or lasso on the plot above to test selection speed.*")

    # This MUST be the last line of the cell so Marimo displays it!
    output_text
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
