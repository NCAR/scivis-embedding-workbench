"""marimo widget factories for the latent-space-exploration notebook.

The only module here that imports marimo, so the notebook can stay a sequence
of calls.

Widget groups are UIElements (`mo.md(...).batch(...)` or `mo.ui.dictionary`)
rather than plain dicts: a plain dict is inert, and cells reading it would
never re-run when a widget changes. They are always *displayed* through their
individual elements -- rendering a group directly shows marimo's JSON-tree
view.
"""

import marimo as mo


def loader_controls():
    """Sampling widgets, applied immediately.

    Deliberately separate from the display controls: these drive a
    multi-second read of the patch table, and keeping them in their own group
    means nudging a colour or zoom never re-triggers the load. Read values as
    `.value["sample_size"]`.
    """
    return mo.md("{sample_size} &nbsp;&nbsp; {random_sample}").batch(
        sample_size=mo.ui.slider(
            start=10_000,
            stop=500_000,
            step=10_000,
            value=50_000,
            label="Patches to load",
            show_value=True,
        ),
        random_sample=mo.ui.dropdown(
            options={"random": True, "head": False},
            value="random",
            label="Sampling",
        ),
    )


def display_widgets(defaults=None):
    """Gallery appearance widgets, keyed by `DisplayOptions` field name.

    Returns a `mo.ui.dictionary`, not a plain dict: a dictionary is itself a
    UIElement, so cells reading its `.value` re-run when any widget changes. A
    plain dict is inert and dependent cells would never update.

    The colour picker is *not* in here -- it is an anywidget, and anywidgets
    cannot be copied, which `mo.ui.dictionary` requires. Build it separately
    with `color_picker()`.
    """
    from helpers.experiment import DisplayOptions

    d = defaults or DisplayOptions()
    return mo.ui.dictionary({
        "n_examples": mo.ui.slider(
            start=4, stop=200, step=4, value=d.n_examples,
            label="Patches", show_value=True,
        ),
        "buffer_patches": mo.ui.slider(
            start=0, stop=6, step=1, value=d.buffer_patches,
            label="Context (patches)", show_value=True,
        ),
        "zoom": mo.ui.slider(
            start=1, stop=12, step=1, value=d.zoom,
            label="Zoom", show_value=True,
        ),
        "border_width": mo.ui.slider(
            start=0, stop=10, step=1, value=d.border_width,
            label="Border px (0 = off)", show_value=True,
        ),
        "show_preview": mo.ui.checkbox(
            value=d.show_preview, label="Hover preview",
        ),
    })


def display_panel(widgets, picker=None):
    """Lay the display widgets out in two rows.

    Renders the dictionary's elements individually -- displaying the
    `mo.ui.dictionary` itself would show marimo's JSON-tree view.
    """
    items = [widgets[k] for k in widgets]
    if picker is not None:
        items.append(picker)
    half = (len(items) + 1) // 2
    return mo.vstack(
        [
            mo.hstack(items[:half], justify="start", align="center"),
            mo.hstack(items[half:], justify="start", align="center"),
        ],
        gap=0.5,
    )


def widget_values(widgets, picker=None) -> dict:
    """Current values of `display_widgets()`, as `DisplayOptions` kwargs.

    `picker` is the separate colour anywidget; its hex is folded in under
    `border_color`.
    """
    values = dict(widgets.value)
    if picker is not None:
        values["border_color"] = picker.color
    return values


def display_options(values, defaults=None):
    """Turn widget values into `DisplayOptions`.

    Empty or None input falls back to defaults, so the gallery still renders
    before the widgets exist. Keys not present keep their dataclass default --
    `resample` has no widget, for instance.
    """
    from helpers.experiment import DisplayOptions

    if not values:
        return defaults or DisplayOptions()
    values = dict(values)
    color = values.get("border_color")
    if isinstance(color, dict):
        values["border_color"] = color.get("color", DisplayOptions.border_color)
    return DisplayOptions(**values)


def projection_picker(tables, value=None):
    """Dropdown of the projection tables found in the experiment."""
    return mo.ui.dropdown(
        options=list(tables),
        value=value or (tables[0] if tables else None),
        label="Projection",
    )


def color_by_dropdown(options, value: str = "density"):
    """Dropdown of colour-by columns.

    `density` is always present and always first: it needs nothing but x/y, so
    it stays valid for a projection with no clusters or metadata at all.
    """
    options = list(options)
    return mo.ui.dropdown(
        options=options,
        value=value if value in options else (options[0] if options else None),
        label="Color by",
    )


def colormap_dropdown(kind: str, value=None):
    """Palettes appropriate to the selected column's kind.

    Built in its own cell rather than folded into a widget group, because the
    *options* depend on another widget's value: a qualitative palette is
    meaningless on a mean aggregation, and a sequential ramp is meaningless on a
    per-category count. Switching kind therefore resets the choice, which is
    intended -- the previous palette would not have applied.
    """
    from helpers import scatter as _scatter

    options = _scatter.cmap_options(kind)
    labels = {v: k for k, v in options.items()}
    return mo.ui.dropdown(
        options=options,
        value=labels.get(value, next(iter(options))),
        label="Colormap",
    )


def background_dropdown(value: str = "Dark gray"):
    """Canvas colour.

    Dict options so the label stays readable while `.value` is the literal
    colour bokeh wants. Dark gray rather than black: the low end of most
    sequential ramps is near-black, and pure black swallows the sparse tail.
    """
    return mo.ui.dropdown(
        options={"Dark gray": "#2b2b2b", "White": "white"},
        value=value,
        label="Background",
    )


def hover_sample_slider(value: int = 2000):
    """How many real glyphs to draw over the raster, for hover and the legend."""
    return mo.ui.slider(
        start=0, stop=10_000, step=500, value=value,
        label="Hover sample", show_value=True,
    )


def thumbnail_checkbox(value: bool = False):
    """Whether hover tooltips carry a patch crop.

    Off by default because it is the one control here that costs real time:
    each thumbnail is a JPEG crop built server-side and embedded in the
    document, about 3 ms and 3.7 KB per point.
    """
    return mo.ui.checkbox(value=value, label="Patch thumbnails on hover")


def thumbnail_limit_slider(value: int = 300):
    """Cap on how many patch crops get built when thumbnails are on.

    Separate from the hover sample because it bounds the *expensive* part.
    When thumbnails are enabled the overlay is truncated to this many points,
    so every glyph still has an image -- there are no half-loaded tooltips.
    Roughly 3 ms and 3.7 KB per point, so 300 is about a second and 1 MB.
    """
    return mo.ui.slider(
        start=50, stop=2000, step=50, value=value,
        label="Max hover images", show_value=True,
    )


def width_buttons(get_width, set_width, step: int = 100,
                  min_width: int = 300, max_width: int = 1600):
    """−/+ buttons stepping the plot's on-screen width, as (narrower, wider).

    Takes a `mo.state` getter/setter pair rather than owning the state, because
    two buttons have to write to the same number and the cell that *reads* it
    must be a different one -- marimo will not re-run the cell that owns a
    state setter.

    Two callbacks per button, and both are needed: `on_click` computes the
    button's own next value, a click counter whose only job is to differ every
    press, and `on_change` fires on that difference and moves the width. The
    state update in `on_click` alone looks right and silently does nothing.
    """
    narrower = mo.ui.button(
        label="−", tooltip="Narrower plot", value=0,
        on_click=lambda clicks: clicks + 1,
        on_change=lambda _: set_width(max(min_width, get_width() - step)),
    )
    wider = mo.ui.button(
        label="+", tooltip="Wider plot", value=0,
        on_click=lambda clicks: clicks + 1,
        on_change=lambda _: set_width(min(max_width, get_width() + step)),
    )
    return narrower, wider


def scatter_panel(*widgets):
    """Lay the scatter controls out in one row, wrapping as needed."""
    return mo.hstack(
        [w for w in widgets if w is not None],
        justify="start",
        align="center",
        gap=1,
        wrap=True,
    )


def color_picker(value: str = "#00ff88"):
    """A real <input type="color">.

    marimo has no native colour element, so this comes from wigglystuff, which
    is already a project dependency. Its `.color` attribute holds the hex, and
    `.value` is a dict of traits.
    """
    from wigglystuff import ColorPicker

    return mo.ui.anywidget(ColorPicker(color=value))
