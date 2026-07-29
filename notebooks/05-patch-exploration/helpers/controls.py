"""marimo widget factories for the patch-exploration notebook.

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


def color_picker(value: str = "#00ff88"):
    """A real <input type="color">.

    marimo has no native colour element, so this comes from wigglystuff, which
    is already a project dependency. Its `.color` attribute holds the hex, and
    `.value` is a dict of traits.
    """
    from wigglystuff import ColorPicker

    return mo.ui.anywidget(ColorPicker(color=value))
