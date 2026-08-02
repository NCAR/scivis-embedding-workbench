"""Box-select a region of the projection and page through its patches.

Everything here is Panel, deliberately. A holoviews stream fires in Python but
does *not* re-run a marimo cell, so a gallery living in its own marimo cell would
never update. Keeping the plot, the paging controls and the gallery inside one
`pn.Column` and updating panes from the stream subscriber stays inside panel's
own update path -- which is the direction that works. Mixing marimo widgets with
a panel-managed gallery would hit the same wall.

The notebook supplies a `render` callable, so this module knows nothing about
experiments, crops or LanceDB: it owns the interaction, not the data.
"""

PER_PAGE_OPTIONS = [12, 24, 48]

# Panel's styling assumes a light theme, so text that inherits its colour comes
# out near-black and vanishes on a dark canvas. Every string this module renders
# carries its own colour, switched by prefers-color-scheme so it stays right in
# both without pinning a theme.
_TEXT_CSS = (
    "<style>"
    ".pc-region{color:#1a1a1a;font-size:12px}"
    "@media (prefers-color-scheme: dark){.pc-region{color:#e8e8e8}}"
    "</style>"
)


def explorer(
    plot_pane,
    render,
    per_page: int = 12,
    empty_message: str = "Drag a box on the plot to see the patches in that region.",
):
    """Compose the plot with a paged gallery of whatever region is selected.

    `plot_pane` is the `pn.pane.HoloViews` from `scatter.scatter_pane`; its
    `.object` is the element the bounds stream attaches to.

    `render(bounds, page, per_page)` returns `(html, total)` -- the gallery
    markup for that page and how many patches the region holds in total. It is
    called once per page turn, never for the whole region, so cost scales with
    pages viewed rather than with the size of the selection.
    """
    import holoviews as hv
    import panel as pn

    state = {"bounds": None, "page": 0, "total": 0}

    header = pn.pane.HTML(_header_html(None, 0, 0, per_page), sizing_mode="stretch_width")
    gallery = pn.pane.HTML(_muted(empty_message), sizing_mode="stretch_width")
    # The gallery scrolls inside its own box; without a height cap a page of 48
    # pushes the plot off the top of the screen.
    scroller = pn.Column(
        gallery, sizing_mode="stretch_width", height=520, scroll=True
    )

    prev_btn = pn.widgets.Button(name="◀ Prev", width=90, disabled=True)
    next_btn = pn.widgets.Button(name="Next ▶", width=90, disabled=True)
    # No `name=`: the widget's own label is Panel-themed and unreadable on a
    # dark canvas. The label is rendered alongside it instead, with our CSS.
    per_page_sel = pn.widgets.Select(
        options=PER_PAGE_OPTIONS, value=per_page, width=80
    )
    per_page_label = pn.pane.HTML(
        f"{_TEXT_CSS}<div class='pc-region'>Per page</div>", width=60
    )

    def _n_pages():
        size = int(per_page_sel.value)
        return max(1, -(-state["total"] // size)) if state["total"] else 0

    def _refresh():
        size = int(per_page_sel.value)
        if state["bounds"] is None:
            gallery.object = _muted(empty_message)
            header.object = _header_html(None, 0, 0, size)
            prev_btn.disabled = next_btn.disabled = True
            return

        html, total = render(state["bounds"], state["page"], size)
        state["total"] = total
        gallery.object = html or _muted("No patches in this region.")
        header.object = _header_html(
            state["bounds"], state["page"], total, size
        )
        prev_btn.disabled = state["page"] <= 0
        next_btn.disabled = state["page"] >= _n_pages() - 1

    def _on_bounds(bounds):
        # A new selection always restarts paging: page 4 of the previous region
        # means nothing here.
        state["bounds"] = bounds
        state["page"] = 0
        _refresh()

    def _step(delta):
        def handler(_event):
            state["page"] = max(0, min(_n_pages() - 1, state["page"] + delta))
            _refresh()

        return handler

    def _on_per_page(_event):
        state["page"] = 0
        _refresh()

    prev_btn.on_click(_step(-1))
    next_btn.on_click(_step(+1))
    per_page_sel.param.watch(_on_per_page, "value")

    # Built by `scatter_pane` against the element itself; see the note there on
    # why attaching it to `pane.object` here would silently do nothing.
    stream = getattr(plot_pane, "bounds_stream", None)
    if stream is None:
        stream = hv.streams.BoundsXY(source=plot_pane.object)
    stream.add_subscriber(_on_bounds)

    controls = pn.Row(
        prev_btn,
        next_btn,
        per_page_label,
        per_page_sel,
        sizing_mode="stretch_width",
    )
    column = pn.Column(
        plot_pane, header, controls, scroller, sizing_mode="stretch_width"
    )
    # Held on the object so the caller can keep the stream alive; a garbage
    # collected stream stops delivering.
    column._bounds_stream = stream
    return column


def _muted(text: str) -> str:
    """A dim note that stays legible on either theme."""
    return f"{_TEXT_CSS}<div class='pc-region' style='opacity:0.7'><em>{text}</em></div>"


def _header_html(bounds, page, total, per_page) -> str:
    """One line describing the current selection and position within it."""
    if bounds is None:
        return _muted("No region selected — the box-select tool is active on the plot.")
    x0, y0, x1, y1 = bounds
    n_pages = max(1, -(-total // per_page)) if total else 0
    where = f"page {page + 1} / {n_pages}" if total else "no patches"
    return (
        f"{_TEXT_CSS}<div class='pc-region'>"
        f"<b>Region</b> x {min(x0, x1):.2f}…{max(x0, x1):.2f} &nbsp; "
        f"y {min(y0, y1):.2f}…{max(y0, y1):.2f}"
        f" &nbsp;·&nbsp; <b>{total:,}</b> patches"
        f" &nbsp;·&nbsp; {where}"
        "</div>"
    )
