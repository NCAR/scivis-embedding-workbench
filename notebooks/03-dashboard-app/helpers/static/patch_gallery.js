function render({ model, el }) {
  el.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "pgal-grid";
  el.appendChild(grid);

  // Tiles are rebuilt only when the images themselves change. Selection,
  // column count and theme are applied without touching the <img> elements,
  // so the browser never re-decodes a thumbnail for those.
  let tiles = [];

  const applySelection = () => {
    const sel = model.get("selected");
    tiles.forEach((tile, i) => tile.classList.toggle("pgal-selected", i === sel));
  };

  const applyLayout = () => {
    const nCols = Math.max(1, model.get("n_cols") || 1);
    grid.style.gridTemplateColumns = `repeat(${nCols}, minmax(0, 1fr))`;
    grid.style.maxHeight = `${model.get("max_h") || 650}px`;
    grid.dataset.theme = model.get("theme") || "light";
    // Reserve each tile's height from its aspect ratio so the grid doesn't
    // reflow as images decode.
    const w = model.get("thumb_w") || 1;
    const h = model.get("thumb_h") || 1;
    tiles.forEach((tile) => {
      tile.querySelector(".pgal-img").style.aspectRatio = `${w} / ${h}`;
    });
  };

  const buildTiles = () => {
    const thumbs = model.get("thumbs") || [];
    const captions = model.get("captions") || [];
    grid.innerHTML = "";
    tiles = thumbs.map((src, i) => {
      const tile = document.createElement("div");
      tile.className = "pgal-tile";
      tile.setAttribute("role", "button");
      tile.tabIndex = 0;

      const img = document.createElement("img");
      img.className = "pgal-img";
      img.src = src;
      img.loading = "lazy";
      img.decoding = "async";
      img.draggable = false;
      img.alt = captions[i] || `result ${i}`;

      const cap = document.createElement("div");
      cap.className = "pgal-caption";
      cap.textContent = captions[i] || "";
      cap.title = captions[i] || "";

      const select = () => {
        // Paint the border straight away, then tell Python. The round-trip
        // that loads the larger view no longer gates the visual feedback.
        model.set("selected", i);
        model.save_changes();
        applySelection();
      };
      tile.addEventListener("click", select);
      tile.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select();
        }
      });

      tile.appendChild(img);
      tile.appendChild(cap);
      grid.appendChild(tile);
      return tile;
    });
  };

  const rebuild = () => {
    buildTiles();
    applyLayout();
    applySelection();
  };

  model.on("change:thumbs", rebuild);
  model.on("change:captions", rebuild);
  model.on("change:n_cols", applyLayout);
  model.on("change:max_h", applyLayout);
  model.on("change:theme", applyLayout);
  model.on("change:thumb_w", applyLayout);
  model.on("change:thumb_h", applyLayout);
  model.on("change:selected", applySelection);

  rebuild();
}

export default { render };
