// Floating preview window for H3 Ripple Chunked.
// The node announces every finished chunk ("ripplechunks.preview"); this shows it as a small looping animation.
// Independent of ComfyUI's image feeds, so it appears for every run.

const STORE = "RippleChunks.preview.v1";

function loadSettings() {
  try {
    return { size: 360, position: "right", custom: false, ...(JSON.parse(localStorage.getItem(STORE)) || {}) };
  } catch (e) {
    return { size: 360, position: "right", custom: false };
  }
}

const settings = loadSettings();
let box = null;
let image = null;
let title = null;
let api = null;

function saveSettings() {
  try {
    localStorage.setItem(STORE, JSON.stringify({ size: settings.size, position: settings.position,
                                                 custom: settings.custom, x: settings.x, y: settings.y }));
  } catch (e) {
    /* storage may be unavailable; the window still works */
  }
}

const PAD = 8; // space around the picture inside the window

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), Math.max(low, high));
}

function applyLayout() {
  // The window is exactly as wide as the picture. The picture is never taller than the screen allows.
  const ratio = image.naturalWidth ? image.naturalHeight / image.naturalWidth : 16 / 9;
  const tallest = Math.max(120, (window.innerHeight * 0.88 - 80) / ratio);
  const width = Math.round(Math.max(100, Math.min(settings.size, tallest, window.innerWidth * 0.94 - 2 * PAD)));
  image.style.width = width + "px";
  box.style.width = width + 2 * PAD + "px";

  box.style.right = "auto";
  box.style.transform = "none";
  if (settings.custom && Number.isFinite(settings.x) && Number.isFinite(settings.y)) {
    // Dragged: keep it fully on screen, whatever the size or the browser window does.
    const boxWidth = box.offsetWidth || width + 2 * PAD;
    const boxHeight = box.offsetHeight || 0;
    settings.x = clamp(settings.x, 0, window.innerWidth - boxWidth);
    settings.y = clamp(settings.y, 0, window.innerHeight - boxHeight);
    box.style.left = settings.x + "px";
    box.style.top = settings.y + "px";
    box.style.bottom = "auto";
    return;
  }
  box.style.top = "auto";
  box.style.bottom = "12px";
  box.style.left = "auto";
  if (settings.position === "left") {
    box.style.left = "12px";
  } else if (settings.position === "center") {
    box.style.left = "50%";
    box.style.transform = "translateX(-50%)";
  } else {
    box.style.right = "12px";
  }
}

// Drag the window by its title bar. Double-click the title bar to put it back in its docked position.
function enableDragging(handle, where) {
  let drag = null;
  handle.style.cursor = "move";
  handle.style.touchAction = "none";
  handle.style.userSelect = "none";
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== undefined && event.button !== 0) return;
    if (event.target && event.target.tagName === "BUTTON") return;
    const rect = box.getBoundingClientRect();
    drag = { dx: event.clientX - rect.left, dy: event.clientY - rect.top };
    settings.custom = true;
    settings.x = rect.left;
    settings.y = rect.top;
    where.value = "custom";
    if (handle.setPointerCapture) {
      try {
        handle.setPointerCapture(event.pointerId);
      } catch (e) {
        /* dragging still works while the pointer stays over the bar */
      }
    }
    applyLayout();
    if (event.preventDefault) event.preventDefault();
  });
  handle.addEventListener("pointermove", (event) => {
    if (!drag) return;
    settings.x = event.clientX - drag.dx;
    settings.y = event.clientY - drag.dy;
    applyLayout();
  });
  const stop = () => {
    if (drag) {
      drag = null;
      saveSettings();
    }
  };
  handle.addEventListener("pointerup", stop);
  handle.addEventListener("pointercancel", stop);
  handle.addEventListener("dblclick", () => {
    settings.custom = false;
    where.value = settings.position;
    applyLayout();
    saveSettings();
  });
}

function labelled(text, control) {
  const wrap = document.createElement("label");
  wrap.style.display = "flex";
  wrap.style.alignItems = "center";
  wrap.style.gap = "4px";
  const span = document.createElement("span");
  span.textContent = text;
  wrap.append(span, control);
  return wrap;
}

function build() {
  box = document.createElement("div");
  box.id = "ripplechunks-preview";
  Object.assign(box.style, {
    position: "fixed",
    bottom: "12px",
    zIndex: "2000",
    background: "#1e1e1e",
    color: "#ddd",
    border: "1px solid #555",
    borderRadius: "8px",
    boxShadow: "0 4px 18px rgba(0,0,0,0.5)",
    font: "12px sans-serif",
    display: "none",
    boxSizing: "border-box",
    maxWidth: "96vw",
    overflow: "hidden",
  });

  const top = document.createElement("div");
  Object.assign(top.style, { display: "flex", alignItems: "center", gap: "6px", padding: PAD + "px " + PAD + "px 2px" });

  title = document.createElement("span");
  Object.assign(title.style, { fontWeight: "bold", flex: "1 1 auto", minWidth: "0", overflow: "hidden",
                               textOverflow: "ellipsis", whiteSpace: "nowrap" });
  title.textContent = "Ripple preview";

  const close = document.createElement("button");
  close.textContent = "✕";
  close.title = "Close (it reopens when the next chunk finishes)";
  close.addEventListener("click", () => {
    box.style.display = "none";
  });
  top.append(title, close);

  const controls = document.createElement("div");
  Object.assign(controls.style, { display: "flex", alignItems: "center", gap: "8px", padding: "0 " + PAD + "px 4px" });

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "120";
  slider.max = "1000";
  slider.step = "10";
  slider.value = String(settings.size);
  Object.assign(slider.style, { flex: "1 1 auto", minWidth: "30px", width: "100%" });
  slider.addEventListener("input", () => {
    settings.size = Number(slider.value);
    applyLayout();
    saveSettings();
  });

  const where = document.createElement("select");
  for (const option of ["left", "center", "right"]) {
    const el = document.createElement("option");
    el.value = option;
    el.textContent = option;
    where.appendChild(el);
  }
  const moved = document.createElement("option");
  moved.value = "custom";
  moved.textContent = "moved";
  moved.disabled = true;
  where.appendChild(moved);
  where.value = settings.custom ? "custom" : settings.position;
  where.addEventListener("change", () => {
    if (where.value === "custom") return;
    settings.position = where.value;
    settings.custom = false;
    applyLayout();
    saveSettings();
  });

  controls.append(slider, where);
  enableDragging(top, where);
  const bar = document.createElement("div");
  bar.append(top, controls);

  image = document.createElement("img");
  image.alt = "Chunk preview";
  Object.assign(image.style, { display: "block", height: "auto", margin: "0 " + PAD + "px " + PAD + "px" });
  image.addEventListener("load", applyLayout); // now the picture's real shape is known
  box.append(bar, image);
  document.body.appendChild(box);
  applyLayout();
}

function show(detail) {
  if (!box) build();
  const query = new URLSearchParams({
    filename: detail.filename,
    subfolder: detail.subfolder || "",
    type: detail.type || "temp",
    rand: String(Math.random()), // never reuse a cached copy
  });
  image.src = api.apiURL("/view?" + query.toString());
  title.textContent = detail.label || "Ripple preview";
  box.style.display = "block";
  applyLayout();
}

async function getApi() {
  if (window.comfyAPI && window.comfyAPI.api && window.comfyAPI.api.api) {
    return window.comfyAPI.api.api;
  }
  return (await import("../../scripts/api.js")).api;
}

getApi()
  .then((found) => {
    api = found;
    api.addEventListener("ripplechunks.preview", (event) => show(event.detail));
  })
  .catch((error) => console.warn("[RippleChunks] preview window could not start:", error));

window.addEventListener && window.addEventListener("resize", () => box && applyLayout());
