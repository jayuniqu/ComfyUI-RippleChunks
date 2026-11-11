"""Chunk planning and stitching for long H3 Ripple runs. Pure numpy, no ComfyUI imports.

Positions are absolute frames on the source video's 24 fps grid. H3 window lengths follow 17k + 5.
"""

from __future__ import annotations

import numpy as np

FPS = 24


def valid_length(frames: int) -> int:
    """Largest 17k+5 length that fits in ``frames`` (0 if fewer than 5)."""
    return 17 * ((frames - 5) // 17) + 5 if frames >= 5 else 0


def plan_chunks(start: int, end: int, length: int, overlap: int) -> list[tuple[int, int]]:
    """Cover frames [start, end) with windows of ``length`` frames that share ``overlap`` frames.

    Windows are spaced evenly and the last one ends exactly at ``end`` (never padded or looped).
    A range shorter than one window becomes a single window of the longest valid length.
    """
    if (length - 5) % 17 or length < 5:
        raise ValueError(f"Chunk length {length} is not a valid H3 length (17k + 5).")
    if not 1 <= overlap <= length - 17:
        raise ValueError(f"Overlap must be between 1 and {length - 17} frames for a {length}-frame chunk.")
    total = end - start
    if total < 5:
        raise ValueError(f"Only {total} frames to cover; H3 needs at least 5.")
    if total <= length:
        return [(start, valid_length(total))]
    # Fewest chunks that cover the range with at least `overlap` shared frames, spaced evenly so
    # the last chunk is not a nearly-empty repeat of the one before it.
    count = 1 + -(-(total - length) // (length - overlap))
    last_start = end - length
    starts = [start + round(i * (last_start - start) / (count - 1)) for i in range(count)]
    return [(s, length) for s in starts]


def anchor_positions(overlap: int, count: int) -> list[int]:
    """Chunk-local frames (inside the overlap) that carry the previous chunk's result forward."""
    count = max(1, min(count, overlap))
    if count == 1:
        return [0]
    return sorted({int(round(v)) for v in np.linspace(0, overlap - 1, count)})


class Stitcher:
    """Collects chunk results into one uint8 buffer, joining chunks with a short fade."""

    def __init__(self, start: int, end: int, height: int, width: int):
        self.start = start
        self.buffer = np.zeros((end - start, height, width, 3), dtype=np.uint8)
        self.written = 0  # frames filled so far, relative to ``start``
        self.last_seam = (0, 0.0)  # (overlap frame used for the join, how different the chunks were there)

    def get(self, absolute_frame: int) -> np.ndarray:
        rel = absolute_frame - self.start
        if not 0 <= rel < self.written:
            raise IndexError(f"Frame {absolute_frame} has not been generated yet.")
        return self.buffer[rel]

    def add(self, frames: np.ndarray, absolute_start: int, blend: int = 2, seam_search: bool = True) -> int:
        """Place a chunk; returns the number of cross-faded frames.

        Inside the overlap the old and new chunk both show the same moment. With ``seam_search`` the
        join is made at the overlap frame where the two look most alike (compared on a shrunken copy),
        so there is almost nothing to fade. Before the seam the old chunk is kept, after it the new
        one; only ``blend`` frames starting at the seam are faded. Fading the whole overlap would show
        both versions at once (ghosting) wherever they differ.
        """
        rel = absolute_start - self.start
        if rel < 0 or rel > self.written:
            raise ValueError("Chunks must be added in order without gaps.")
        count = min(len(frames), len(self.buffer) - rel)
        frames = frames[:count]
        ov = min(max(0, self.written - rel), count)
        seam, diff = 0, 0.0
        if ov and seam_search and ov > 1:
            a = self.buffer[rel:rel + ov, ::6, ::6].astype(np.float32)
            b = frames[:ov, ::6, ::6].astype(np.float32)
            gap = np.abs(a - b).mean(axis=(1, 2, 3))
            seam = int(np.argmin(gap))
            diff = float(gap[seam])
        self.last_seam = (seam, diff)
        n = min(max(0, blend), max(0, ov - seam))
        if n:
            w = ((np.arange(n, dtype=np.float32) + 1) / (n + 1)).reshape(-1, 1, 1, 1)
            old = self.buffer[rel + seam:rel + seam + n].astype(np.float32)
            new = frames[seam:seam + n].astype(np.float32)
            self.buffer[rel + seam:rel + seam + n] = np.clip(old * (1 - w) + new * w + 0.5, 0, 255).astype(np.uint8)
        self.buffer[rel + seam + n:rel + count] = frames[seam + n:]
        self.written = max(self.written, rel + count)
        return n

    def result(self) -> np.ndarray:
        return self.buffer[:self.written]


def write_preview(frames: np.ndarray, path: str, fps: float = FPS, max_side: int = 480, step: int = 2) -> None:
    """Small animated WebP of uint8 frames [N, H, W, 3]: every ``step``-th frame, longest side ``max_side``."""
    from PIL import Image

    picked = frames[::max(1, step)]
    h, w = picked.shape[1:3]
    scale = min(1.0, max_side / max(h, w))
    size = (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
    images = [Image.fromarray(f).resize(size, Image.BILINEAR) if scale < 1 else Image.fromarray(f) for f in picked]
    images[0].save(path, save_all=True, append_images=images[1:], duration=max(20, int(1000 * step / fps)),
                   loop=0, quality=70, method=1)


# --------------------------------------------------------------------------------------------
# Putting the original (unedited) pixels back through a mask, e.g. face and hands from SAM3.
# --------------------------------------------------------------------------------------------

def mask_frame_index(frame: int, mask_len: int, grid_total: int) -> int:
    """Mask frame for a 24 fps grid frame, even if the mask was made at another frame rate."""
    if mask_len <= 0 or grid_total <= 0:
        raise ValueError("Empty mask or video.")
    return min(mask_len - 1, max(0, int((frame + 0.5) * mask_len / grid_total)))


def fit_mask(mask: np.ndarray, crop, size: tuple[int, int]) -> np.ndarray:
    """Masks [N, H, W] made on the full source -> the Ripple crop, resized to ``size`` (width, height)."""
    from PIL import Image

    n, height, width = mask.shape
    x, y, w, h = crop
    left = min(width - 1, int(np.floor(x * width + 0.5)))
    top = min(height - 1, int(np.floor(y * height + 0.5)))
    right = min(width, max(left + 1, int(np.floor((x + w) * width + 0.5))))
    bottom = min(height, max(top + 1, int(np.floor((y + h) * height + 0.5))))
    cropped = mask[:, top:bottom, left:right]
    if cropped.shape[1:] == (size[1], size[0]):
        return cropped.astype(np.float32)
    out = np.empty((n, size[1], size[0]), dtype=np.float32)
    for i in range(n):
        out[i] = np.asarray(Image.fromarray(np.ascontiguousarray(cropped[i], dtype=np.float32), mode="F")
                            .resize(size, Image.BILINEAR))
    return out


def _slide(a: np.ndarray, radius: int, axis: int, reduce) -> np.ndarray:
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(a, pad, mode="edge")
    size = a.shape[axis]
    out = None
    for shift in range(2 * radius + 1):
        part = np.take(padded, range(shift, shift + size), axis=axis)
        out = part if out is None else reduce(out, part)
    return out


def _box_blur(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius + 1, radius)
    padded = np.pad(a, pad, mode="edge").astype(np.float64)
    c = np.cumsum(padded, axis=axis)
    size = a.shape[axis]
    hi = np.take(c, range(2 * radius + 1, 2 * radius + 1 + size), axis=axis)
    lo = np.take(c, range(0, size), axis=axis)
    return ((hi - lo) / (2 * radius + 1)).astype(np.float32)


def soften(mask: np.ndarray, grow: int = 0, feather: int = 0) -> np.ndarray:
    """Grow (or shrink, if negative) a mask [N, H, W] by ``grow`` px, then blur its edge over ~``feather`` px."""
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if grow:
        r, reduce = abs(int(grow)), (np.maximum if grow > 0 else np.minimum)
        m = _slide(_slide(m, r, 1, reduce), r, 2, reduce)
    if feather > 0:
        r = max(1, int(np.ceil(feather / 3)))
        for _ in range(3):  # three box blurs approximate a Gaussian
            m = _box_blur(_box_blur(m, r, 1), r, 2)
    return np.clip(m, 0.0, 1.0)


def paste(generated: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """uint8 [N, H, W, 3] generated frames with the original showing through where mask is 1."""
    m = mask[..., None].astype(np.float32)
    out = generated.astype(np.float32) * (1.0 - m) + original.astype(np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def temporal_mean(mask: np.ndarray, radius: int) -> np.ndarray:
    """Average a mask [N, H, W] over +-``radius`` neighbouring frames (calms a mask that jitters between frames)."""
    return mask if radius <= 0 else np.clip(_box_blur(mask.astype(np.float32), int(radius), 0), 0.0, 1.0)


def tint(frames: np.ndarray, mask: np.ndarray, color=(255, 0, 0), strength: float = 0.55) -> np.ndarray:
    """Colour the masked area of uint8 frames [N, H, W, 3]; for checking what gets pasted."""
    m = (np.clip(mask, 0, 1) * strength)[..., None].astype(np.float32)
    out = frames.astype(np.float32) * (1 - m) + np.array(color, np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def contact_sheet(frames: np.ndarray, count: int = 6, height: int = 256) -> np.ndarray:
    """A single uint8 image with ``count`` evenly spaced frames side by side."""
    from PIL import Image

    picks = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    tiles = []
    for i in picks:
        h, w = frames[i].shape[:2]
        tiles.append(np.asarray(Image.fromarray(frames[i]).resize((max(2, round(w * height / h)), height), Image.BILINEAR)))
    return np.concatenate(tiles, axis=1)
