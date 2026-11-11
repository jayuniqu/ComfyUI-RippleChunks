"""H3 Ripple Chunked: apply a Ripple edit to a video longer than one window, one chunk at a time.

Reuses the WepeNerd Ripple nodes (Ripple Clip defines the first window and the edited anchors;
Ripple Apply builds each chunk's conditioning). Between chunks the result is moved to system RAM and
GPU memory is released, so peak VRAM is that of one chunk, not the whole video.
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils

try:
    from . import chunk_core as cc
except ImportError:  # loaded standalone
    import chunk_core as cc


# Same text as WepeNerd's default. Copied here because ComfyUI reads INPUT_TYPES at startup, possibly
# before the WepeNerd pack has loaded.
DEFAULT_PROMPT = (
    "Edit propagation: each edited frame shows exactly how that moment of the video should look after the edit. "
    "Apply the same changes to every frame of the clip, before, between and after the edited frames, so the edit "
    "stays consistent through the whole video. Keep the guide video's motion, timing, camera movement, composition "
    "and all unedited content unchanged."
)


def _wepe():
    """The loaded WepeNerd ripple module (any package name it was loaded under)."""
    for name, module in list(sys.modules.items()):
        if name.split(".")[-1] == "ripple_node" and hasattr(module, "WN_RippleApply"):
            return module
    raise RuntimeError("ComfyUI-WepeNerd is not loaded. Install it first: https://github.com/WepeNerd/ComfyUI-WepeNerd")


def _sample(model, positive, latent, noise_seed, sampler, sigmas, quiet=False):
    from comfy.cli_args import LatentPreviewMethod, args
    from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced

    guider = Guider_Basic(model)
    guider.set_conds(positive)
    noise = Noise_RandomNoise(noise_seed)

    def run():
        if hasattr(SamplerCustomAdvanced, "execute"):
            return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).args[0]
        return SamplerCustomAdvanced().sample(noise, guider, sampler, sigmas, latent)[0]  # older ComfyUI

    saved = args.preview_method
    if quiet:
        # The sampler's own step-by-step preview would draw over the finished-chunk preview on this node.
        args.preview_method = LatentPreviewMethod.NoPreviews
    try:
        try:
            out = run()
        except AttributeError as error:
            # ComfyUI's own sampler preview crashes this way when "TAESD" previews are selected but the H3 preview
            # file (taeh3*, in models/vae_approx) cannot be loaded. It is not needed here, so use the basic preview.
            if "show_progress_bar" not in str(error):
                raise
            print("[RippleChunked] The sampler preview is set to TAESD but the H3 TAESD file could not be loaded. "
                  "Using the basic preview for this run. To fix it, change the live preview method in ComfyUI's "
                  "settings, or remove the broken taeh3 file from models/vae_approx.")
            args.preview_method = LatentPreviewMethod.Latent2RGB
            out = run()
    finally:
        args.preview_method = saved
    return out["samples"]


def _decode(vae, samples) -> np.ndarray:
    """Latent -> uint8 frames [N, H, W, 3] in system RAM. ComfyUI's VAE falls back to tiled decode on OOM."""
    if samples.is_nested:
        samples = samples.unbind()[0]
    images = vae.decode(samples)
    if images.dim() == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return (images[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def _free():
    gc.collect()
    mm.soft_empty_cache()


class WN_RippleChunked:
    CATEGORY = "WepeNerd/H3 Ripple"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "frame_rate")
    DESCRIPTION = (
        "Applies a Ripple edit to a video longer than one window. The window on Ripple Clip is the chunk size "
        "and its markers/edits define the first chunk. Each later chunk is anchored on the end of the previous "
        "result, so the edit carries forward. Replaces Ripple Apply, the guider, the sampler and the VAE decode."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "ripple": ("WN_RIPPLE", {"tooltip": "From Ripple Clip. Its window length is the chunk size."}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "seed": ("INT", {"default": 123456, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "total_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01,
                                            "tooltip": "How much video to process, from the Ripple Clip window start. 0 = to the end."}),
                "overlap_frames": ("INT", {"default": 22, "min": 6, "max": 90, "step": 1,
                                           "tooltip": "Frames shared between neighbouring chunks. Room for the carried-over frames; the new chunk takes over after the first few."}),
                "carry_anchors": ("INT", {"default": 4, "min": 1, "max": 6,
                                          "tooltip": "How many frames of the previous result are locked into the next chunk, spread across the overlap. More = the chunks stay matched, fewer = more freedom."}),
                "prompt": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT,
                                      "tooltip": "Leave as is, same as Ripple Apply."}),
                "blend_frames": ("INT", {"default": 2, "min": 1, "max": 22, "step": 1,
                                         "tooltip": "Frames faded at the join. 1 = clean cut (no ghosting). Higher = softer but can ghost."}),
                "live_preview": ("BOOLEAN", {"default": True,
                                             "tooltip": "After each chunk, show everything merged so far on this node. Cancel the run if it looks wrong. See sampler_preview for the sampler's own step-by-step preview."}),
                "mask_grow": ("INT", {"default": 6, "min": -64, "max": 128, "step": 1,
                                      "tooltip": "Only used with detail_mask. Pixels to widen the mask (negative = shrink) before softening."}),
                "mask_feather": ("INT", {"default": 10, "min": 0, "max": 128, "step": 1,
                                         "tooltip": "Only used with detail_mask. How soft the edge of the pasted original is, in pixels."}),
                "seam_search": ("BOOLEAN", {"default": True,
                                            "tooltip": "Join chunks at the overlap frame where they look most alike, instead of always at the start of the overlap."}),
                "mask_smooth": ("INT", {"default": 3, "min": 0, "max": 15, "step": 1,
                                        "tooltip": "Only used with detail_mask. Averages the mask over this many frames either side, to calm a mask that jitters from frame to frame. 0 = off."}),
                "debug_mask_overlay": ("BOOLEAN", {"default": False,
                                                   "tooltip": "Only used with detail_mask. Tints the pasted area red in the output so you can see exactly what is being pasted. Turn off for the real render."}),
                "sampler_preview": (["every chunk", "first chunk only", "never"], {
                    "default": "every chunk",
                    "tooltip": "The sampler's own step-by-step preview (use it to check the reference image is picked up). "
                               "'every chunk' leaves it alone. 'first chunk only' / 'never' hide it for later chunks (or all), "
                               "which keeps the finished-chunk preview from being drawn over on this node."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "edited_images": ("IMAGE", {"tooltip": "First chunk only: edited frames, one per marker in time order."}),
                "edited_anchor_1": ("IMAGE", {"tooltip": "First chunk only: edited image for marker 1."}),
                "edited_anchor_2": ("IMAGE", {"tooltip": "First chunk only: edited image for marker 2."}),
                "detail_mask": ("MASK", {"tooltip": "Optional. Masks for the WHOLE original video (e.g. SAM3 face and hands). "
                                                    "White = put the original pixels back on top of the finished video."}),
            },
        }

    def run(self, model, clip, vae, ripple, sampler, sigmas, seed, total_seconds, overlap_frames,
            carry_anchors, prompt, blend_frames=2, live_preview=True, mask_grow=6, mask_feather=10, seam_search=True,
            mask_smooth=3, debug_mask_overlay=False, sampler_preview="every chunk",
            unique_id=None,
            edited_images=None, edited_anchor_1=None, edited_anchor_2=None, detail_mask=None):
        rn = _wepe()
        applier = rn.WN_RippleApply()

        path = rn.video_path(ripple["source"])
        info = rn.probe(path)
        geometry = ripple["geometry"]
        s0, length = int(ripple["window_start"]), int(ripple["length"])
        available = info["grid_frames"] - s0
        wanted = available if total_seconds <= 0 else min(available, round(total_seconds * cc.FPS))
        plan = cc.plan_chunks(s0, s0 + wanted, length, min(overlap_frames, length - 17))
        notes = [f"{len(plan)} chunk(s) of {length} frames covering frames {s0}–{s0 + wanted} "
                 f"({wanted / cc.FPS:.1f} s)"]
        print(f"[RippleChunked] {notes[0]}")

        stitcher = cc.Stitcher(s0, plan[-1][0] + plan[-1][1], geometry["height"], geometry["width"])
        first_window_time, audio_src = ripple.get("window_time"), None

        for index, (start, n) in enumerate(plan):
            mm.throw_exception_if_processing_interrupted()
            if index == 0:
                current = ripple
                if n != length:  # range shorter than the clip window
                    loaded = rn.load_window(path, info, start, n, geometry)
                    current = self._window_ripple(ripple, loaded, start, ripple["markers"], path)
                anchors = {}
                if edited_anchor_1 is not None:
                    anchors["edited_anchor_1"] = edited_anchor_1
                if edited_anchor_2 is not None:
                    anchors["edited_anchor_2"] = edited_anchor_2
                edits = edited_images
            else:
                loaded = rn.load_window(path, info, start, n, geometry)
                overlap = stitcher.written - (start - s0)
                positions = cc.anchor_positions(overlap, carry_anchors)
                markers = [{"id": i + 1, "frame": f, "edited": "", "edited_at": None}
                           for i, f in enumerate(positions)]
                frames = np.stack([stitcher.get(start + f) for f in positions])
                edits = rn.trim_content(torch.from_numpy(frames).float().div_(255), geometry)
                current = self._window_ripple(ripple, loaded, start, markers, path)
                anchors = {}

            began = time.time()
            print(f"[RippleChunked] chunk {index + 1}/{len(plan)}: frames {start}–{start + n}")
            positive, latent = applier.apply(clip, vae, current, prompt, edited_images=edits, **anchors)["result"]
            quiet = live_preview and (sampler_preview == "never" or (sampler_preview == "first chunk only" and index > 0))
            samples = _sample(model, positive, latent, seed, sampler, sigmas, quiet=quiet)
            del positive, latent
            frames_out = _decode(vae, samples)
            del samples
            stitcher.add(frames_out, start, blend_frames, seam_search)
            del frames_out, current
            loaded = None
            _free()
            print(f"[RippleChunked] chunk {index + 1}/{len(plan)} done in {time.time() - began:.0f}s")
            if index:
                seam, gap = stitcher.last_seam
                print(f"[RippleChunked] joined {seam} frames into the overlap; chunks differed by {gap:.1f}/255 there")
            if live_preview and index + 1 < len(plan):
                self._preview(stitcher, geometry, unique_id, f"{unique_id}_{index + 1}",
                              label=f"chunk {index + 1} of {len(plan)} done", since=start - s0 - 12)

        if detail_mask is not None:
            notes.append("Original pixels pasted back through detail_mask.")
            notes.extend(self._paste_original(rn, path, info, stitcher, s0, geometry, detail_mask, mask_grow,
                                              mask_feather, mask_smooth, debug_mask_overlay))

        result = stitcher.result()
        x, y, w, h = geometry["content_rect"]
        images = torch.from_numpy(np.ascontiguousarray(result[:, y:y + h, x:x + w])).float().div_(255)
        audio = None
        if info.get("audio"):
            audio = rn.decode_audio(path, first_window_time, len(result) / cc.FPS)
        ui = {"text": notes}
        if live_preview:
            shown = self._preview(stitcher, geometry, unique_id, f"{unique_id}_final", send=False,
                                  label=f"finished: all {len(plan)} chunk(s) merged")
            if shown:
                ui.update({"images": [shown], "animated": (True,)})
        return {"ui": ui, "result": (images, audio, float(cc.FPS))}

    @staticmethod
    def _paste_original(rn, path, info, stitcher, s0, geometry, mask, grow, feather, smooth=0, debug=False, block=48):
        """Lay the original frames over the finished video wherever ``mask`` is white, a block at a time.

        Returns warnings about the mask (also printed) so a wrongly prepared mask is easy to spot.
        """
        mask = mask.detach().float().cpu()
        if mask.dim() == 4:  # [N, H, W, 1]
            mask = mask[..., 0]
        if mask.dim() == 2:
            mask = mask[None]
        x, y, w, h = geometry["content_rect"]
        grid_total = info["grid_frames"]
        count, mh, mw = mask.shape
        cx, cy, cw, ch = geometry["crop"]
        warnings = []
        print(f"[RippleChunked] mask: {count} frames, {mw}x{mh}; original video: {grid_total} frames at 24 fps")
        if abs((cw * mw) / (ch * mh) / (w / h) - 1) > 0.015:
            warnings.append(f"WARNING: the mask shape ({mw}x{mh}) does not match the video crop. Was it made from the "
                            "Ripple Clip output instead of the original video? It should come from the full original video.")
        if count < 0.5 * grid_total:
            warnings.append(f"WARNING: the mask has only {count} frames but the video has {grid_total}. It will be stretched "
                            "out of step. Make the mask from the whole original video.")
        for line in warnings:
            print(f"[RippleChunked] {line}")
        total = stitcher.written
        for b in range(0, total, block):
            mm.throw_exception_if_processing_interrupted()
            n = min(block, total - b)
            src, _ = rn.decode_grid_window(path, s0 + b, n, geometry)
            n = min(n, len(src))
            if n <= 0:
                break
            lo, hi = max(0, b - smooth), min(total, b + n + smooth)  # extra frames so averaging works at block edges
            idx = [cc.mask_frame_index(s0 + f, count, grid_total) for f in range(lo, hi)]
            fitted = cc.temporal_mean(cc.fit_mask(mask[idx].numpy(), geometry["crop"], (w, h)), smooth)
            soft = cc.soften(fitted[b - lo:b - lo + n], grow, feather)
            region = stitcher.buffer[b:b + n, y:y + h, x:x + w]
            out = cc.paste(region, src[:n, y:y + h, x:x + w], soft)
            stitcher.buffer[b:b + n, y:y + h, x:x + w] = cc.tint(out, soft) if debug else out
        print("[RippleChunked] original pixels pasted back through the mask" + (" (red = pasted area)" if debug else ""))
        return warnings

    @staticmethod
    def _preview(stitcher, geometry, node_id, tag, send=True, label="", since=0):
        """Write a small animation of the merged video (from frame ``since``) and show it in the preview window.

        Never raises: a preview problem must not stop a long render.
        """
        try:
            import folder_paths

            x, y, w, h = geometry["content_rect"]
            name = f"ripplechunk_preview_{tag}_{time.time_ns()}.webp"
            full = os.path.join(folder_paths.get_temp_directory(), name)
            cc.write_preview(stitcher.result()[max(0, since):, y:y + h, x:x + w], full)
            shown = {"filename": name, "subfolder": "", "type": "temp"}
            from server import PromptServer

            server = PromptServer.instance
            # Our own floating window (web/ripplechunks_preview.js) listens for this.
            server.send_sync("ripplechunks.preview", dict(shown, node=str(node_id), label=label), server.client_id)
            if send:  # also the standard node/feed message
                server.send_sync("executed", {"node": node_id, "display_node": node_id,
                                              "output": {"images": [shown], "animated": (True,)},
                                              "prompt_id": server.last_prompt_id}, server.client_id)
            print(f"[RippleChunked] preview sent ({label or tag}): {name}")
            return shown
        except Exception as error:  # noqa: BLE001
            print(f"[RippleChunked] preview skipped: {error}")
            return None

    @staticmethod
    def _window_ripple(base, loaded, start, markers, path):
        """A copy of the Ripple Clip result re-pointed at another window of the same video."""
        current = dict(base)
        n = loaded["length"]
        edits = tuple(m["frame"] for m in markers)
        current.update(
            window=loaded["window"], length=n, window_start=start, window_time=loaded["window_time"],
            audio=None, window_key=loaded["key"], markers=markers,
            binding=(base["source"], path.stat().st_mtime_ns, start, n, edits, base["width"], base["height"]),
        )
        return current


NODE_CLASS_MAPPINGS = {"WN_RippleChunked": WN_RippleChunked}
NODE_DISPLAY_NAME_MAPPINGS = {"WN_RippleChunked": "H3 Ripple Chunked (long video)"}
