# ComfyUI-RippleChunks

**H3 Ripple Chunked (long video)**: run a Ripple edit over a video longer than one window without the
VRAM growing with video length. Needs ComfyUI-WepeNerd.

## Install
Copy this folder to `ComfyUI/custom_nodes/` and restart.

## Wire it into H3_Ripple_Simple
Keep: all loaders, both LoRAs, Model Attention Backend, Ripple Clip, Scheduler, Sampler Select, Save Video.
Remove: Ripple Apply, Random Noise, Basic Guider, Sampler Custom Advanced, VAE Decode.

| Chunked node input | Connect from |
|---|---|
| model | Model Attention Backend (model) |
| clip / vae | CLIP loader / VAE loader |
| ripple | Ripple Clip (ripple) |
| sampler | Sampler Select |
| sigmas | Basic Scheduler |
| edited_images / edited_anchor_N | same as you used for Ripple Apply (or leave empty if you uploaded edits on the Ripple Clip card) |

Output `images` / `audio` / `frame_rate` go to Save Video (or Ripple Output is not needed).

## Settings
- **Chunk size** = the duration set on Ripple Clip (e.g. 3.75 s). Smaller = less VRAM, more chunks.
- **total_seconds**: how much video to process from the Ripple Clip window start. 0 = to the end.
- **overlap_frames** (default 22) and **carry_anchors** (default 2): how the next chunk picks up from the last.
- **live_preview** (default on): after each chunk, a floating preview window opens at the bottom of the screen and loops the chunk that just finished (with a few frames before it, so you can see the join). It has a Size slider and a Position choice (left / center / right), and a final preview of the whole video when the run ends. The window is part of this pack (web/ripplechunks_preview.js), so it needs no other add-on. Running the exact same workflow twice does nothing the second time, so no preview then; change the seed or any setting to force a fresh run. If it looks wrong, cancel the run. Watch the ComfyUI console for lines starting `[RippleChunked]`. **sampler_preview** (default "first chunk only") controls the sampler's own step-by-step preview, which is how you check the reference image is being picked up: it is shown for chunk 1, then hidden for later chunks so the finished-chunk preview stays on screen. Choose "every chunk" to always see it, or "never". Running the exact same workflow twice does nothing the second time (ComfyUI reuses the old result), so set the seed's control_after_generate to randomize, or change any setting, to force a fresh run.
- **mask_smooth** (default 3): averages the mask over neighbouring frames to stop it flickering. **debug_mask_overlay**: tints the pasted area red so you can see exactly what is pasted (turn off for the real render).
- **detail_mask** (optional input) + **mask_grow** / **mask_feather**: connect a mask for the whole original video (for example SAM3 on "face, hands"). It must be made from the FULL ORIGINAL video, not from the Ripple Clip output. After all chunks are joined, the original pixels are pasted back on top where the mask is white. grow widens the mask, feather softens its edge.
- **blend_frames** (default 2): frames faded at each join. 1 is a clean cut. Higher can ghost moving edges.
- **seam_search** (default on): the join is made at the overlap frame where the two chunks look most alike.
- **carry_anchors** (default 4): previous-result frames locked into the next chunk across the overlap, so the two stay matched.
- Place the edited anchor frame inside the first window.

## Notes
- All finished frames are held in system RAM (about 12 MB per frame at 1 MP): 30 s is roughly 9 GB RAM.
- Edits carry forward chunk to chunk, so a very long clip can slowly drift. Watch the later chunks.
- A video range shorter than one window is trimmed to the nearest valid H3 length (17k + 5).

## Tests
`python tests/test_chunk_core.py` covers chunk planning, stitching and masks; `node tests/test_preview_window.mjs` covers the preview window (no ComfyUI needed).
