import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chunk_core import plan_chunks, anchor_positions, Stitcher, valid_length


def check_plan(start, end, L, O):
    plan = plan_chunks(start, end, L, O)
    assert plan[0][0] == start
    assert plan[-1][0] + plan[-1][1] == end or end - start < L
    for (s, n) in plan:
        assert (n - 5) % 17 == 0 and s >= start and s + n <= end
    for (a, an), (b, bn) in zip(plan, plan[1:]):
        assert b > a and b < a + an          # overlap, no gap
        assert a + an - b >= O               # at least the requested overlap
    return plan


def test_plans():
    for start in (0, 30):
        for total in (5, 22, 90, 107, 124, 125, 300, 721, 1500):
            for L, O in ((107, 22), (124, 34), (107, 6)):
                check_plan(start, start + total, L, O)
    assert plan_chunks(0, 124, 124, 22) == [(0, 124)]
    assert plan_chunks(0, 60, 124, 22) == [(0, 56)]
    assert plan_chunks(0, 200, 107, 22) == [(0, 107), (46, 107), (93, 107)]
    assert plan_chunks(0, 190, 107, 22) == [(0, 107), (83, 107)]
    assert plan_chunks(0, 721, 107, 22)[-1] == (614, 107)


def test_bad_args():
    for args in ((0, 100, 100, 10), (0, 100, 107, 0), (0, 100, 107, 91), (0, 3, 107, 22)):
        try:
            plan_chunks(*args)
        except ValueError:
            continue
        raise AssertionError(args)


def test_anchors():
    assert anchor_positions(22, 1) == [0]
    assert anchor_positions(22, 2) == [0, 21]
    assert anchor_positions(22, 3) == [0, 10, 21] or anchor_positions(22, 3) == [0, 11, 21]
    assert anchor_positions(2, 3) == [0, 1]


def test_stitch_identity_and_blend():
    s = Stitcher(10, 10 + 200, 4, 4)
    plan = plan_chunks(10, 210, 107, 22)
    truth = np.arange(200, dtype=np.uint8).reshape(-1, 1, 1, 1) * np.ones((1, 4, 4, 3), np.uint8)
    for (a, n) in plan:
        s.add(truth[a - 10:a - 10 + n], a)
    assert s.written == 200
    assert np.array_equal(s.result(), truth)          # identical overlaps stay identical
    # different overlap content: short fade, then the new chunk fully replaces the old tail (no ghosting)
    s = Stitcher(0, 30, 2, 2)
    s.add(np.zeros((20, 2, 2, 3), np.uint8), 0)
    ov = s.add(np.full((20, 2, 2, 3), 200, np.uint8), 10, blend=4)
    r = s.result()
    assert ov == 4 and r[9, 0, 0, 0] == 0 and 0 < r[10, 0, 0, 0] < r[13, 0, 0, 0] < 200 and r[14, 0, 0, 0] == 200 and r[19, 0, 0, 0] == 200 and r[20, 0, 0, 0] == 200
    assert s.get(25)[0, 0, 0] == 200


def test_seam_search():
    s = Stitcher(0, 40, 6, 6)
    old = np.full((20, 6, 6, 3), 50, np.uint8)
    s.add(old, 0)
    new = np.full((20, 6, 6, 3), 200, np.uint8)
    new[3:4] = 50                               # the chunks agree at overlap frame 3 (absolute 13)
    s.add(new, 10, blend=1)
    r = s.result()
    assert s.last_seam[0] == 3 and s.last_seam[1] == 0
    assert (r[10:13] == 50).all()               # old kept before the seam
    assert (r[13] == 50).all() and (r[14:] == 200).all()   # no ghost frames, only old or new
    # without seam search it joins at the start of the overlap
    s2 = Stitcher(0, 40, 6, 6); s2.add(old, 0); s2.add(new, 10, blend=1, seam_search=False)
    assert s2.last_seam[0] == 0 and (s2.result()[11:13] == 200).all() and s2.result()[10, 0, 0, 0] == 125


def test_preview(tmp_path=None):
    import tempfile
    from chunk_core import write_preview
    frames = np.random.randint(0, 255, (20, 90, 50, 3), dtype=np.uint8)
    out = os.path.join(tempfile.mkdtemp(), "p.webp")
    write_preview(frames, out, max_side=40)
    from PIL import Image
    im = Image.open(out)
    assert im.n_frames == 10 and max(im.size) <= 40 and im.size[0] % 2 == 0


def test_mask_tools():
    from chunk_core import mask_frame_index, fit_mask, soften, paste
    assert [mask_frame_index(i, 10, 10) for i in range(10)] == list(range(10))
    assert mask_frame_index(0, 30, 24) == 0 and mask_frame_index(23, 30, 24) == 29 and mask_frame_index(5, 5, 24) == 1
    m = np.zeros((2, 20, 40), np.float32); m[:, 5:10, 10:20] = 1
    same = fit_mask(m, [0, 0, 1, 1], (40, 20))
    assert np.array_equal(same, m)
    half = fit_mask(m, [0.25, 0, 0.5, 1], (20, 20))            # crop the middle half (x 10..30) of the width
    assert half.shape == (2, 20, 20) and half[0, 7, 2] == 1 and half[0, 7, 15] == 0
    big = fit_mask(m, [0, 0, 1, 1], (80, 40))
    assert big.shape == (2, 40, 80) and big[0, 15, 30] == 1
    dot = np.zeros((1, 21, 21), np.float32); dot[0, 10, 10] = 1
    g = soften(dot, grow=2)
    assert g.sum() == 25 and g[0, 8:13, 8:13].all()
    assert soften(g, grow=-2).sum() == 1                       # shrink back to the dot
    f = soften(g, feather=6)
    assert 0 < f[0, 10, 10] <= 1 and f[0, 0, 0] < 0.01 and abs(f.sum() - 25) < 1.0   # edge softens, area kept
    assert (np.diff(f[0, 10, :11]) >= -1e-6).all()            # monotonic fall-off
    gen = np.full((1, 4, 4, 3), 100, np.uint8); org = np.full((1, 4, 4, 3), 200, np.uint8)
    mk = np.zeros((1, 4, 4), np.float32); mk[0, :, 2:] = 1; mk[0, 0, 0] = 0.5
    out = paste(gen, org, mk)
    assert out[0, 1, 0, 0] == 100 and out[0, 1, 3, 0] == 200 and out[0, 0, 0, 0] == 150


def test_extras():
    from chunk_core import temporal_mean, tint, contact_sheet
    m = np.zeros((11, 3, 3), np.float32); m[5] = 1
    s = temporal_mean(m, 1)
    assert s.shape == m.shape and np.allclose(s[4:7, 0, 0], 1 / 3) and s[3, 0, 0] == 0 and s[7, 0, 0] == 0
    assert np.array_equal(temporal_mean(m, 0), m)
    flat = np.full((6, 3, 3), 0.5, np.float32)
    assert np.allclose(temporal_mean(flat, 2), 0.5)                     # edges keep their value
    fr = np.full((2, 2, 2, 3), 100, np.uint8); mk = np.zeros((2, 2, 2), np.float32); mk[:, 0, 0] = 1
    o = tint(fr, mk, strength=0.5)
    assert tuple(o[0, 0, 0]) == (178, 50, 50) and tuple(o[0, 1, 1]) == (100, 100, 100)
    sheet = contact_sheet(np.zeros((20, 90, 50, 3), np.uint8), count=6, height=64)
    assert sheet.shape[0] == 64 and sheet.shape[1] == 6 * round(50 * 64 / 90)


if __name__ == "__main__":
    test_plans(); test_bad_args(); test_anchors(); test_stitch_identity_and_blend(); test_seam_search(); test_preview(); test_mask_tools(); test_extras()
    print("all tests passed")
