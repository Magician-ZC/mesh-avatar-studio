"""Strand layers for build-layers.py (--strand-layers): a held lock moves on its own layer over the picture filled in
behind it, instead of bending the base.

Without them a held strand bends the base like its sway does: the hair round the strand's line follows it with a soft
fall-off, so a pull of more than its sway (`max`) stretches the lock's edges and drags what lies beside it. For each
strand with a `reach` (M11 rigs) two layers are cut:

  strand_<i>.png        the lock: the hair along the strand (the hair mask inside LOCK sigmas of its line, below its
                        root, and the thin hair hanging from it out to REGION sigmas, told by colour from what lies
                        round it), with a hard edge EDGE px outside the hair, so at rest it and the plate under it draw
                        the drawing exactly
  strand_<i>_under.png  the plate: the drawn base (base_bare, base_cloth or base) as far round the strand as it bends it
                        (REGION sigmas), with the lock taken out and filled in smoothly from the opaque pixels round it
                        that are not hair (transparent where the lock hung over the background); the runtime draws it in
                        place of the
                        base there, so the base's own bend never shows

layers.json gets `strands`: { <strand name>: { lock, under, unsure } }: the layer names, and the share of the lock's
outline that borders the background (`unsure`): there the drawing does not tell how far the figure goes on behind the
lock, so a build may keep that strand's pull short or leave it out. base.png is not changed, and the runtime
draws both layers only while the strand is held and springs back, so frames without a grab, avatars without the data
and older runtimes are as before.

A strand is left out (and the reason returned) when its lock cannot be told: fewer than MIN_PIXELS hair pixels along
its line. Loaded by build-layers.py from its own folder only when --strand-layers is given.
"""

import cv2
import numpy as np

LOCK = 1.5  # sigmas from the strand's line: the lock
REGION = 2.2  # sigmas from the strand's line: where the base bends with the strand (its weight there is below 1%)
ROOT = 0.03  # of the strand's length: the lock starts below its root (the runtime's weight rises from 0.04)
EDGE = 3  # px the lock reaches beyond the hair: its antialiased rim goes with it
MIN_PIXELS = 200
PALETTE = 4  # colours of the lock and of what lies round it
MARGIN = 6.0  # Lab units a thin hair pixel is nearer the lock's colours than the others (and a pixel the fill uses
# is nearer the others' by)


def corridor(nodes, box):
    """Distance (px) to the polyline and the position t (0 root .. 1 tip) of the nearest point, over box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    best = np.full(xx.shape, np.inf, np.float32)
    along = np.zeros(xx.shape, np.float32)
    count = len(nodes) - 1
    for i in range(count):
        (ax, ay), (bx, by) = nodes[i], nodes[i + 1]
        vx, vy = bx - ax, by - ay
        u = np.clip(((xx - ax) * vx + (yy - ay) * vy) / max(vx * vx + vy * vy, 1e-6), 0, 1)
        d = np.hypot(xx - ax - vx * u, yy - ay - vy * u)
        closer = d < best
        best[closer] = d[closer]
        along[closer] = ((i + u) / count)[closer]
    return best, along


def hair_like(lab, core, other, candidates, margin=MARGIN):
    """Candidates whose colour is nearer the lock's (core) than what lies round it (other), by palettes of PALETTE colours."""
    if core.sum() < PALETTE or other.sum() < PALETTE:
        return np.zeros(candidates.shape, bool)

    def palette(values):
        values = np.ascontiguousarray(values[:: max(1, len(values) // 4000)], np.float32)
        cv2.setRNGSeed(0)
        _, _, centres = cv2.kmeans(values, min(PALETTE, len(values)), None,
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 1, cv2.KMEANS_PP_CENTERS)
        return centres

    def nearest(values, centres):
        return np.min(np.linalg.norm(values[:, None, :] - centres[None, :, :], axis=2), axis=1)

    values = lab[candidates]
    out = np.zeros(candidates.shape, bool)
    out[candidates] = nearest(values, palette(lab[core])) + margin < nearest(values, palette(lab[other]))
    return out


def cut(drawn, hair, rig, smooth_fill):
    """The strand layers of the strands with a reach: ({name: (image, rect)}, {strand name: {lock, under}}, [notes]).

    drawn: the base the runtime draws (RGBA, source size); hair: 8-bit hair mask (source size, white is hair);
    smooth_fill(rgb, known): build-layers.py's push-pull fill (the colours, from the pixels round the lock that are not
    hair)."""
    h, w = hair.shape
    layers, data, notes = {}, {}, []
    for index, strand in enumerate(rig.get("strands", [])):
        if not strand.get("reach"):
            continue
        nodes = [(float(x), float(y)) for x, y in strand["nodes"]]
        sigma = float(strand["sigma"])
        pad = int(np.ceil(REGION * sigma)) + EDGE + 2
        xs, ys = [p[0] for p in nodes], [p[1] for p in nodes]
        box = (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad), min(w, int(max(xs)) + pad + 1), min(h, int(max(ys)) + pad + 1))
        x0, y0, x1, y1 = box
        distance, along = corridor(nodes, box)
        window = (slice(y0, y1), slice(x0, x1))
        opaque = drawn[window][..., 3] > 0
        parsed = hair[window] >= 128
        lock = (distance <= LOCK * sigma) & (along > ROOT) & parsed & opaque
        # thin hair the mask leaves out (tips, loose hairs) hangs together with the lock: told by colour
        lab = cv2.cvtColor(np.ascontiguousarray(drawn[window][..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
        near = (distance <= REGION * sigma) & (along > ROOT) & opaque
        solid = drawn[window][..., 3] > 200
        lock |= hair_like(lab, lock & solid, near & ~parsed & solid & (distance > LOCK * sigma * 0.5), near & ~lock)
        # only the hair that hangs together with the strand's line
        line = np.zeros(lock.shape, np.uint8)
        start = max(1, int(np.ceil(0.1 * (len(nodes) - 1))))
        cv2.polylines(line, [np.rint(np.asarray(nodes[start - 1 :]) - (x0, y0)).astype(np.int32)], False, 1, 3)
        count, labels = cv2.connectedComponents(lock.astype(np.uint8), connectivity=8)
        keep = np.unique(labels[(line > 0) & lock])
        lock = np.isin(labels, keep[keep > 0])
        if lock.sum() < MIN_PIXELS:
            notes.append(f"{strand['name']}: too little hair along the strand")
            continue
        grown = cv2.dilate(lock.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE + 1,) * 2)) > 0
        grown &= opaque & (along > ROOT)
        region = ((distance <= REGION * sigma) & (along >= ROOT * 0.5)) | grown
        # the lock: the drawn pixels inside it, with their own alpha
        lock_image = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        lock_image[grown] = drawn[window][grown]
        # the plate: the drawn base round it with the lock filled in. Opaque where the alpha inpainted from round the
        # lock is at least half (the figure goes on behind it: never a hole where it lay over a dress or skin; beside an
        # outline the fill may reach a little past it), transparent where the lock hung over the background; the
        # colours from the solid pixels round it that are not hair and not like it (else any), smoothly
        filled = drawn[window].copy()
        behind = cv2.inpaint(np.ascontiguousarray(filled[..., 3]), grown.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
        filled[..., 3][grown] = np.where(behind[grown] >= 128, 255, 0).astype(np.uint8)
        # the share of the lock's outline that borders the background: there the drawing does not tell how far the
        # figure goes on behind the lock (beside a dress's outline the fill may reach past it, or a notch may open)
        ring = (cv2.dilate(grown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~grown
        unsure = float((ring & ~opaque).sum()) / float(max(1, ring.sum()))
        known = solid & ~grown & ~parsed & ~hair_like(lab, lock & solid, solid & ~parsed & ~grown, solid & ~grown, -MARGIN)
        if known.sum() < MIN_PIXELS:
            known = solid & ~grown
        filled[..., :3][grown] = smooth_fill(np.ascontiguousarray(drawn[window][..., :3]), known)[grown]
        rows, cols = np.nonzero(region)
        ry0, ry1, rx0, rx1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        lrows, lcols = np.nonzero(grown)
        ly0, ly1, lx0, lx1 = lrows.min(), lrows.max() + 1, lcols.min(), lcols.max() + 1
        name = f"strand_{index}"
        layers[name] = (np.ascontiguousarray(lock_image[ly0:ly1, lx0:lx1]), [int(x0 + lx0), int(y0 + ly0), int(lx1 - lx0), int(ly1 - ly0)])
        layers[f"{name}_under"] = (np.ascontiguousarray(filled[ry0:ry1, rx0:rx1]), [int(x0 + rx0), int(y0 + ry0), int(rx1 - rx0), int(ry1 - ry0)])
        data[strand["name"]] = {"lock": name, "under": f"{name}_under", "unsure": round(unsure, 4)}
    return layers, data, notes
