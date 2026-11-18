"""Limb layers for build-layers.py (ER4): each rig.limbs entry becomes limb_<name>.png, and base_bare.png is the base
without them.

A limb layer keeps the source pixels inside the limb's outline exactly (alpha included, no soft edge), so the resting
avatar - base_bare under the limb layers - shows the source unchanged. Under each limb, base_bare continues only the
visible pixels next to it, within a band as wide as the limb can move at its limits: a covered pixel is body where the
visible body lies on both sides of the limb along its row or its column (body edges run straight on under the limb) or
where the nearest visible pixel is body; elsewhere, and behind the limb's antialiased silhouette, it is background.
Body colours are filled in from the neighbouring visible colours with OpenCV Telea; given a texture fill (build-layers.py
passes tools/cloth/behind.py), the body under a limb goes on with the garment's own texture instead - its folds, seams
and print copied from the visible body round the limb - so a held arm swung out uncovers the jacket, not a flat patch.
base.png keeps the limbs, so runtimes that do not know rig.limbs still draw them.

Loaded by build-layers.py from its own folder only when the rig has limbs.
"""

import math
import re

import cv2
import numpy as np

NAME = re.compile(r"[A-Za-z0-9_]+")
SWING = (4.0, 10.0)  # default and largest swing limit, degrees (runtime: packages/runtime/src/limbs/chain.ts)
BEND = (6.0, 12.0)
BAND_MARGIN = 6  # px beyond the farthest an edge can move
SOLID = 250  # source alpha from which a pixel counts as part of the body, not of an antialiased silhouette
RING_WIDTH = 16  # px: the ring round a limb whose colour is the garment beside it
BESIDE_TINT = 10.0  # CIE a*b* distance from that colour within which a pixel is the garment, not a strap or skin
DOMINANT = 0.6  # share of the ring within it that makes the garment one colour
PATCH = 22  # px: the side of a patch the texture is copied in (tools/cloth/behind.py)


def _points(value, path, minimum, w, h):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < minimum or not np.isfinite(p).all():
        raise ValueError(f"{path}: expected at least {minimum} finite [x, y] points")
    if (p[:, 0] < 0).any() or (p[:, 0] > w).any() or (p[:, 1] < 0).any() or (p[:, 1] > h).any():
        raise ValueError(f"{path}: points must be within source.png")
    return p


def validate(rig, shape, names):
    """Input checks of rig.limbs (the runtime's validateRig repeats them); `names` are the layer names already taken."""
    h, w = shape
    seen = set()
    for i, limb in enumerate(rig.get("limbs", [])):
        name = limb.get("name")
        layer = f"limb_{name}"
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in seen or layer in names:
            raise ValueError(f"limbs[{i}].name must be a unique name of letters, digits and underscores")
        seen.add(name)
        names.add(layer)
        if limb.get("kind") not in ("arm", "leg", "prop"):
            raise ValueError(f"limbs[{i}].kind must be arm, leg or prop")
        joints = _points(limb.get("joints"), f"limbs[{i}].joints", 3, w, h)
        if len(joints) != 3:
            raise ValueError(f"limbs[{i}].joints: expected root, middle and end")
        if min(np.hypot(*(joints[1] - joints[0])), np.hypot(*(joints[2] - joints[1]))) < 1:
            raise ValueError(f"limbs[{i}].joints must be at least 1 px apart")
        outline = _points(limb.get("outline"), f"limbs[{i}].outline", 3, w, h)
        if abs(cv2.contourArea(outline.astype(np.float32))) < 1:
            raise ValueError(f"limbs[{i}].outline must have a positive area")
        for key, (_, top) in (("swing", SWING), ("bend", BEND)):
            value = limb.get(key)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= top):
                raise ValueError(f"limbs[{i}].{key} must be between 0 and {top:g} degrees")


def reach(limb):
    """How far (px) an edge of the limb can move at its swing and bend limits, plus a margin: the band to fill."""
    a, b, c = (np.asarray(p, float) for p in limb["joints"])
    swing = math.radians(min(limb.get("swing", SWING[0]), SWING[1]))
    bend = math.radians(min(limb.get("bend", BEND[0]), BEND[1]))
    first, second = float(np.hypot(*(b - a))), float(np.hypot(*(c - b)))
    return int(math.ceil(2 * first * math.sin(swing / 2) + 2 * second * math.sin((swing + bend) / 2) + BAND_MARGIN))


def masks(rgba, rig):
    """Each limb's pixels: inside its outline and not fully transparent."""
    shape = rgba.shape[:2]
    result = []
    for limb in rig.get("limbs", []):
        poly = np.zeros(shape, np.uint8)
        cv2.fillPoly(poly, [np.rint(np.asarray(limb["outline"], float)).astype(np.int32)], 255)
        result.append((poly > 0) & (rgba[..., 3] > 0))
    return result


def crop(layer, mask, pad=2):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros((1, 1, 4), np.uint8), [0, 0, 1, 1]
    h, w = mask.shape
    x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
    x1, y1 = min(int(xs.max()) + pad + 1, w), min(int(ys.max()) + pad + 1, h)
    return layer[y0:y1, x0:x1].copy(), [x0, y0, x1 - x0, y1 - y0]


def opaque_colours(rgba, solid=None):
    """The RGB of `rgba` with every not-opaque pixel (or every pixel outside `solid`) taking the colour of the nearest
    opaque one, so that inpainting never pulls in the meaningless colour of transparent pixels."""
    solid = rgba[..., 3] >= 128 if solid is None else solid
    if not solid.any() or solid.all():
        return np.ascontiguousarray(rgba[..., :3])
    _, labels = cv2.distanceTransformWithLabels((~solid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(solid)
    nearest = np.clip(labels.astype(np.int64) - 1, 0, len(xs) - 1)
    rgb = rgba[..., :3].copy()
    rgb[~solid] = rgba[ys[nearest[~solid]], xs[nearest[~solid]], :3]
    return np.ascontiguousarray(rgb)


def enclosed(hole, visible):
    """Hole pixels with visible body right next to the hole on both sides along their row or along their column."""
    result = np.zeros(hole.shape, bool)
    for transpose in (False, True):
        h, v, out = (hole.T, visible.T, result.T) if transpose else (hole, visible, result)
        for r in np.flatnonzero(h.any(axis=1)):
            edges = np.diff(np.concatenate(([0], h[r].astype(np.int8), [0])))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
                if a > 0 and b < h.shape[1] and v[r, a - 1] and v[r, b]:
                    out[r, a:b] = True
    return result


def garment_beside(rgba, mask, visible):
    """The visible pixels of the garment beside a limb that its texture may be copied from. When the body round the limb
    (a RING_WIDTH ring) is mostly of one colour (DOMINANT of it within BESIDE_TINT of its median CIE a*b*) and that colour
    holds whole patches (PATCH), only pixels of that colour: a strap, buttons, a zipper or skin lying beside a plain
    jacket's sleeve are not copied under it. A print of several colours (stripes, flowers) keeps them all."""
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((2 * RING_WIDTH + 1,) * 2, np.uint8)) > 0
    ring &= visible & ~mask
    if ring.sum() < 16:
        return visible
    lab = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
    tint = np.median(lab[ring][:, 1:], axis=0)
    plain = np.hypot(lab[..., 1] - tint[0], lab[..., 2] - tint[1]) <= BESIDE_TINT
    if plain[ring].mean() < DOMINANT:
        return visible
    plain &= visible
    whole = cv2.erode(plain.astype(np.uint8), np.ones((PATCH, PATCH), np.uint8), borderType=cv2.BORDER_CONSTANT,
                      borderValue=0)
    return plain if whole.any() else visible


def complete_sleeve(rgba, mask, limb, torso_x, store):
    """A foreground strap can hide the inner half of a sleeve. Continue the free cuff's
    cross-section through that notch in forearm coordinates, carrying the sleeve's weave,
    and keep the occluding bag on a separate layer attached to the torso."""
    width = limb.get("coveredSleeveWidth")
    if not width:
        return
    elbow, wrist = np.asarray(limb["joints"][1:], np.float32)
    axis = wrist - elbow
    length = np.linalg.norm(axis)
    axis /= length
    inward = np.array([axis[1], -axis[0]])
    if (torso_x - elbow[0]) * inward[0] < 0:
        inward = -inward
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    h, w = mask.shape
    y0, y1 = max(0, int(ys.min() - 2 * width)), min(h, int(ys.max() + 2 * width + 1))
    x0, x1 = max(0, int(xs.min() - 2 * width)), min(w, int(xs.max() + 2 * width + 1))
    y, x = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    along = (x - elbow[0]) * axis[0] + (y - elbow[1]) * axis[1]
    across = (x - elbow[0]) * inward[0] + (y - elbow[1]) * inward[1]
    start, stop = int(-0.4 * length), int(length)
    rows = np.clip(np.rint(along).astype(np.int32) - start, 0, stop - start)
    visible = mask[y0:y1, x0:x1] & (along >= start) & (along <= stop)
    lo = np.full(stop - start + 1, np.inf, np.float32)
    hi = np.full_like(lo, -np.inf)
    np.minimum.at(lo, rows[visible], across[visible])
    np.maximum.at(hi, rows[visible], across[visible])
    samples = np.flatnonzero(np.isfinite(hi))
    if not len(samples):
        return
    def contour(values):
        v = np.interp(np.arange(len(values)), samples, values[samples]).astype(np.float32)
        return cv2.GaussianBlur(v[:, None], (1, 0), sigmaX=0, sigmaY=max(2, width * 0.025)).ravel()
    outer, inner = contour(lo), contour(hi)
    cuff = (samples + start >= length * 0.73) & (samples + start <= length * 0.94)
    if not cuff.any():
        return
    full = np.clip(np.median((inner - outer)[samples[cuff]]), width, width * 1.5)
    # Where the visible arm already has its full section the original drawing is complete.
    deficit = np.maximum(0, full - (inner - outer))
    ease = np.clip(np.arange(len(lo)) / (0.26 * length), 0, 1)
    extra = deficit * ease * np.clip((deficit - 0.10 * full) / (0.18 * full), 0, 1)
    extra = cv2.GaussianBlur(extra[:, None], (1, 0), sigmaX=0, sigmaY=max(2, width * 0.04)).ravel()
    # The drawn piping at the bag's edge can occupy part of the segmented sleeve.
    # Replace that strip as well, so the swinging arm carries only its own leather.
    boundary = inner[rows] - np.minimum(extra[rows] * 0.5, 0.15 * full)
    depth = across - boundary
    valid = (along >= start) & (along < stop) & (extra[rows] > 1)
    region = valid & (depth >= 0) & (across <= inner[rows] + extra[rows]) & (rgba[y0:y1, x0:x1, 3] >= SOLID)
    if not region.any():
        return
    # Follow the same folds across the hidden side without reflecting the painted
    # silhouette or repeating the piping. The far edge samples the sleeve's core.
    strip = np.maximum(2, boundary - outer[rows])
    donor = boundary - strip * (0.18 + 0.36 * np.tanh(np.maximum(depth, 0) / (0.36 * strip)))
    map_x = (elbow[0] + along * axis[0] + donor * inward[0]).astype(np.float32)
    map_y = (elbow[1] + along * axis[1] + donor * inward[1]).astype(np.float32)
    sleeve = cv2.remap(rgba, map_x, map_y, cv2.INTER_LINEAR)
    # Blend across the original sleeve on the moving bone, before reaching the occluder.
    seam = np.clip(depth / np.maximum(1, inner[rows] - boundary), 0, 1)
    seam = np.where(mask[y0:y1, x0:x1], seam**2 * (3 - 2 * seam), 1)
    edge = np.clip(inner[rows] + extra[rows] - across, 0, 1)
    sleeve[..., 3] = np.where(region, 255 * seam * edge, 0).astype(np.uint8)
    # The bag extends to the body side of the sleeve; it stays in front throughout the arm's swing.
    covered = valid & (depth >= 0) & (depth <= extra[rows] + reach(limb)) & (rgba[y0:y1, x0:x1, 3] >= SOLID)
    front = rgba[y0:y1, x0:x1].copy()
    front[..., 3] = np.where(covered, 255, 0).astype(np.uint8)
    for prefix, image in (("sleeve", sleeve), ("sleeve_cover", front)):
        tile, rect = crop(image, image[..., 3] > 0)
        rect[0] += x0
        rect[1] += y0
        store(f"{prefix}_{limb['name']}", tile, rect)


def cut(rgba, base, rig, store, texture=None):
    """Store limb_<name> for every limb; return base_bare (the runtime base without the limbs). `texture`: a fill
    `texture(rgba, hole, source)` returning RGBA with the hole filled from the `source` pixels (tools/cloth/behind.py),
    used for the body under each limb; without it the body is filled smoothly (Telea)."""
    bare = base.copy()
    limb_masks = masks(rgba, rig)
    anywhere = np.zeros(rgba.shape[:2], bool)
    for mask in limb_masks:
        anywhere |= mask
    for limb, mask in zip(rig.get("limbs", []), limb_masks, strict=True):
        layer = rgba.copy()
        layer[..., 3] = np.where(mask, rgba[..., 3], 0).astype(np.uint8)
        store(f"limb_{limb['name']}", *crop(layer, mask))
        complete_sleeve(rgba, mask, limb, rig["body"]["chest"]["cx"] if "body" in rig else rgba.shape[1] * 0.5, store)
        if not mask.any():
            continue
        hole = mask.astype(np.uint8) * 255
        rgb = cv2.inpaint(opaque_colours(np.where(mask[..., None], 0, bare).astype(np.uint8)), hole, 9, cv2.INPAINT_TELEA)
        # Body or background under each covered pixel; only a band the limb can uncover is continued, and behind its
        # antialiased silhouette there is background.
        visible = (bare[..., 3] >= 128) & ~mask
        inside, labels = cv2.distanceTransformWithLabels(hole, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(hole == 0)
        nearest = np.clip(labels.astype(np.int64) - 1, 0, len(xs) - 1)
        body = enclosed(mask, visible) | (visible[ys[nearest], xs[nearest]] & mask)
        solid = body & (inside <= reach(limb)) & (rgba[..., 3] >= SOLID)
        bare[mask, :3] = rgb[mask]
        if texture is not None and solid.any():
            # the garment beside the limb goes on under it: patches of the visible body round it (no limb, no
            # background), the whole limb unknown so that nothing of it is copied (where no patch reaches, the smooth
            # colours stay)
            source = garment_beside(bare, mask, (bare[..., 3] == 255) & ~anywhere)
            if source.any():
                # the limb's pixels the fill keeps as they are (where it hung over the background) are the garment's
                # nearest colours: what is copied next to them matches the garment, not the smooth fill's blend
                beside = np.dstack([opaque_colours(bare, source), bare[..., 3]])
                filled = texture(np.where(mask[..., None], beside, bare).astype(np.uint8), mask, source)
                bare[solid, :3] = filled[solid, :3]
        bare[mask, 3] = np.where(solid[mask], 255, 0).astype(np.uint8)
    return bare
