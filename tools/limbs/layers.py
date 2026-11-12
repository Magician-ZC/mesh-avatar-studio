"""Limb layers for build-layers.py (ER4): each rig.limbs entry becomes limb_<name>.png, and base_bare.png is the base
without them.

A limb layer keeps the source pixels inside the limb's outline exactly (alpha included, no soft edge), so the resting
avatar - base_bare under the limb layers - shows the source unchanged. Under each limb, base_bare continues only the
visible pixels next to it, within a band as wide as the limb can move at its limits: a covered pixel is body where the
visible body lies on both sides of the limb along its row or its column (body edges run straight on under the limb) or
where the nearest visible pixel is body; elsewhere, and behind the limb's antialiased silhouette, it is background.
Body colours are filled in from the neighbouring visible colours with OpenCV Telea. base.png keeps the limbs, so
runtimes that do not know rig.limbs still draw them.

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


def opaque_colours(rgba):
    """The RGB of `rgba` with every not-opaque pixel taking the colour of the nearest opaque one, so that inpainting
    never pulls in the meaningless colour of transparent pixels."""
    solid = rgba[..., 3] >= 128
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


def cut(rgba, base, rig, store):
    """Store limb_<name> for every limb; return base_bare (the runtime base without the limbs)."""
    bare = base.copy()
    for limb, mask in zip(rig.get("limbs", []), masks(rgba, rig), strict=True):
        layer = rgba.copy()
        layer[..., 3] = np.where(mask, rgba[..., 3], 0).astype(np.uint8)
        store(f"limb_{limb['name']}", *crop(layer, mask))
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
        bare[mask, 3] = np.where(solid[mask], 255, 0).astype(np.uint8)
    return bare
