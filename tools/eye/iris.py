"""Iris layers: the iris as its own sprite over an eye white that has no iris.

The eye ball layer holds the eye white and the iris in one picture, and the runtime follows the gaze by warping
that picture inside the opening. That suits a tall drawn eye with white all around the iris. On an eye whose iris
is as tall as the opening (most semi-realistic eyes) the warp has nowhere to go: the iris top and bottom stay with
the lids while its middle moves, and near lids and corners the warp reads beyond the cut-out.

With `eyes[i].iris` in the rig (the iris outline as an ellipse, also where the lids cover it) two more layers are
cut for that eye, and a runtime that knows them moves the iris as a rigid sprite instead:

- `eye{i}_white`: the eye ball with the iris painted out (the eye white carried on across it), carried on a few
  pixels beyond the opening;
- `eye{i}_iris`: the iris, with as much of the eye white around it as it takes to reach plain eye white (the dark
  ring and any rim light belong to the iris), continued under the lids from its own rings.

Over each other, at rest, they are the eye ball again. Along the lids the picture is a blend of the iris with the
lid or lash over it; there the blend is taken apart: the sprite holds the iris under it, and the lid's share is
returned as a matte for the open-eye layer, which lies over the eye layers (so the lid's soft edge stays with the
lid when the iris moves, and the three together are the picture).

`eye{i}_ball` is still written, so older runtimes render the avatar as before. When the picture does not allow it
(no eye white beside the iris, hardly any iris showing) nothing is cut for that eye and the reason is returned.
"""

import cv2
import numpy as np

RIM = 2  # px just inside the opening that are not taken as iris colour: blends with lashes and lids
BLEED = 5  # px the eye white carries on beyond the opening: under an open-eye layer the runtime draws it 3 px out
FEATHER = 2.0  # px over which the sprite's edge fades
MARGIN_MIN = 2.0  # px the sprite reaches beyond the iris outline at least
MARGIN_MAX = 0.2  # ... and at most, in iris radii
FLAT = 6.0  # grey levels: the eye white is plain where it changes less than this over 2 px
SIDE = 0.55  # the eye white beside the iris is read within this share of the iris height around its centre
MIN_WHITE = 0.06  # of the opening: eye white needed beside the iris to paint the iris out
MIN_SEEN = 0.25  # of the iris that lies in the opening: how much must show cleanly
RING_EDGE = 1.5  # px beyond the outline that still belong to the iris where it is hidden (its dark ring is soft)
LID_REACH = 3  # px from a lid over which the eye white takes the lid's shading
PAINT_RADIUS = 5
LUMA = np.float32([0.299, 0.587, 0.114])
GAIN = (0.45, 1.1)  # how much darker or lighter than its rings the continued iris may be
SEPARATE = 24.0  # colour distance under which a lid and the iris under it cannot be told apart
MATTE_MIN = 0.08  # a lid's share of a pixel below this is taken as none
MATTE_MAX = 0.85  # ... and above this as all of it: what is under it is made up
MATTE_TRUST = 56.0  # grey levels: an iris colour taken out from under a lid is kept only this close to its ring's colour
MATTE_EXACT = 8.0  # grey levels: a pixel the lid and the iris under it do not give back this closely is left alone


def _nearest(values, known):
    """`values` everywhere taken from the nearest pixel where `known`."""
    _, labels = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(known)
    lookup = np.zeros((int(labels.max()) + 1, 2), np.int32)
    lookup[labels[ys, xs]] = np.stack([ys, xs], 1)
    at = lookup[labels]
    return values[at[..., 0], at[..., 1]]


def _margin(luma, usable, px, py, rx, ry):
    """How far beyond the outline the sprite must reach to end on plain eye white: read on each side of the iris
    as the median brightness per pixel of distance, the first distance from which it stays flat."""
    radius = min(rx, ry)
    limit = max(MARGIN_MIN, MARGIN_MAX * radius)
    beyond = (np.sqrt((px / rx) ** 2 + (py / ry) ** 2) - 1) * radius
    side = usable & (np.abs(py) < SIDE * ry)
    reach = []
    for sign in (-1, 1):
        half = side & (px * sign > 0)
        steps = np.arange(int(np.floor(MARGIN_MIN)), int(np.ceil(limit)) + 3)
        levels = []
        for step in steps:
            ring = half & (beyond >= step) & (beyond < step + 1)
            levels.append(float(np.median(luma[ring])) if int(ring.sum()) >= 3 else np.nan)
        levels = np.array(levels)
        if np.isnan(levels).sum() > len(levels) // 2:
            continue  # no eye white on this side (the iris reaches the corner)
        found = limit
        for k, step in enumerate(steps[:-2]):
            if step > limit:
                break
            if not np.isnan(levels[k]) and not np.isnan(levels[k + 2]) and abs(levels[k] - levels[k + 2]) < FLAT:
                found = float(step)
                break
        reach.append(found)
    # the sprite is fully opaque up to half a feather inside its outline: that is where the plain white must begin
    return float(min(limit, max(reach)) + FEATHER / 2) if reach else None


def _lid_shading(source, rgb, painted, core, kept, inner):
    """The painted eye white, shaded near the lids the way the eye white of the picture is: for each pixel of
    distance from a lid, the median of what the picture has there over the same eye white without the lids'
    shading. Without it the white would run bright up to the lash line where the iris was."""
    if int(inner.sum()) < 12:
        return painted
    flat = cv2.inpaint(source, (~inner).astype(np.uint8) * 255, PAINT_RADIUS, cv2.INPAINT_TELEA).astype(np.float32)
    inside = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 3)
    columns = np.nonzero(core.any(0))[0]
    rows = np.arange(core.shape[0], dtype=np.float32)[:, None]
    middle = (np.where(core, rows, 0).sum(0)[columns] / core.sum(0)[columns]).astype(np.float32)
    upper = rows < np.interp(np.arange(core.shape[1]), columns, middle)[None, :]
    shaded = painted.copy()
    for side in (upper, ~upper):
        factor = None
        for step in range(LID_REACH - 1, -1, -1):  # towards the lid
            band = side & core & (inside > step) & (inside <= step + 1)
            samples = band & kept
            if int(samples.sum()) >= 6:
                factor = np.median(rgb[samples] / np.maximum(flat[samples], 8), 0)
            if factor is not None:
                target = band & ~kept
                shaded[target] = np.clip(painted[target] * factor, 0, 255)
    return shaded


def cut(face, eye, opening, curves_inside):
    """The two layers of one eye, and the lids' matte over them.

    face: the picture the eye layers are cut from (RGBA, full size); eye: the rig's eye, with `iris`;
    opening: its opening as a mask (uint8, full size); curves_inside(shape, origin): the pixels of a window the
    runtime draws fully from the eye layers at rest (between the lid curves): the hole of the open-eye layer.
    Returns ({"white": (rgba, rect), "iris": (rgba, rect), "matte": {...}, "report": {...}}, None) or (None, reason).
    The matte, for the open-eye layer: `origin` [x, y], `alpha` (0..1, a window) and `colour` (the lid's, RGB).
    """
    iris = eye["iris"]
    cx, cy, rx, ry = (float(iris[key]) for key in ("cx", "cy", "rx", "ry"))
    height, width = opening.shape
    ys, xs = np.nonzero(opening)
    if not len(xs):
        return None, "the opening is empty"
    reach = max(rx, ry) * (1 + MARGIN_MAX) + FEATHER + BLEED + 4
    x0 = int(max(0, np.floor(min(xs.min(), cx - reach)) - BLEED - 2))
    y0 = int(max(0, np.floor(min(ys.min(), cy - reach)) - BLEED - 2))
    x1 = int(min(width, np.ceil(max(xs.max() + 1, cx + reach)) + BLEED + 2))
    y1 = int(min(height, np.ceil(max(ys.max() + 1, cy + reach)) + BLEED + 2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, "the iris is outside the picture"
    window = face[y0:y1, x0:x1]
    rgb = window[..., :3].astype(np.float32)
    have = window[..., 3] > 250
    core = opening[y0:y1, x0:x1] > 0
    as_mask = lambda m: m.astype(np.uint8)
    shrink = lambda m, n: cv2.erode(as_mask(m), np.ones((2 * n + 1, 2 * n + 1), np.uint8)) > 0
    grow = lambda m, n: cv2.dilate(as_mask(m), np.ones((2 * n + 1, 2 * n + 1), np.uint8)) > 0
    yy, xx = np.mgrid[0 : y1 - y0, 0 : x1 - x0].astype(np.float32)
    px, py = xx + 0.5 + x0 - cx, yy + 0.5 + y0 - cy
    inside_iris = (px / rx) ** 2 + (py / ry) ** 2 <= 1
    shown = core & inside_iris
    if int(shown.sum()) < 0.15 * int(inside_iris.sum()) or int(shown.sum()) < 0.1 * int(core.sum()):
        return None, "the iris outline is not on this eye"

    # what the runtime draws fully from the eye layers at rest: the hole of the open-eye layer. Inside it the layers
    # hold the picture; outside it they are covered at rest, and hold what makes the hole's edge come out right
    # when the textures are filtered: the white carries on as it is at the edge, the iris as its rings go
    drawn = curves_inside((y1 - y0, x1 - x0), (x0, y0))
    eye_area = core | drawn
    clean = shrink(core, RIM) & have
    luma = rgb @ LUMA
    margin = _margin(luma, clean & ~inside_iris, px, py, rx, ry)
    if margin is None:
        return None, "no eye white beside the iris"
    sx, sy = rx + margin, ry + margin
    ring = np.sqrt((px / sx) ** 2 + (py / sy) ** 2)
    alpha = np.clip(0.5 + (1 - ring) * min(sx, sy) / FEATHER, 0, 1).astype(np.float32)
    sprite = alpha > 0
    # what the white layer has to make up: everything the sprite covers fully. Its faded edge lies on plain eye
    # white, which stays in the white layer as it is, so the two over each other are the picture there
    hole = alpha >= 0.999

    # the eye white: the iris painted out from the white on both sides, shaded near the lids like that white;
    # beyond the opening it carries on as it is at the opening's edge (the runtime draws it there under the
    # open-eye layer, so that layer's edge never shows anything but the eye)
    known = shrink(core, 1) & have & ~hole
    if int(known.sum()) < MIN_WHITE * int(core.sum()):
        return None, "too little eye white beside the iris"
    source = np.rint(rgb).astype(np.uint8)
    painted = cv2.inpaint(source, as_mask(~known) * 255, PAINT_RADIUS, cv2.INPAINT_TELEA).astype(np.float32)
    kept = have & ~hole & drawn
    painted = _lid_shading(source, rgb, painted, core, kept, shrink(core, LID_REACH) & have & ~hole)
    within = np.where(kept[..., None], rgb, painted)
    white_rgb = np.where(drawn[..., None], within, _nearest(within, drawn)) if drawn.any() else within
    white_alpha = grow(eye_area, BLEED).astype(np.float32)

    # the iris: its own pixels where the opening shows them cleanly; under the lids each ring carries on in the
    # median colour it has where it shows, as dark or light as the nearest part that does show (a lid's shadow
    # goes on under the lid)
    seen = sprite & clean
    if int(seen.sum()) < MIN_SEEN * int((sprite & core).sum()) or int(seen.sum()) < 24:
        return None, "the iris hardly shows"
    radius = int(np.ceil(max(sx, sy)))
    rings = np.clip(np.floor(ring * radius).astype(np.int32), 0, radius + 1)
    profile = np.full((radius + 2, 3), np.nan, np.float32)
    for k in range(radius + 2):
        pixels = rgb[seen & (rings == k)]
        if len(pixels) >= 6:
            profile[k] = np.median(pixels, 0)
    filled = np.nonzero(~np.isnan(profile[:, 0]))[0]
    if not len(filled):
        return None, "the iris hardly shows"
    profile = profile[filled[np.abs(np.arange(radius + 2)[:, None] - filled[None, :]).argmin(1)]]
    model = profile[rings]
    ratio = np.where(seen, luma / np.maximum(model @ LUMA, 8), 0).astype(np.float32)
    share = cv2.GaussianBlur(seen.astype(np.float32), (0, 0), 2.5)
    local = cv2.GaussianBlur(ratio, (0, 0), 2.5) / np.maximum(share, 1e-3)
    gain = np.clip(_nearest(local, share > 0.3), *GAIN)
    gain = cv2.GaussianBlur(gain, (0, 0), 2.0)
    continued = np.clip(model * gain[..., None], 0, 255)

    # along the lids the picture blends the iris with the lid or lash over it. Taken apart: how much of each pixel
    # is the lid's (its colour: the nearest of the picture beyond the opening), and what the iris is under it.
    # Where the two colours are too alike to tell apart, the pixel stays with the iris as it is; so does the white
    # the sprite carries around the iris (it is white shaded by the lid, no blend of two known colours).
    bare = np.sqrt((px / (rx + RING_EDGE)) ** 2 + (py / (ry + RING_EDGE)) ** 2)
    rim = hole & drawn & have & ~clean & (bare <= 1)
    beyond = ~grow(eye_area, 1) & have
    lid = _nearest(rgb, beyond) if beyond.any() else continued
    apart, off = lid - continued, rgb - continued
    span = (apart**2).sum(-1)
    cover = np.clip((off * apart).sum(-1) / np.maximum(span, 1e-3), 0, 1)
    cover = np.where(rim & (span >= SEPARATE**2) & (cover >= MATTE_MIN), cover, 0).astype(np.float32)
    under = (rgb - cover[..., None] * lid) / np.maximum(1 - cover[..., None], 1e-3)
    plausible = (cover < MATTE_MAX) & (np.abs(under - continued).max(-1) <= MATTE_TRUST)
    under = np.where(plausible[..., None], np.clip(under, 0, 255), continued)
    # a pixel the two do not give back (a highlight at the lid, a lash darker than the lid) is no blend of them:
    # it stays with the iris as it is, so the layers over each other are still the picture
    given = cover[..., None] * lid + (1 - cover[..., None]) * under
    cover = np.where(np.abs(given - rgb).max(-1) > MATTE_EXACT, 0, cover).astype(np.float32)
    own = sprite & drawn & have & (cover == 0)
    iris_rgb = np.where(own[..., None], rgb, np.where((cover > 0)[..., None], under, continued))

    # the white carried around the iris is what the picture shows there (a rim light, say); under the lids nothing
    # is known about it, so there the sprite ends with the iris and its ring
    hidden = np.clip(0.5 + (1 - bare) * min(rx, ry) / FEATHER, 0, 1).astype(np.float32)
    near = np.clip(2 * cv2.GaussianBlur(seen.astype(np.float32), (0, 0), 2.0), 0, 1)
    alpha = np.where(sprite & drawn & have, alpha, hidden + (alpha - hidden) * near).astype(np.float32)

    def layer(colour, opacity, pad):
        rows, cols = np.nonzero(opacity > 0)
        top, bottom = max(0, rows.min() - pad), min(opacity.shape[0], rows.max() + 1 + pad)
        left, right = max(0, cols.min() - pad), min(opacity.shape[1], cols.max() + 1 + pad)
        image = np.dstack([np.rint(colour), np.rint(opacity * 255)]).astype(np.uint8)[top:bottom, left:right]
        image[image[..., 3] == 0] = 0
        return image, [x0 + int(left), y0 + int(top), int(right - left), int(bottom - top)]

    # the open-eye layer's share: only where it has its hole, the pixels the runtime draws fully from the eye layers
    drawn = drawn & have
    matte = np.where(drawn, cover, 0).astype(np.float32)
    # how far the layers over each other are from the picture, where the runtime draws them fully at rest
    lid_colour = np.rint(lid)
    composed = np.rint(white_rgb) * (1 - alpha[..., None]) + np.rint(iris_rgb) * alpha[..., None]
    composed = lid_colour * np.rint(matte * 255)[..., None] / 255 + composed * (1 - np.rint(matte * 255)[..., None] / 255)
    error = np.abs(composed - rgb).max(-1)[drawn]
    report = {
        "margin": round(margin, 2),
        "white": int(known.sum()),
        "seen": int(seen.sum()),
        "continued": int((alpha > 0).sum() - (sprite & drawn & have).sum()),
        "matte": int((matte > 0).sum()),
        "rest": {
            "mean": round(float(error.mean()), 2) if len(error) else 0.0,
            "p99": round(float(np.percentile(error, 99)), 1) if len(error) else 0.0,
            "max": round(float(error.max()), 1) if len(error) else 0.0,
        },
    }
    return {
        "white": layer(white_rgb, white_alpha, 3),
        "iris": layer(iris_rgb, alpha, 2),
        "matte": {"origin": [x0, y0], "alpha": matte, "colour": lid_colour.astype(np.uint8)},
        "report": report,
    }, None
