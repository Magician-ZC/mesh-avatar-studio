#!/usr/bin/env python3
"""Build local animation layers from source.png and rig geometry.

Run with: uv run --with numpy --with pillow --with opencv-python-headless
          tools/build-layers.py <project-dir> [--rig rig.draft.json]
          [--hairmask mask.png] [--cloth-under keep|hem [--hem-legs legs.png]] [--open-eyes] [--head-layers]
          [--head-plate plate.png --head-footprint mask.png] [--cheek-pull]

--hairmask takes the hair mask from a file (white is hair, exactly the size of source.png) instead of
clustering hair colours. --cloth-under keep makes base_cloth.png the base with the garments kept: nothing
is redrawn or inpainted under them and source_under.png is not read. --cloth-under hem keeps the garments too, except
a band along each garment's free edges that the garment uncovers when it swings (see tools/cloth/hem.py): there the base
is transparent, or, with --hem-legs (white = legs, source size), continues the legs that come out under the hem; the
band follows the garment's own travel (cloth[*].max) and nothing in it is generated. --open-eyes also cuts eye0_open.png and
eye1_open.png: the source pixels around each eye, with a hole for the eye ball, which runtimes that know them
show while the eye is fully open (older runtimes ignore them). --head-layers also cuts face_field.png, hair_front.png
and face_skin.png (see tools/head/layers.py): with them a head turn slides the face under the hair instead of bulging
the whole head; when the picture does not allow it the reason is printed and they are left out (older runtimes ignore
them; base.png is the same either way). --head-plate with --head-footprint makes the same three layers from a clean
plate instead of by colour (see tools/head/plate.py): the plate is the picture without the hair in front of the face
(source size; its alpha says where it is defined), the footprint says where that hair is (white, source size). The
hair is then lifted off the plate exactly, the eye layers are cut from the face without that hair, and
eye0_open_under.png / eye1_open_under.png are the open-eye layers of that face (with --open-eyes).
--cheek-pull also cuts cheek_left.png and cheek_right.png and writes "cheeks" in layers.json (see tools/head/cheeks.py):
the skin of each cheek from the face's outer contour inwards, which runtimes that know them pull past the contour; a side
whose contour cannot be traced is left out with the reason on stderr (older runtimes ignore them; base.png is the same).
Without these options the output is unchanged.

An eye with `iris` in the rig (the iris outline as an ellipse) also gets eye{i}_white.png and eye{i}_iris.png (see
tools/eye/iris.py): the eye white with the iris painted out, and the iris as a sprite, which runtimes that know them
move with the gaze instead of warping the eye ball (older runtimes ignore them and use eye{i}_ball.png, which is still
written). The open-eye layer of such an eye has a hole as large as everything the eye layers draw, so no part of the
iris stays in it. When the picture does not allow it the reason is printed and that eye is cut as before.

A rig with `limbs` (generic limb chains) also gets limb_<name>.png for every limb and base_bare.png, the base without
them, named by "bare" in layers.json (see tools/limbs/layers.py); base.png keeps the limbs for older runtimes. Under a
limb, base_bare goes on with the garment's own texture beside it (tools/cloth/behind.py).
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

# The renderer samples each eye at 24 columns. Other constants below control
# filtering, antialiasing and solver convergence, rather than illustration geometry.
EYE_SAMPLES = 24
# Open-eye layers (--open-eyes): fully opaque this far beyond the eye region (the line layers reach 2 px past
# it), then fading out over pixels where the base is the source's own; the eye ball keeps the opening shrunk
# by OPEN_EYE_HOLE, so the iris still follows the gaze.
OPEN_EYE_MARGIN = 3
OPEN_EYE_FEATHER = 4
OPEN_EYE_HOLE = 2


LIMB_LAYERS = Path(__file__).resolve().parent / "limbs" / "layers.py"
_limb_layers = None
HEAD_LAYERS = Path(__file__).resolve().parent / "head" / "layers.py"
HEAD_PLATE = Path(__file__).resolve().parent / "head" / "plate.py"
IRIS_LAYERS = Path(__file__).resolve().parent / "eye" / "iris.py"
CHEEK_PULL = Path(__file__).resolve().parent / "head" / "cheeks.py"
STRAND_LAYERS = Path(__file__).resolve().parent / "hair" / "strands.py"
_cheek_pull = None
SKIRT_LEGS = Path(__file__).resolve().parent / "cloth" / "legs.py"
HEM_BAND = Path(__file__).resolve().parent / "cloth" / "hem.py"
_hem_band = None
BEHIND = Path(__file__).resolve().parent / "cloth" / "behind.py"
_behind = None
_head_layers = None
_head_plate = None
_iris_layers = None


def hem_band():
    """tools/cloth/hem.py, loaded from this folder without writing bytecode (only --cloth-under hem needs it)."""
    global _hem_band
    if _hem_band is None:
        spec = importlib.util.spec_from_file_location("mesh_avatar_hem_band", HEM_BAND)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _hem_band = module
    return _hem_band


def behind_fill():
    """tools/cloth/behind.py, loaded from this folder without writing bytecode (only accessories cut by an outline
    and limbs need it)."""
    global _behind
    if _behind is None:
        spec = importlib.util.spec_from_file_location("mesh_avatar_behind", BEHIND)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _behind = module
    return _behind


def iris_layers():
    """tools/eye/iris.py, loaded from this folder without writing bytecode (only eyes with `iris` need it)."""
    global _iris_layers
    if _iris_layers is None:
        spec = importlib.util.spec_from_file_location("mesh_avatar_iris_layers", IRIS_LAYERS)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _iris_layers = module
    return _iris_layers


def head_plate():
    """tools/head/plate.py, loaded from this folder without writing bytecode (only --head-plate needs it)."""
    global _head_plate
    if _head_plate is None:
        spec = importlib.util.spec_from_file_location("mesh_avatar_head_plate", HEAD_PLATE)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _head_plate = module
    return _head_plate


def head_layers():
    """tools/head/layers.py, loaded from this folder without writing bytecode (only --head-layers needs it)."""
    global _head_layers
    if _head_layers is None:
        spec = importlib.util.spec_from_file_location("mesh_avatar_head_layers", HEAD_LAYERS)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _head_layers = module
    return _head_layers


def strand_layers():
    """tools/hair/strands.py, loaded from this folder without writing bytecode (only --strand-layers needs it)."""
    spec = importlib.util.spec_from_file_location("mesh_avatar_strand_layers", STRAND_LAYERS)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def cheek_pull():
    """tools/head/cheeks.py, loaded from this folder without writing bytecode (only --cheek-pull needs it)."""
    global _cheek_pull
    if _cheek_pull is None:
        spec = importlib.util.spec_from_file_location("mesh_avatar_cheek_pull", CHEEK_PULL)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _cheek_pull = module
    return _cheek_pull


def limb_layers():
    """tools/limbs/layers.py, loaded from this folder without writing bytecode (only rigs with limbs need it)."""
    global _limb_layers
    if _limb_layers is None:
        spec = importlib.util.spec_from_file_location("mesh_avatar_limb_layers", LIMB_LAYERS)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _limb_layers = module
    return _limb_layers


def polygon_mask(shape, poly):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    return m


def accessory_mask(rgba, accessory):
    if accessory.get("outline") is not None:
        # an outline (a keyhole polygon may leave a hole out) cuts the accessory whatever its colour
        return polygon_mask(rgba.shape[:2], accessory["outline"]) & np.where(rgba[..., 3] > 0, 255, 0).astype(np.uint8)
    rgb = rgba[..., :3]
    box = accessory["box"]
    x0, y0, x1, y1 = box
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    redness = (r - np.maximum(g, b)) / np.maximum(r, 1)
    m = (
        (redness > accessory["color"]["redness"])
        & (r > accessory["color"]["minRed"])
        & (rgba[..., 3] > 0)
    ).astype(np.uint8) * 255
    box_m = np.zeros_like(m)
    box_m[y0:y1, x0:x1] = 255
    m &= box_m
    # keep only the biggest connected blob (the tassel), close small holes
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    # grab the anti-aliased dark outline around the red
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)


def crop_layer(rgba, mask, pad=4, soft=True):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.zeros((1, 1, 4), np.uint8), [0, 0, 1, 1]
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    x1, y1 = (
        min(xs.max() + pad + 1, rgba.shape[1]),
        min(ys.max() + pad + 1, rgba.shape[0]),
    )
    layer = rgba.copy()
    soft = (cv2.GaussianBlur(mask, (3, 3), 0) if soft else mask).astype(
        np.float32
    ) / 255
    layer[..., 3] = (layer[..., 3].astype(np.float32) * soft).astype(np.uint8)
    return layer[y0:y1, x0:x1], [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def smooth_fill(rgb, known):
    """Colours for the pixels that are not `known`, smoothly from the known ones round them (push-pull: masked averages
    down an image pyramid, filled back up), without the streaks inpainting draws across a large hole."""
    levels = [(rgb.astype(np.float32) * known[..., None], known.astype(np.float32))]
    while min(levels[-1][1].shape) > 8 and levels[-1][1].min() == 0:
        colour, weight = levels[-1]
        levels.append((cv2.pyrDown(colour), cv2.pyrDown(weight)))
    colour, weight = levels[-1]
    estimate = colour / np.maximum(weight, 1e-6)[..., None]
    for colour, weight in reversed(levels[:-1]):
        size = (weight.shape[1], weight.shape[0])
        estimate = cv2.pyrUp(estimate, dstsize=size)
        share = np.clip(weight, 0, 1)[..., None]
        estimate = share * (colour / np.maximum(weight, 1e-6)[..., None]) + (1 - share) * estimate
    return np.clip(estimate, 0, 255).astype(np.uint8)


def behind_front(rgba, region, band, fill, front):
    """The hem band along an accessory the garment goes on behind (a handbag over a skirt) is not the hem: the garment
    continues there, so when it swings away from the accessory the base shows its colours, smoothly filled, opaque."""
    front = front > 0
    if not np.any(band & cv2.dilate(front.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)):
        return fill
    to_front = cv2.distanceTransform((~front).astype(np.uint8), cv2.DIST_L2, 5)
    to_edge = cv2.distanceTransform((region | front).astype(np.uint8), cv2.DIST_L2, 5)
    # only where the garment is opaque: under a see-through edge the drawing has nothing behind it at rest
    along = band & (to_front < to_edge) & (rgba[..., 3] == 255)
    if not np.any(along):
        return fill
    ys, xs = np.nonzero(along)
    h, w = along.shape
    pad = 64
    window = (slice(max(0, ys.min() - pad), min(h, ys.max() + pad + 1)), slice(max(0, xs.min() - pad), min(w, xs.max() + pad + 1)))
    known = region[window] & ~band[window] & (rgba[window][..., 3] > 128)
    colours = smooth_fill(rgba[window][..., :3], known)
    out = fill.copy()
    out[window][..., :3][along[window]] = colours[along[window]]
    out[window][..., 3][along[window]] = 255
    return out


def fill_behind(rgba, base, area, garment=None, store=None, name=None):
    """The base behind accessories cut by an outline (a handbag hanging over a skirt): the garment goes on behind them
    with its own texture - its print, folds and seams copied from round it (tools/cloth/behind.py) - as far as the
    figure's outline, interpolated across them, reaches, with a smooth edge there; beyond it, where they hung over the
    background, transparent. `garment` (bool): the garments traced round them, where the texture comes from (else every
    opaque pixel round them). The small tassels cut by colour keep the fade above."""
    inside = area > 0
    source = (rgba[..., 3] == 255) & ~inside
    if garment is not None and (garment & source).sum() >= inside.sum():
        source &= garment
    filled = behind_fill().fill(rgba, inside, source)
    if garment is not None and store is not None:
        # The hard accessory outline includes its antialiased fringe and contact shadow.
        # Reconstruct a wider piece, then crossfade in visible cloth beyond the old edge.
        # The outer collar is a separate layer shown as the accessory swings away.
        _, _, width, height = cv2.boundingRect(area)
        radius = min(90, round(0.14 * min(width, height)))
        if radius >= behind_fill().PATCH:
            distance = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
            wide = distance <= radius
            donors = source & ~wide
            cloth = behind_fill().fill(rgba, wide, donors, align_tiers=True)
            behind_fill().continue_seams(rgba, cloth, wide, garment)
            # The wider opening can move the inferred garment silhouette. Map its colours
            # to the established silhouette and keep the original coverage there.
            x, y, w, h = cv2.boundingRect(wide.astype(np.uint8))
            roi = (slice(y, y + h), slice(x, x + w))
            patch = cloth[roi].copy()
            valid = patch[..., 3] >= 128
            _, nearest = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            yy, xx = np.nonzero(valid)
            if len(xx):
                patch[~valid, :3] = patch[yy[nearest[~valid] - 1], xx[nearest[~valid] - 1], :3]
                inner = inside[roi]
                filled[roi][inner, :3] = patch[inner, :3]
                weight = np.clip((radius - distance[roi]) / (radius - 4), 0, 1)
                collar = (rgba[roi][..., 3] >= 250) & ~inner
                patch[..., 3] = np.where(inner, filled[roi][..., 3],
                                         np.rint(255 * weight**2 * (3 - 2 * weight) * collar)).astype(np.uint8)
                store(f"accessory_under_{name}", patch, [x, y, w, h])
    out = base.copy()
    out[inside] = filled[inside]
    return out


def inpaint_rgba(rgba, mask):
    rgb = cv2.inpaint(np.ascontiguousarray(rgba[..., :3]), mask, 9, cv2.INPAINT_TELEA)
    a = cv2.inpaint(np.ascontiguousarray(rgba[..., 3]), mask, 9, cv2.INPAINT_TELEA)
    out = np.dstack([rgb, a])
    return out


def column_span(poly, x):
    """min / max y where the vertical line at x crosses the polygon."""
    ys = []
    n = len(poly)
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        if (ax - x) * (bx - x) <= 0 and ax != bx:
            ys.append(ay + (by - ay) * (x - ax) / (bx - ax))
    if len(ys) < 2:
        raise ValueError(
            "eye opening must enclose a vertical span at every sampled column"
        )
    return min(ys), max(ys)


def eye_curves(e):
    """Top / bottom edge of the opening sampled at EYE_SAMPLES columns (for the renderer)."""
    xs = [p[0] for p in e["opening"]]
    x0, x1 = min(xs), max(xs)
    top, bot = [], []
    for i in range(EYE_SAMPLES):
        x = x0 + (x1 - x0) * min(max(i / (EYE_SAMPLES - 1), 0.002), 0.998)
        t, b = column_span(e["opening"], x)
        top.append(round(t, 2))
        bot.append(round(b, 2))
    # the ends are the eye corners: top and bottom meet there
    for k in (0, -1):
        top[k] = bot[k] = round((top[k] + bot[k]) / 2, 2)
    return {"x0": x0, "x1": x1, "top": top, "bot": bot}


def unblend(rgba, skin_rgb, mask, ink):
    """RGBA layer that reproduces rgba where it lies over skin_rgb (soft line edges keep
    their antialiasing instead of carrying a halo of the old background)."""
    orig = rgba[..., :3].astype(np.float32)
    sk = skin_rgb.astype(np.float32)
    ref = np.array(ink, np.float32)
    a = np.clip(((sk - orig) / np.maximum(sk - ref, 1)).max(-1), 0, 1)
    a *= mask > 0
    col = np.clip(
        (orig - sk * (1 - a[..., None])) / np.maximum(a[..., None], 1e-3), 0, 255
    )
    out = np.zeros_like(rgba)
    out[..., :3] = col.astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def luminance(rgb):
    return rgb @ np.array([0.3, 0.59, 0.11], np.float32)


def skin_samples(img, hole):
    """Estimate local skin from the brighter half of the opaque boundary ring."""
    ring = (
        (cv2.dilate(hole, np.ones((25, 25), np.uint8)) > 0)
        & (hole == 0)
        & (img[..., 3] > 200)
    )
    pixels = img[..., :3][ring].astype(np.float32)
    if not len(pixels):
        raise ValueError("no opaque pixels around a cut-out region to estimate skin")
    light = pixels[luminance(pixels) >= np.percentile(luminance(pixels), 55)]
    colour = np.median(light, axis=0)
    spread = np.median(np.abs(light - colour).max(-1))
    return colour, max(22, min(60, spread * 3))


def fill_skin(img, hole):
    """Continue locally estimated shading into a hole by a membrane solve."""
    if not np.any(hole):
        return img.copy()
    rgb = img[..., :3].astype(np.float32)
    colour, tolerance = skin_samples(img, hole)
    skin = (
        (hole == 0) & (np.abs(rgb - colour).max(-1) < tolerance) & (img[..., 3] > 200)
    )
    est = rgb.copy()
    known = skin.astype(np.float32)
    for sigma in (3, 6, 12, 24, 48):
        num = cv2.GaussianBlur(rgb * known[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(known, (0, 0), sigma)[..., None]
        fill = (known == 0) & (den[..., 0] > 0.02)
        est[fill] = (num / np.maximum(den, 1e-4))[fill]
        known = np.maximum(known, fill.astype(np.float32) * 0.5)
    solve = (hole > 0) | ((cv2.dilate(hole, np.ones((17, 17), np.uint8)) > 0) & ~skin)
    ys, xs = np.where(solve)
    y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, img.shape[0])
    x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, img.shape[1])
    cur = np.where(solve[..., None], est, rgb)[y0:y1, x0:x1].copy()
    m = solve[y0:y1, x0:x1]
    fixed = cur.copy()
    for _ in range(600):
        cur = np.where(m[..., None], cv2.blur(cur, (3, 3)), fixed)
    out = rgb.copy()
    out[y0:y1, x0:x1] = cur
    k = cv2.GaussianBlur((hole > 0).astype(np.float32), (0, 0), 1.0)[..., None]
    res = img.copy()
    res[..., :3] = np.clip(rgb * (1 - k) + out * k, 0, 255).astype(np.uint8)
    # A cut-out cannot create opacity outside the original illustration.
    return res


def eye_classes(rgba, eye):
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    c = eye_curves(eye)
    top = np.interp(xx, np.linspace(c["x0"], c["x1"], EYE_SAMPLES), c["top"])
    inside_x = (xx >= c["x0"]) & (xx <= c["x1"])
    rgb = rgba[..., :3].astype(np.float32)
    region = polygon_mask((h, w), eye["roi"])
    core = polygon_mask((h, w), eye["opening"])
    opaque = rgba[..., 3] > 0
    est = fill_skin(rgba, region)[..., :3].astype(np.float32)
    skin = np.abs(rgb - est).max(-1) < 22
    ink = (region > 0) & ~skin & (core == 0) & opaque
    core_pixels = rgb[(core > 0) & opaque]
    white_colour = np.median(
        core_pixels[
            luminance(core_pixels) >= np.percentile(luminance(core_pixels), 85)
        ],
        axis=0,
    )
    white = (np.abs(rgb - white_colour).max(-1) < 20) & (
        luminance(rgb) > luminance(est)
    )
    dark_pixels = rgb[(region > 0) & opaque]
    dark = np.percentile(luminance(dark_pixels), 10)
    light = np.median(luminance(est[region > 0]))
    corner_line = np.interp(xx, [c["x0"], c["x1"]], [c["top"][0], c["top"][-1]])
    upper = np.where(inside_x, yy <= top + 3, yy <= corner_line + 2)
    crease = (
        ink & inside_x & (yy < top - 9) & (luminance(rgb) > (light + dark) / 2) & ~white
    )
    lash = ink & upper & ~crease & ~white
    low = ink & ~upper & ~white
    m8 = lambda m: m.astype(np.uint8) * 255
    ball = cv2.dilate(core, np.ones((3, 3), np.uint8))
    return (
        ball,
        m8(lash),
        m8(low),
        m8(crease),
        m8(ink & white & (ball == 0)),
        core,
        white,
    )


def infer_ink(rgba, mask, skin):
    """Darkest local stroke colour, without a fixed illustration-specific palette."""
    pixels = rgba[..., :3][(mask > 0) & (rgba[..., 3] > 200)].astype(np.float32)
    if not len(pixels):
        return skin
    dark = luminance(pixels) <= np.percentile(luminance(pixels), 10)
    return np.median(pixels[dark], axis=0)


def ellipse_mask(shape, region):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((xx - region["cx"]) / region["rx"]) ** 2 + (
        (yy - region["cy"]) / region["ry"]
    ) ** 2 <= 1


def hair_mask(rgba, rig, external=None):
    """Grow sampled hair colours over the head and buns, excluding skin and clothes.

    With `external` (an 8-bit mask of the hair, same size as the source, from --hairmask) that mask replaces
    the colour clustering; it gets the same blur and keeps the eyes, brow, mouth, nose, hand and garments out.
    """
    shape = rgba.shape[:2]
    sample = np.zeros(shape, np.uint8)
    scope = ellipse_mask(shape, rig["head"]).astype(np.uint8) * 255
    blur = 9.0
    for strand in rig.get("strands", []):
        nodes = np.rint(strand["nodes"]).astype(np.int32)
        cv2.polylines(sample, [nodes], False, 255, max(3, round(strand["sigma"])))
        cv2.polylines(scope, [nodes], False, 255, max(3, round(strand["sigma"] * 6)))
        blur = max(blur, strand["max"] * 1.2)
    for bun in rig.get("buns", {}).values():
        scope[ellipse_mask(shape, bun)] = 255
        centre = {**bun, "rx": bun["rx"] * 0.5, "ry": bun["ry"] * 0.5}
        sample[ellipse_mask(shape, centre)] = 255
    scope = cv2.dilate(scope, np.ones((19, 19), np.uint8)) > 0
    excluded = np.zeros(shape, bool)
    for eye in rig["eyes"]:
        excluded |= polygon_mask(shape, eye["roi"]) > 0
    for key in ("brow", "mouth", "nose"):
        excluded |= ellipse_mask(shape, rig["face"][key])
    if rig.get("hand"):
        excluded |= polygon_mask(shape, rig["hand"]["outline"]) > 0
    for cloth in rig.get("cloth", []):
        excluded |= polygon_mask(shape, cloth["outline"]) > 0
    if external is not None:
        mask = external.copy()
        mask[excluded | (rgba[..., 3] == 0)] = 0
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
        mask[excluded | (rgba[..., 3] == 0)] = 0
        return mask
    opaque = rgba[..., 3] > 200
    lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    sampling = (sample > 0) & opaque & ~excluded
    pixels = lab[sampling]
    if not len(pixels):
        return np.zeros(shape, np.uint8)
    # Lab separates brown shadows from similarly dark, neutral clothing. Cluster
    # several shades rather than allowing one broad RGB tolerance to include both.
    pixels = pixels[pixels[:, 0] <= np.percentile(pixels[:, 0], 90)]

    def palette(values, count):
        # Regular subsampling bounds work and keeps the result deterministic.
        values = np.ascontiguousarray(
            values[:: max(1, len(values) // 12000)], np.float32
        )
        cv2.setRNGSeed(0)
        _, _, centres = cv2.kmeans(
            values,
            min(count, len(values)),
            None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1),
            1,
            cv2.KMEANS_PP_CENTERS,
        )
        return centres

    def distance(colours, values):
        result = np.full(values.shape[:-1], np.inf, np.float32)
        for colour in colours:
            result = np.minimum(result, np.linalg.norm(values - colour, axis=-1))
        return result

    hair_colours = palette(pixels, 8)
    hair_distance = distance(hair_colours, lab)
    tolerance = max(
        16, min(35, np.percentile(distance(hair_colours, pixels), 95) * 1.5)
    )
    body = ellipse_mask(shape, rig["body"]["chest"])
    for shoulder in rig["body"]["shoulders"]:
        body |= ellipse_mask(shape, shoulder)
    negative = lab[body & ~scope & opaque]
    skin_colours = []
    for eye in rig["eyes"]:
        colour, _ = skin_samples(rgba, polygon_mask(shape, eye["roi"]))
        skin_colours.append(cv2.cvtColor(np.uint8([[colour]]), cv2.COLOR_RGB2LAB)[0, 0])
    other_colours = np.array(skin_colours, np.float32)
    if len(negative):
        other_colours = np.vstack([other_colours, palette(negative, 8)])
    other_distance = distance(other_colours, lab)
    not_hair = other_distance < hair_distance
    candidate = (hair_distance < tolerance) & scope & opaque & ~excluded & ~not_hair
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    candidate[excluded | ~scope | ~opaque | not_hair] = 0
    _, labels = cv2.connectedComponents(candidate)
    seeded = np.unique(labels[sampling & (candidate > 0)])
    seeded = seeded[seeded != 0]
    mask = np.isin(labels, seeded).astype(np.uint8) * 255
    # Shadows/highlights enclosed by a hair component belong to the same hair.
    # Fill small colour-classification holes, while keeping skin and bright,
    # differently coloured accessories out of the inferred region.
    count, holes, stats, _ = cv2.connectedComponentsWithStats(
        (mask == 0).astype(np.uint8)
    )
    filled = np.zeros(shape, bool)
    max_hole = max(64, rig["head"]["rx"] * rig["head"]["ry"] * 0.03)
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        if (
            x == 0
            or y == 0
            or x + width == shape[1]
            or y + height == shape[0]
            or area > max_hole
        ):
            continue
        filled |= (holes == i) & (hair_distance < tolerance * 2) & ~excluded & opaque
    mask[filled] = 255
    mask = cv2.GaussianBlur(mask, (0, 0), blur)
    mask[
        excluded
        | ~scope
        | (not_hair & ~filled)
        | (hair_distance >= tolerance * 2)
        | (rgba[..., 3] == 0)
    ] = 0
    return mask


def hand_prefill(rgba, remove, hand):
    """Continue the visible jaw through the hand, using only the rig's landmarks."""
    prefill = rgba.copy()
    yy, xx = np.mgrid[: rgba.shape[0], : rgba.shape[1]]
    jx, jy = np.array(hand["jaw"], float).T
    coef = np.polyfit(jx, jy, min(4, len(jx) - 1))
    curve = np.polyval(coef, xx)
    slope = np.polyval(np.polyder(coef), xx)
    dist = (yy - curve) / np.sqrt(1 + slope**2)
    in_x = (xx >= hand["jawRange"][0]) & (xx <= hand["jawRange"][1])
    hole = remove > 0
    bg_region = polygon_mask(rgba.shape[:2], hand["background"]) > 0
    face = hole & in_x & (dist < 0) & (yy > min(p[1] for p in hand["background"]))
    bg = hole & bg_region & (dist >= 0)
    filled = fill_skin(rgba, face.astype(np.uint8) * 255)
    prefill[face] = filled[face]
    prefill[bg] = (0, 0, 0, 0)
    colours, widths = [], []
    for x, y in hand["jaw"][:4] + hand["jaw"][-4:]:
        ix, iy = int(x), int(y)
        col = rgba[max(iy - 3, 0) : min(iy + 4, rgba.shape[0]), ix, :3].astype(
            np.float32
        )
        lum = luminance(col)
        colours.append(col[np.argmin(lum)])
        widths.append(max(1, np.count_nonzero(lum < (lum.min() + lum.max()) / 2)))
    line_col = np.median(colours, axis=0)
    half_width = np.median(widths) / 2
    lw = (np.clip(half_width + 0.5 - np.abs(dist), 0, 1) * (hole & in_x))[..., None]
    prefill[..., :3] = (prefill[..., :3] * (1 - lw) + line_col * lw).astype(np.uint8)
    prefill[..., 3] = np.maximum(prefill[..., 3], (lw[..., 0] * 255).astype(np.uint8))
    painted = (face | bg | (lw[..., 0] > 0)).astype(np.uint8) * 255 & remove
    return prefill, painted


def has_neckline(cloth):
    return cloth.get("material") == "knit" or (cloth.get("material") == "woven" and bool(cloth.get("neckline")))


def build_cloth(project, rig, rgba, base, store, images, cloth_under="auto", legs=None, covered=None, notes=None, occluders=None):
    """Cut each garment into its own layer and write base_cloth.png without them.

    base.png keeps the garments, so runtimes that do not know rig.cloth still draw them.
    A garment is its outline. With source_under.png (the same drawing without the
    garments) it also takes the edge pixels just outside the approximate outline that
    are background in the under drawing, or differ strongly from it; base_cloth takes
    the under drawing wherever the two differ. Without it, the outline area is inpainted.
    With cloth_under="keep", base_cloth is the base with the garments kept: nothing is
    drawn or inpainted under them and source_under.png is never read.
    With cloth_under="hem", the same, except each front garment's hem band (tools/cloth/hem.py): there base_cloth is
    transparent or continues the legs below the hem (`legs`), and the garment layer takes its drawn edge along the band,
    cut hard. `covered` (hand, accessories) stays out of the band; where the band runs along `occluders` (accessories cut
    by an outline, which the garment goes on behind) it holds the garment's own colours, smoothly, instead of what is below
    the hem. `notes` receives what each band holds.
    """
    h, w = rgba.shape[:2]
    shape = (h, w)
    under = None
    under_path = project / "source_under.png"
    if cloth_under not in ("keep", "hem") and under_path.exists():
        with Image.open(under_path) as image:
            under = np.array(image.convert("RGBA"))
        if under.shape != rgba.shape:
            raise ValueError("source_under.png must match source.png dimensions")
        under[..., 3] = np.where(under[..., 3] > 240, 255, under[..., 3])
        diff = np.abs(rgba.astype(np.int16) - under.astype(np.int16)).max(2)
        # small differences over the body are redrawing noise, not garment
        changed = (
            (diff > 24) & (rgba[..., 3] > 0) & ((under[..., 3] < 128) | (diff > 90))
        ).astype(np.uint8) * 255
    foreground_limbs = np.zeros(shape, bool)
    if rig.get("limbs") and any(c.get("material") for c in rig["cloth"]):
        for limb, pixels in zip(rig["limbs"], limb_layers().masks(rgba, rig)):
            if not limb.get("behind"):
                foreground_limbs |= pixels
    masks = {}
    for cloth in rig["cloth"]:
        outline = polygon_mask(shape, cloth["outline"])
        if under is not None:
            # the drawn outline is approximate: reach a little past it for garment edges
            outline |= changed & cv2.dilate(outline, np.ones((31, 31), np.uint8))
        if cloth.get("material"):
            outline[foreground_limbs] = 0
        masks[cloth["name"]] = outline
    bodice_front = np.zeros(shape, np.uint8)
    for cloth in rig["cloth"]:
        if has_neckline(cloth):
            bodice_front |= masks[cloth["name"]]
    for cloth in rig["cloth"]:
        if cloth.get("material") == "woven" and not cloth.get("neckline"):
            # The bound bodice is the visible front between the lapels. A coarse lapel
            # outline or its hem ring must not carry a duplicate strip of the bodice.
            masks[cloth["name"]] &= ~bodice_front
    front = np.zeros(shape, np.uint8)
    for cloth in rig["cloth"]:
        if not cloth.get("behind"):
            front |= masks[cloth["name"]]
    taken = np.zeros(shape, np.uint8)
    bands = []
    material_edges = np.zeros(shape, bool)
    soft_alpha = (rgba[..., 3] > 0) & (rgba[..., 3] < 255)
    # Material surfaces are picked in draw order: the last foreground garment owns the visible pixels.
    # Legacy rigs use their original first-garment overlap convention.
    cut_order = reversed(rig["cloth"]) if any(c.get("material") for c in rig["cloth"]) else rig["cloth"]
    for cloth in cut_order:
        mask = masks[cloth["name"]] & ~taken
        if cloth.get("behind"):
            mask &= ~front
        if has_neckline(cloth):
            # Shoulder seams and the waistband are attached. Only the neckline opens,
            # and the approved innerwear is already aligned to that opening.
            if covered is not None:
                mask &= ~covered
            taken |= mask
            material_edges |= (mask > 0) & soft_alpha
            store(f"cloth_{cloth['name']}", *crop_layer(rgba, mask, soft=False))
            continue
        if cloth_under == "hem" and not cloth.get("behind"):
            others = np.zeros(shape, np.uint8)
            for other in rig["cloth"]:
                if not cloth.get("material") and other["name"] != cloth["name"]:
                    others |= masks[other["name"]]
            keep_out = (others | taken | (covered if covered is not None else 0)) > 0
            if cloth.get("material"):
                keep_out |= foreground_limbs
            if cloth.get("material") == "woven":
                keep_out |= bodice_front > 0
            region, band, fill, report = hem_band().band(rgba, mask, cloth, legs, keep_out)
            if occluders is not None:
                fill = behind_front(rgba, region, band, fill, occluders)
            mask = region.astype(np.uint8) * 255
            if cloth.get("material"):
                material_edges |= region & soft_alpha
            bands.append((band, fill))
            if notes is not None:
                notes[cloth["name"]] = report
            taken |= mask
            store(f"cloth_{cloth['name']}", *crop_layer(rgba, mask, soft=False))
            continue
        taken |= mask
        store(f"cloth_{cloth['name']}", *crop_layer(rgba, mask))
    if cloth_under in ("keep", "hem"):
        bare = base.copy()
        for band, fill in bands:
            bare[band] = fill[band]
        # A translucent silhouette belongs to the sprite once. Keeping its alpha in both layers
        # makes a 50% stocking edge 75% opaque at rest. Historical garments retain their old bake.
        bare[material_edges] = 0
    elif under is not None:
        bare = base.copy()
        replace = (diff > 0) | (cv2.dilate(taken, np.ones((3, 3), np.uint8)) > 0)
        bare[replace] = under[replace]
    else:
        print("cloth: no source_under.png; filling the garment area by inpainting")
        bare = inpaint_rgba(base, cv2.dilate(taken, np.ones((7, 7), np.uint8)))
    images["base_cloth"] = bare


def matte_collar(sprite, source, patch, coverage, delta, margin, bare):
    """Unmix the antialiased lip from the original skin so the edge stays cloth-coloured when pulled."""
    background = patch[..., :3].astype(np.float32)
    original = source[..., :3].astype(np.float32)
    difference = original - background
    band = coverage & (np.abs(delta) <= margin)
    distance = np.where(band, np.sum(difference * difference, axis=2), -1)
    ink = np.argmax(distance, axis=0)
    ref = difference[ink, np.arange(source.shape[1])]
    alpha = np.sum(difference * ref[None], axis=2) / np.maximum(np.sum(ref * ref, axis=1)[None], 1)
    # A minimum alpha for each channel keeps the straight foreground RGB in gamut.
    minimum = np.maximum(-difference / np.maximum(background, 1),
                         difference / np.maximum(255 - background, 1)).max(axis=2)
    alpha = np.minimum(255, np.ceil(255 * np.maximum(minimum, alpha))).clip(0, 255).astype(np.uint8)
    soft = band & (np.arange(source.shape[0])[:, None] < ink[None]) & (alpha < 255)
    if soft.any():
        foreground = background[soft] + difference[soft] * (255 / np.maximum(1., alpha[soft, None]))
        sprite[..., :3][soft] = np.clip(np.rint(foreground), 0, 255).astype(np.uint8)
        sprite[..., 3][soft] = alpha[soft]
        bare[soft] = patch[soft]


def build_garment_underlayers(folder, entries, rig, meta, images, store, rgba):
    """Bake approved garment crops, with separate source-matched legs beneath movable skirts."""
    indexed = {item["name"]: item for item in entries if item.get("sha256") or item.get("legReveal")}
    drawn = images.get("base_bare", images.get("base_cloth", images["base"]))
    result = None
    layers = {}
    for cloth in rig.get("cloth", []):
        name = cloth["name"]
        entry = indexed.get(name)
        if not entry:
            continue
        key = f"cloth_{name}"
        x, y, w, h = meta["layers"][key]
        patch = None
        if entry.get("sha256"):
            path = folder / f"{name}.png"
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"garment patch {name}: bytes differ from the approved image")
            with Image.open(path) as image:
                if image.size != (drawn.shape[1], drawn.shape[0]):
                    raise ValueError(f"garment patch {name}: expected source dimensions")
                patch = np.array(image.convert("RGBA").crop((x, y, x + w, y + h)))
        if entry.get("legReveal"):
            if entry["kind"] == "underwear" and patch is None:
                continue
            spec = importlib.util.spec_from_file_location("mesh_avatar_skirt_legs", SKIRT_LEGS)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            bones = entry.get("legs") or rig.get("legs")
            original = images[key][..., 3] > 0
            # The fitted polygon can stop above the painted hem. Include the adjoining
            # stocking/skin rows so their transition can be traced in the original picture.
            extra = min(rgba.shape[0] - y - h, max(6, round(w * .29)))
            original = np.pad(original, ((0, extra), (0, 0)))
            if patch is not None:
                patch = np.pad(patch, ((0, extra), (0, 0), (0, 0)))
            h += extra
            fit = module.recover(rgba, (x, y, w, h), original, bones) if bones else None
            if not fit:
                print(f"garment {name}: two visible leg silhouettes could not be traced", file=sys.stderr)
                continue
            outer, clear, surfaces, contact = fit
            source = rgba[y:y + h, x:x + w]
            skirt_height = min(h, int(np.flatnonzero(outer.any(1))[-1]) + 5)
            images[key] = np.where(outer[:skirt_height, :, None], source[:skirt_height], 0)
            meta["layers"][key] = [x, y, w, skirt_height]
            drawn[y:y + h, x:x + w][original & ~outer] = source[original & ~outer]
            if result is None:
                result = drawn.copy()
            layer = f"under_{name}"
            # Store the erase mask independently of any resulting coverage. The gap between
            # the thighs and the air beside a flared skirt must reveal the stage.
            erase = np.zeros((h, w, 4), np.uint8)
            erase[..., 3] = clear * 255

            def store_local(key, pixels):
                sx, sy, sw, sh = cv2.boundingRect(pixels[..., 3])
                rect = [x + sx, y + sy, sw, sh]
                store(key, pixels[sy:sy + sh, sx:sx + sw], rect)
                return rect

            restored = result[y:y + h, x:x + w]
            restored[original & ~outer] = source[original & ~outer]
            restored[clear] = 0
            data = {"kind": entry["kind"], "source": "visible_legs", "layer": layer,
                    "clear": layer + "_clear", "legs": {}}
            store_local(data["clear"], erase)
            back = module.coat_back(rgba, (x, y, w, h), outer, clear, surfaces, bones)
            if back is not None:
                data["back"] = layer + "_back"
                store_local(data["back"], back)
                visible = back[..., 3] > 0
                restored[visible] = back[visible]
            # The exposed fitted stocking is its own grabbable mesh. Give its overlapping
            # pixels the same local lighting as the hidden thigh; the contact sprite carries
            # the original hem shadow once, above the stockings.
            for stocking in (part for part in rig["cloth"] if part.get("material") == "stocking"):
                sock_key = f"cloth_{stocking['name']}"
                px, py, pw, ph = meta["layers"][sock_key]
                x0, y0 = max(x, px), max(y, py)
                x1, y1 = min(x + w, px + pw), min(y + h, py + ph)
                if x0 >= x1 or y0 >= y1:
                    continue
                overlap = np.s_[y0 - y:y1 - y, x0 - x:x1 - x]
                fitted = images[sock_key].copy()
                sock = fitted[y0 - py:y1 - py, x0 - px:x1 - px]
                painted = False
                for pixels in surfaces.values():
                    thigh = pixels[overlap]
                    changed = (contact[overlap][..., 3] > 0) & (thigh[..., 3] == 255) & (sock[..., 3] > 0)
                    if changed.any():
                        sock[..., :3][changed] = thigh[..., :3][changed]
                        painted = True
                if painted:
                    variant = f"{layer}_{stocking['name']}"
                    store(variant, fitted, [px, py, pw, ph])
                    data.setdefault("stockings", {})[stocking["name"]] = variant
            if contact[..., 3].any():
                data["contact"] = layer + "_contact"
                store_local(data["contact"], contact)
            preview = Image.new("RGBA", (w, h))
            for side, pixels in surfaces.items():
                sprite = f"{layer}_leg_{side}"
                store_local(sprite, pixels)
                data["legs"][side] = sprite
                preview.alpha_composite(Image.fromarray(pixels))
            if patch is not None:
                patch[~(clear & (source[..., 3] == 255))] = 0
                if patch[..., 3].any():
                    data["shorts"] = layer + "_shorts"
                    store_local(data["shorts"], patch)
                    preview.alpha_composite(Image.fromarray(patch))
            data["rect"] = store_local(layer, np.asarray(preview))
            layers[name] = data
            continue
        neckline = cloth.get("neckline") if has_neckline(cloth) else None
        if neckline:
            # The drawn collar includes an antialiased ink line on the skin side.
            # Carry that line with the moving collar and continue the wearer's adjacent
            # exposed skin across its old position, right up to the approved patch.
            rim = np.asarray(neckline)
            cols = np.arange(x, x + w)
            rim_y = np.interp(cols, rim[:, 0], rim[:, 1])
            margin = max(2, round(.014 * np.ptp(rim[:, 0])))
            delta = np.arange(y, y + h)[:, None] - rim_y
            seam = ((cols >= rim[0, 0]) & (cols <= rim[-1, 0])
                    & (delta >= -margin) & (delta <= margin) & (images[key][..., 3] == 255))
            sample = np.stack([rgba[np.clip(np.rint(rim_y - d).astype(int), 0, rgba.shape[0] - 1), cols]
                               for d in range(margin, margin + 3)])
            skin = cv2.GaussianBlur(np.median(sample, axis=0).astype(np.float32)[None], (5, 1), 0)[0]
            weight = np.clip(delta / margin, 0, 1) * (patch[..., 3] / 255)
            colour = skin[None, :, :3] * (1 - weight[..., None]) + patch[..., :3] * weight[..., None]
            patch[..., :3][seam] = np.rint(colour[seam]).astype(np.uint8)
            patch[..., 3][seam] = 255
        # Preserve the original anti-aliased outer edge at rest and exclude holes cut for foreground objects.
        coverage = (patch[..., 3] == 255) & (images[f"cloth_{name}"][..., 3] == 255)
        if not coverage.any():
            continue
        patch[~coverage] = 0
        if result is None:
            result = drawn.copy()
        result[y:y + h, x:x + w][coverage] = patch[coverage]
        if neckline:
            matte_collar(images[key], rgba[y:y + h, x:x + w], patch, coverage,
                         delta, margin, drawn[y:y + h, x:x + w])
        sx, sy, sw, sh = cv2.boundingRect(coverage.astype(np.uint8))
        layer = f"under_{name}"
        rect = [x + sx, y + sy, sw, sh]
        store(layer, patch[sy:sy + sh, sx:sx + sw], rect)
        layers[name] = {"kind": entry["kind"], "layer": layer, "rect": rect}
    if result is not None:
        store("base_garment_inner", result, [0, 0, drawn.shape[1], drawn.shape[0]])
        meta.update(garmentBase="base_garment_inner", garmentUnderlayers=layers)


def validate_inputs(rig, rgba):
    h, w = rgba.shape[:2]
    if rig.get("version") != 1 or rig.get("image") != {"width": w, "height": h}:
        raise ValueError(
            "rig.image must match source.png dimensions, with rig.version = 1"
        )
    if len(rig.get("eyes", [])) != 2:
        raise ValueError("rig.eyes must contain two eyes")

    def points(value, path, minimum=3):
        p = np.asarray(value, dtype=float)
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or len(p) < minimum
            or not np.isfinite(p).all()
        ):
            raise ValueError(
                f"{path}: expected at least {minimum} finite [x, y] points"
            )
        if (
            (p[:, 0] < 0).any()
            or (p[:, 0] > w).any()
            or (p[:, 1] < 0).any()
            or (p[:, 1] > h).any()
        ):
            raise ValueError(f"{path}: points must be within source.png")
        if minimum >= 3 and cv2.contourArea(p.astype(np.float32)) <= 0:
            raise ValueError(f"{path}: polygon must have positive area")

    for i, eye in enumerate(rig["eyes"]):
        for key in ("opening", "roi"):
            points(eye[key], f"eyes[{i}].{key}")
        core = polygon_mask((h, w), eye["opening"])
        region = polygon_mask((h, w), eye["roi"])
        if not np.any((core > 0) & (region > 0)):
            raise ValueError(f"eyes[{i}].roi must overlap opening")
        if not np.any((core > 0) & (rgba[..., 3] > 0)):
            raise ValueError(f"eyes[{i}].opening has no source pixels")
        eye_curves(eye)
        if eye.get("iris") is not None:
            iris = eye["iris"]
            if (
                not isinstance(iris, dict)
                or not all(isinstance(iris.get(k), (int, float)) and math.isfinite(iris[k]) for k in ("cx", "cy", "rx", "ry"))
                or min(iris["rx"], iris["ry"]) <= 0
            ):
                raise ValueError(f"eyes[{i}].iris: expected a finite ellipse with positive radii")
            if not (0 <= iris["cx"] <= w and 0 <= iris["cy"] <= h):
                raise ValueError(f"eyes[{i}].iris: the centre must be within source.png")
    for key in ("brow", "mouth", "nose"):
        region = rig["face"][key]
        if (
            not all(np.isfinite(region[k]) for k in ("cx", "cy", "rx", "ry"))
            or min(region["rx"], region["ry"]) <= 0
        ):
            raise ValueError(
                f"face.{key}: expected a finite ellipse with positive radii"
            )
    for i, strand in enumerate(rig.get("strands", [])):
        points(strand["nodes"], f"strands[{i}].nodes", 2)
        if any(not np.isfinite(strand[k]) or strand[k] <= 0 for k in ("sigma", "max")):
            raise ValueError(f"strands[{i}]: sigma and max must be positive")
    if rig.get("hand") is not None:
        hand = rig["hand"]
        points(hand["outline"], "hand.outline")
        points(hand["background"], "hand.background")
        points(hand["jaw"], "hand.jaw", 2)
        if len({p[0] for p in hand["jaw"]}) < 2:
            raise ValueError("hand.jaw needs different x coordinates")
        if (
            len(hand["jawRange"]) != 2
            or not 0 <= hand["jawRange"][0] < hand["jawRange"][1] <= w
        ):
            raise ValueError(
                "hand.jawRange must be an increasing range inside the image"
            )
    names = {
        "base",
        "hairmask",
        "hand",
        *(
            f"eye{i}_{part}"
            for i in range(2)
            for part in ("ball", "lash", "low", "crease", "white", "iris")
        ),
    }
    for i, accessory in enumerate(rig.get("accessories", [])):
        name = accessory["name"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
            or name in names
        ):
            raise ValueError(
                f"accessories[{i}].name must be a unique, safe layer filename"
            )
        names.add(name)
        box = accessory["box"]
        if len(box) != 4 or any(
            not isinstance(n, (int, float)) or not math.isfinite(n) for n in box
        ):
            raise ValueError(f"accessories[{i}].box must contain four finite coordinates")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"accessories[{i}].box must have positive width and height")
        rounded = [
            max(0, min(w, math.floor(box[0]))),
            max(0, min(h, math.floor(box[1]))),
            max(0, min(w, math.ceil(box[2]))),
            max(0, min(h, math.ceil(box[3]))),
        ]
        if rounded[0] >= rounded[2] or rounded[1] >= rounded[3]:
            raise ValueError(f"accessories[{i}].box has no area inside the image")
        if rounded != box:
            print(f"accessories[{i}].box rounded outward and clipped: {box} -> {rounded}")
        accessory["box"] = rounded
        colour = accessory["color"]
        if not 0 <= colour["redness"] <= 1 or not 0 <= colour["minRed"] <= 255:
            raise ValueError(f"accessories[{i}].color thresholds are out of range")
        outline = accessory.get("outline")
        if outline is not None and (
            not isinstance(outline, list)
            or len(outline) < 3
            or any(
                not isinstance(p, (list, tuple)) or len(p) != 2
                or any(not isinstance(n, (int, float)) or not math.isfinite(n) for n in p)
                for p in outline
            )
        ):
            raise ValueError(f"accessories[{i}].outline must be a polygon of at least three finite points")
    cloth_names = set()
    for i, cloth in enumerate(rig.get("cloth", [])):
        name = cloth["name"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
            or name in cloth_names
            or f"cloth_{name}" in names
        ):
            raise ValueError(f"cloth[{i}].name must be a unique, safe layer filename")
        cloth_names.add(name)
        names.add(f"cloth_{name}")
        points(cloth["outline"], f"cloth[{i}].outline")
        points(cloth["waist"], f"cloth[{i}].waist", 2)
        points(cloth["hem"], f"cloth[{i}].hem", 2)
    if rig.get("limbs"):
        limb_layers().validate(rig, (h, w), names)


def ball_inside(shape, curves, margin):
    """Pixels the renderer draws fully from the eye ball layer at rest: between its top and bottom lid curves
    (sampled like the shader's crv(), at pixel centres) with `margin` px to spare beyond the clip's soft edge."""
    h, w = shape
    xs = np.arange(w) + 0.5
    u = (xs - curves["x0"]) / max(curves["x1"] - curves["x0"], 1e-6)
    samples = np.linspace(0, 1, EYE_SAMPLES)
    top = np.interp(u, samples, curves["top"])
    bot = np.interp(u, samples, curves["bot"])
    ys = np.arange(h)[:, None] + 0.5
    inside = (ys >= top + 0.5 + margin) & (ys <= bot - 0.9 - margin)
    inside &= ((u >= 0) & (u <= 1))[None, :]
    return inside


def open_eye_layer(rgba, eye, covered, exact=False, matte=None):
    """The source pixels around an open eye: everything the cut-out eye layers redraw (lines, lids, the eye
    hole filled with skin), opaque over the eye region and a margin, fading out beyond it, with a hole where
    the eye ball layer alone shows the drawing. `covered` (hand, accessories) stays out, so those layers leave
    no copy behind when they move. `exact` (an eye with iris layers): the hole is every pixel the eye layers
    draw fully, with nothing to spare, because a rim of the picture kept around it would hold a strip of the
    iris in place while the iris moves. `matte` (from tools/eye/iris.py): inside that hole, along the lids, the
    lid's share of the pixels it blends into: its colour at that opacity, so the lid's soft edge stays in this layer.
    """
    h, w = rgba.shape[:2]
    region = polygon_mask((h, w), eye["roi"])
    opening = polygon_mask((h, w), eye["opening"])
    grow = 2 * OPEN_EYE_MARGIN + 1
    area = cv2.dilate(region, np.ones((grow, grow), np.uint8))
    distance = cv2.distanceTransform((area == 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(1 - distance / (OPEN_EYE_FEATHER + 1), 0, 1)
    shrink = 2 * OPEN_EYE_HOLE + 1
    hole = (cv2.erode(opening, np.ones((shrink, shrink), np.uint8)) > 0) & ball_inside((h, w), eye_curves(eye), 1)
    if exact:
        hole = ball_inside((h, w), eye_curves(eye), 0)
    alpha[hole] = 0
    alpha[covered > 0] = 0
    layer = rgba.copy()
    layer[..., 3] = np.rint(rgba[..., 3].astype(np.float32) * alpha).astype(np.uint8)
    if matte is not None:
        x, y = matte["origin"]
        mh, mw = matte["alpha"].shape
        part = layer[y : y + mh, x : x + mw]
        on = (matte["alpha"] > 0) & hole[y : y + mh, x : x + mw] & (covered[y : y + mh, x : x + mw] == 0)
        part[on, :3] = matte["colour"][on]
        part[on, 3] = np.rint(matte["alpha"][on] * 255).astype(np.uint8)
    return crop_layer(layer, (layer[..., 3] > 0).astype(np.uint8) * 255, pad=1, soft=False)


def read_hairmask(path, size, option="--hairmask"):
    """An external mask (hair, legs): grayscale (or RGB) PNG, white where it is, exactly the source size."""
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(
                f"{option} {path.name} is {image.size[0]} x {image.size[1]}, "
                f"source.png is {size[0]} x {size[1]}"
            )
        return np.array(image.convert("L"))


def read_plate(path, size):
    """A clean plate: RGB or RGBA PNG, exactly the source size (without alpha it is defined everywhere)."""
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"head plate: {image.size[0]}x{image.size[1]}, expected {size[0]}x{size[1]} like source.png")
        return np.array(image.convert("RGBA"))


def build(project, rig_path, hairmask=None, cloth_under="auto", open_eyes=False, head=False, plate=None, footprint=None,
          hem_legs=None, cheeks=False, strand_layers_on=False, garment_patches=None, strand_garment=None):
    source = project / "source.png"
    rig_bytes = rig_path.read_bytes()
    rig = json.loads(rig_bytes)
    with Image.open(source) as image:
        rgba = np.array(image.convert("RGBA"))
    validate_inputs(rig, rgba)
    rgba[..., 3] = np.where(rgba[..., 3] > 240, 255, rgba[..., 3])
    h, w = rgba.shape[:2]
    external_hair = read_hairmask(hairmask, (w, h)) if hairmask is not None else None
    # Options take part in the cache-busting hash only when given, so a plain build keeps its id.
    options = b""
    if hairmask is not None:
        options += b"hairmask:" + hairmask.read_bytes()
    if cloth_under != "auto":
        options += f"cloth-under:{cloth_under}".encode()
    if cloth_under == "hem":
        options += b"hem:" + HEM_BAND.read_bytes() + (b"legs:" + Path(hem_legs).read_bytes() if hem_legs is not None else b"")
    if open_eyes:
        options += b"open-eyes"
    if head:
        options += b"head-layers:" + HEAD_LAYERS.read_bytes()
    if plate is not None:
        options += b"head-plate:" + HEAD_LAYERS.read_bytes() + HEAD_PLATE.read_bytes() + Path(plate).read_bytes() + Path(footprint).read_bytes()
    if cheeks:
        options += b"cheek-pull:" + CHEEK_PULL.read_bytes()
    if strand_layers_on:
        options += b"strand-layers:" + STRAND_LAYERS.read_bytes()
        if strand_garment is not None:
            options += b"strand-garment:" + Path(strand_garment).read_bytes()
    if rig.get("limbs"):  # only rigs with limbs depend on the limb cutter
        options += b"limbs:" + LIMB_LAYERS.read_bytes() + b"behind:" + BEHIND.read_bytes()  # the body under them
    if any(accessory.get("outline") is not None for accessory in rig.get("accessories", [])):
        options += b"behind:" + BEHIND.read_bytes()  # what lies behind an accessory cut by an outline
    if any(eye.get("iris") is not None for eye in rig["eyes"]):  # only eyes with an iris depend on the iris cutter
        options += b"iris:" + IRIS_LAYERS.read_bytes()
    entries = []
    if garment_patches is not None:
        manifest = (garment_patches / "manifest.json").read_bytes()
        entries = json.loads(manifest)
        options += b"garment-patches:" + manifest + (SKIRT_LEGS.read_bytes() if any(e.get("legReveal") for e in entries) else b"")
    build_id = hashlib.sha256(
        source.read_bytes() + rig_bytes + Path(__file__).read_bytes() + options
    ).hexdigest()[:12]
    meta = {"size": [w, h], "build": build_id, "layers": {}}
    images = {}

    def store(name, image, rect):
        images[name] = image
        meta["layers"][name] = rect

    remove = np.zeros((h, w), np.uint8)
    accessory_bg = np.zeros((h, w), np.uint8)
    outlined = np.zeros((h, w), np.uint8)  # accessories cut by an outline: what is behind them is filled in
    hand = rig.get("hand")
    if hand is not None:
        mask = polygon_mask((h, w), hand["outline"])
        store("hand", *crop_layer(rgba, mask))
        remove |= cv2.dilate(mask, np.ones((5, 5), np.uint8))
        tips = cv2.dilate(mask, np.ones((11, 11), np.uint8))
        cutoff = int(max(p[1] for p in hand["jaw"]) + 20)
        tips[max(cutoff, 0) :] = 0
        remove |= tips
    for accessory in rig.get("accessories", []):
        mask = accessory_mask(rgba, accessory)
        if not np.any(mask):
            raise ValueError(
                f"accessory {accessory['name']}: no pixels match its box and color thresholds"
            )
        # cut by an outline: a hard edge, so the accessory and the garment it lies over each draw their own pixels at rest
        store(accessory["name"], *crop_layer(rgba, mask, soft=accessory.get("outline") is None))
        if accessory.get("outline") is not None:
            # exact: the layer has a hard edge, so the base round it stays the drawing
            accessory_bg |= mask
            outlined |= mask
        else:
            accessory_bg |= cv2.dilate(mask, np.ones((5, 5), np.uint8))

    # Head layers from a clean plate: the hair in front of the face is lifted off, and the face without it (`face`) is
    # what the eye layers are cut from. base.png stays the whole picture.
    head_note = None
    lifted = None
    face = rgba
    if plate is not None:
        outside = (remove | accessory_bg) > 0
        for limb_mask in limb_layers().masks(rgba, rig) if rig.get("limbs") else []:
            outside |= limb_mask
        for cloth in rig.get("cloth", []):
            outside |= polygon_mask((h, w), cloth["outline"]) > 0
        lifted, head_note = head_plate().cut(
            rgba, rig, read_plate(plate, (w, h)), read_hairmask(footprint, (w, h)),
            head_layers().face_frame, head_layers().field_of, head_layers().MOVING, outside,
        )
        if lifted:
            face = lifted["under"]

    eye_hole = np.zeros((h, w), np.uint8)
    eye_parts = []
    for i, eye in enumerate(rig["eyes"]):
        ball, lash, low, crease, stray, core, white = eye_classes(face, eye)
        taken = cv2.erode(core, np.ones((3, 3), np.uint8))
        lines, inks = {}, {}
        region = polygon_mask((h, w), eye["roi"])
        skin, _ = skin_samples(face, region)
        for part, mask in (("lash", lash), ("low", low), ("crease", crease)):
            inks[part] = infer_ink(face, mask, skin)
            grown = cv2.dilate(mask, np.ones((5, 5), np.uint8)) & ~taken
            lines[part] = grown & cv2.dilate(region, np.ones((5, 5), np.uint8))
            taken |= lines[part]
        eye_parts.append((i, ball, lines, inks))
        near = cv2.dilate(core, np.ones((15, 15), np.uint8)) & ~core
        eye_hole |= taken | core | stray | (white.astype(np.uint8) * 255 & near)
        eye.update(eye_curves(eye))
    skin_under = fill_skin(face, eye_hole)[..., :3]
    # eyes with an iris outline: the eye white without the iris, and the iris as a sprite (tools/eye/iris.py)
    iris_eyes, iris_mattes, iris_drawn, iris_notes = {}, {}, {}, []
    for i, eye in enumerate(rig["eyes"]):
        if eye.get("iris") is None:
            continue
        drawn = ball_inside((h, w), eye, 0)
        pair, reason = iris_layers().cut(
            face, eye, polygon_mask((h, w), eye["opening"]),
            lambda shape, origin: drawn[origin[1] : origin[1] + shape[0], origin[0] : origin[0] + shape[1]],
        )
        if pair is None:
            iris_notes.append(f"eye {i}: {reason}")
            continue
        iris_eyes[i], iris_mattes[i], iris_drawn[i] = pair, pair["matte"], drawn
    for i, ball, lines, inks in eye_parts:
        store(f"eye{i}_ball", *crop_layer(face, ball, pad=3, soft=False))
        for part, mask in lines.items():
            full = unblend(face, skin_under, mask, inks[part])
            edge = np.clip(
                cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 0.7) * 1.6, 0, 1
            )
            full[..., 3] = (full[..., 3] * edge).astype(np.uint8)
            if i in iris_drawn:
                # the line layers are unblended against skin, which is right beside the eye and wrong over it: where
                # the eye layers draw fully they stay out (the lids' share of those pixels is in the open-eye layer)
                full[..., 3][iris_drawn[i]] = 0
            store(
                f"eye{i}_{part}",
                *crop_layer(
                    full, (full[..., 3] > 0).astype(np.uint8) * 255, pad=2, soft=False
                ),
            )
    for i, pair in iris_eyes.items():
        store(f"eye{i}_white", *pair["white"])
        store(f"eye{i}_iris", *pair["iris"])
    meta["eyes"] = [
        {k: eye[k] for k in ("x0", "x1", "top", "bot")} for eye in rig["eyes"]
    ]
    meta["version"] = 1
    if iris_eyes:
        meta["iris"] = [iris_eyes[i]["report"] if i in iris_eyes else None for i in range(len(rig["eyes"]))]

    prefill, painted = (
        hand_prefill(rgba, remove, hand) if hand else (rgba, np.zeros_like(remove))
    )
    base = inpaint_rgba(prefill, (remove & ~painted) | accessory_bg | eye_hole)
    base = fill_skin(base, eye_hole)
    if np.any(accessory_bg):
        kept = ((rgba[..., 3] > 128) & (accessory_bg == 0)).astype(np.uint8)
        dist = cv2.distanceTransform(1 - kept, cv2.DIST_L2, 5)
        fade = np.clip(1 - dist / 2.5, 0, 1)
        base[..., 3] = np.where(
            (accessory_bg > 0) & (outlined == 0), (base[..., 3] * fade).astype(np.uint8), base[..., 3]
        )
        if np.any(outlined):
            # the texture behind them comes from the garments traced round them, never a hand or another accessory
            near = cv2.dilate(outlined, np.ones((21, 21), np.uint8)) > 0
            fabrics = {}
            for cloth in rig.get("cloth", []):
                shape = polygon_mask((h, w), cloth["outline"]) > 0
                if (shape & near).any():
                    fabrics[cloth["name"]] = shape & ((remove | accessory_bg) == 0)
            for accessory in rig.get("accessories", []):
                if accessory.get("outline") is not None:
                    mask = accessory_mask(rgba, accessory)
                    rim = cv2.dilate(mask, np.ones((21, 21), np.uint8)) > 0
                    garment = max(fabrics, key=lambda name: np.count_nonzero(rim & fabrics[name]), default=None)
                    source = fabrics[garment] if garment and (rim & fabrics[garment]).any() else None
                    name = accessory["name"]
                    base = fill_behind(rgba, base, mask, source, store, name)
                    if f"accessory_under_{name}" in meta["layers"]:
                        meta.setdefault("accessoryBacks", {})[name] = garment
    images["base"] = base
    images["hairmask"] = hair_mask(rgba, rig, external_hair)
    if rig.get("cloth"):
        legs = read_hairmask(hem_legs, (w, h), "--hem-legs") if hem_legs is not None else None
        hems = {}
        build_cloth(project, rig, rgba, base, store, images, cloth_under, legs, remove | accessory_bg, hems,
                    outlined if np.any(outlined) else None)
        if cloth_under == "hem":
            meta["clothUnder"] = {"mode": "hem", "garments": hems}
    limbs = np.zeros((h, w), np.uint8)
    if rig.get("limbs"):
        # the runtime base is base_cloth with garments, base otherwise; base_bare is that without the limbs
        images["base_bare"] = limb_layers().cut(rgba, images.get("base_cloth", images["base"]), rig, store, behind_fill().fill)
        meta["bare"] = "base_bare"
        for mask in limb_layers().masks(rgba, rig):
            limbs[mask] = 255
    if open_eyes:
        for i, eye in enumerate(rig["eyes"]):
            store(f"eye{i}_open", *open_eye_layer(rgba, eye, remove | accessory_bg | limbs, exact=i in iris_eyes, matte=iris_mattes.get(i)))
    if lifted:
        # face_field, hair_front, and face_skin: the face without the lifted hair where it differs from the base (the
        # plate under the hair, and the eye regions filled like the base's), for runtimes that turn the face under the hair
        crop = head_layers()._crop
        moving = lifted["field"] > head_layers().MOVING
        store("face_field", *crop(np.clip(lifted["field"] * 255, 0, 255).astype(np.uint8), moving))
        store("hair_front", *crop(lifted["hair"], lifted["hair"][..., 3]))
        under_base = fill_skin(face, eye_hole)
        weight = lifted["weight"].copy()
        weight[(remove | accessory_bg | limbs) > 0] = 0
        under_base[..., 3] = np.clip(np.rint(under_base[..., 3].astype(np.float32) * weight), 0, 255).astype(np.uint8)
        store("face_skin", *crop(under_base, under_base[..., 3]))
        if open_eyes:
            for i, eye in enumerate(rig["eyes"]):
                store(f"eye{i}_open_under", *open_eye_layer(face, eye, remove | accessory_bg | limbs, exact=i in iris_eyes, matte=iris_mattes.get(i)))
        meta["head"] = lifted["report"]
    if head and not lifted:
        # the picture the runtime draws as its base: without the limbs, with the garments kept
        drawn = images.get("base_bare", images.get("base_cloth", images["base"]))
        outside = (remove | accessory_bg | limbs) > 0
        for cloth in rig.get("cloth", []):
            outside |= polygon_mask((h, w), cloth["outline"]) > 0
        tools = SimpleNamespace(polygon_mask=polygon_mask, ellipse_mask=ellipse_mask, skin_samples=skin_samples)
        cut, head_note = head_layers().cut(drawn, rig, images["hairmask"], tools, outside, external_hair)
        for name, (image, rect) in (cut or {}).items():
            store(name, image, rect)
    cheek_notes = []
    if cheeks:
        # cheek layers (tools/head/cheeks.py): cut from the face under the hair in front of it, which keeps its own
        # layer; any other hair stops the contour and, lying over a cheek, shortens its travel
        drawn = images.get("base_bare", images.get("base_cloth", images["base"]))
        head_full = None
        if "hair_front" in meta["layers"] and "face_skin" in meta["layers"]:
            head_full = {}
            for name, key in (("hair_front", "hair"), ("face_skin", "skin")):
                x, y, lw, lh = meta["layers"][name]
                head_full[key] = np.zeros((h, w, 4), np.uint8)
                head_full[key][y : y + lh, x : x + lw] = images[name]
        outside = (remove | accessory_bg | limbs) > 0
        for cloth in rig.get("cloth", []):
            outside |= polygon_mask((h, w), cloth["outline"]) > 0
        cut, data, cheek_notes, head_full = cheek_pull().cut(drawn, images["hairmask"], rig, outside, polygon_mask, head_full)
        for name, (image, rect) in cut.items():
            store(name, image, rect)
        if data:
            meta["cheeks"] = data
            if head_full is not None:
                # the contour's own drawn line moved from the hair layer to the skin under it (same picture at rest)
                for name, key in (("hair_front", "hair"), ("face_skin", "skin")):
                    store(name, *head_layers()._crop(head_full[key], head_full[key][..., 3]))

    strand_notes = []
    if strand_layers_on and any(s.get("reach") for s in rig.get("strands", [])):
        drawn = images.get("base_bare", images.get("base_cloth", images["base"]))
        fabric = read_hairmask(strand_garment, (w, h), "--strand-garment") >= 128 if strand_garment is not None else None
        cut, data, strand_notes = strand_layers().cut(drawn, external_hair if external_hair is not None else images["hairmask"], rig, fabric)
        for name, (image, rect) in cut.items():
            store(name, image, rect)
        if data:
            meta["strands"] = data

    if entries:
        build_garment_underlayers(garment_patches, entries, rig, meta, images, store, rgba)

    # Calculate everything before replacing files. Keep sprites and unrelated
    # assets intact; only filenames in this build's manifest are consumed.
    with tempfile.TemporaryDirectory(prefix=".build-layers-", dir=project) as temporary:
        stage = Path(temporary)
        for name, img in images.items():
            Image.fromarray(img).save(stage / f"{name}.png")
        (stage / "layers.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "rig.json").write_text(
            json.dumps(rig, indent=2) + "\n", encoding="utf-8"
        )
        out = project / "built"
        out.mkdir(exist_ok=True)
        for name in images:
            (stage / f"{name}.png").replace(out / f"{name}.png")
        (stage / "layers.json").replace(out / "layers.json")
        (stage / "rig.json").replace(project / "rig.json")
    if head_note:
        print(f"build-layers: head layers left out: {head_note}", file=sys.stderr)
    for note in iris_notes:
        print(f"build-layers: iris layers left out: {note}", file=sys.stderr)
    for note in cheek_notes:
        print(f"build-layers: cheek layers left out: {note}", file=sys.stderr)
    for note in strand_notes:
        print(f"build-layers: strand layers left out: {note}", file=sys.stderr)
    print(
        json.dumps({"build": build_id, "size": [w, h], "layers": list(meta["layers"])})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "project_dir", type=Path, help="Folder containing source.png and a rig"
    )
    parser.add_argument(
        "--rig",
        default="rig.json",
        help="Rig filename relative to project-dir (default: rig.json)",
    )
    parser.add_argument(
        "--hairmask",
        type=Path,
        help="Hair mask PNG (white is hair, same size as source.png) used instead of colour clustering",
    )
    parser.add_argument(
        "--cloth-under",
        choices=("auto", "keep", "hem"),
        default="auto",
        help="keep: base_cloth.png keeps the garments, without inpainting or source_under.png; hem: the same, except a "
        "band along each garment's hem that it uncovers when it swings, transparent or continuing the legs below "
        "(default auto: source_under.png if present, else inpaint)",
    )
    parser.add_argument(
        "--hem-legs",
        type=Path,
        help="with --cloth-under hem: where legs are (PNG, white, source size); under the hem the band continues them",
    )
    parser.add_argument(
        "--open-eyes",
        action="store_true",
        help="also cut eye0_open.png and eye1_open.png, the source pixels around each eye that newer "
        "runtimes show while the eye is fully open",
    )
    parser.add_argument(
        "--head-layers",
        action="store_true",
        help="also cut face_field.png, hair_front.png and face_skin.png, with which newer runtimes turn the face "
        "under the hair (left out, with the reason on stderr, when the picture does not allow it)",
    )
    parser.add_argument(
        "--head-plate",
        type=Path,
        help="the picture without the hair in front of the face (PNG, source size; alpha marks where it is defined): "
        "head layers are lifted off it instead of being told apart by colour; needs --head-footprint",
    )
    parser.add_argument(
        "--head-footprint",
        type=Path,
        help="where the hair missing from --head-plate is (PNG, white is hair, source size)",
    )
    parser.add_argument(
        "--cheek-pull",
        action="store_true",
        help="also cut cheek_left.png and cheek_right.png, the skin of each cheek from the face's contour inwards, with "
        "which newer runtimes pull a cheek past the contour (a side is left out, with the reason on stderr, when its "
        "contour cannot be traced)",
    )
    parser.add_argument(
        "--strand-layers",
        action="store_true",
        help="also cut strand_<i>.png and strand_<i>_under.png for each strand with a reach: its lock, and the base round "
        "it with the lock filled in, with which newer runtimes move a held lock without bending the base (a strand is "
        "left out, with the reason on stderr, when its lock cannot be told)",
    )
    parser.add_argument("--garment-patches", type=Path, help="directory containing the approved, masked garment patches and manifest.json")
    parser.add_argument("--strand-garment", type=Path, help="visible fabric mask for continuing textile grain behind a moving lock")
    args = parser.parse_args()
    if (args.head_plate is None) != (args.head_footprint is None):
        parser.error("--head-plate and --head-footprint go together")
    if args.hem_legs is not None and args.cloth_under != "hem":
        parser.error("--hem-legs goes with --cloth-under hem")
    try:
        build(
            args.project_dir, args.project_dir / args.rig, args.hairmask, args.cloth_under, args.open_eyes, args.head_layers,
            args.head_plate, args.head_footprint, args.hem_legs, args.cheek_pull, args.strand_layers, strand_garment=args.strand_garment, garment_patches=args.garment_patches,
        )
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        print(f"build-layers: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
