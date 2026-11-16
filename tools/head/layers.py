"""Head layers for build-layers.py (--head-layers): a head turn that slides the face under the hair.

Without them a head turn moves the face inside the head on one mesh: the hair outline stays, so the hair on one side is
squeezed and on the other stretched, like a lens passing over the picture. With them the turn only moves the face:

  face_field.png  where a turn slides the picture and how much (grey, 255 = the whole shift): flat over the eyes, nose
                  and mouth, falling to nothing at the face contour, zero outside the drawing
  hair_front.png  the hair over and around the face as its own layer, drawn above the face; it sways and rolls with the
                  head but does not slide with the turn
  face_skin.png   skin continued under that hair: it moves with the face, covers the base's own copy of the hair, and is
                  what shows next to the hair when the face slides

base.png is not changed, so runtimes that do not know these layers draw the avatar as before.

The layers are only cut when the picture allows it; otherwise the reason is returned and nothing is written:
- the hair is told from the skin by colour inside the hair mask's neighbourhood, so both need to be there and distinct;
- hair hanging over an eye cannot be separated from the eye's own layers, and the turn would tear it.

Loaded by build-layers.py from its own folder only when --head-layers is given.
"""

import math

import cv2
import numpy as np

# The face frame comes from the eyes and the mouth: D is the distance between the eye centres, L the distance from the
# eye line to the mouth. The field is an ellipse along the face (it follows a tilted head).
FIELD_CENTRE = 0.225  # centre below the eye line, in L
FIELD_ACROSS = 0.86  # semi-axis across the face, in D
FIELD_ALONG = 1.33  # semi-axis along the face, in L
FIELD_TOP = 0.6  # of 1 - r^2: the field is flat (1) where this much is left, so eyes, nose and mouth move as drawn
MOVING = 1 / 255  # the field below this does not move the picture
HAIR_REACH = 30  # px the hair layer reaches beyond the moving face, onto hair that lies exactly on the base's own
HAIR_FADE = 8  # of which the outermost pixels fade out
PATCH_GROW = 6  # px the skin under the hair reaches beyond the hair's edge (its antialiased rim and cast line)
HAIR_COLOURS = 8
EYE_COVER_LIMIT = 0.08  # share of an eye region covered by hair from which the layers are not cut
MIN_PIXELS = 200  # fewer core hair or skin pixels than this cannot give a colour


def _sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def face_frame(rig):
    """Centre, semi-axes and tilt of the face field, from the rig's eyes and mouth."""
    a, b = rig["face"]["eyeA"], rig["face"]["eyeB"]
    eyes = np.array([(a["cx"] + b["cx"]) / 2, (a["cy"] + b["cy"]) / 2], np.float64)
    mouth = np.array([rig["mouth"]["cx"], rig["mouth"]["cy"]], np.float64)
    along = float(np.linalg.norm(mouth - eyes))
    across = math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])
    if along < 4 or across < 4:
        return None
    down = (mouth - eyes) / along
    centre = eyes + FIELD_CENTRE * along * down
    return {
        "cx": float(centre[0]),
        "cy": float(centre[1]),
        "rx": FIELD_ACROSS * across,
        "ry": FIELD_ALONG * along,
        "angle": math.atan2(down[0], down[1]),  # 0 for an upright face
    }


def field_of(shape, frame, opaque):
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    c, s = math.cos(frame["angle"]), math.sin(frame["angle"])
    dx, dy = xx - np.float32(frame["cx"]), yy - np.float32(frame["cy"])
    u = (dx * c - dy * s) / np.float32(frame["rx"])
    v = (dx * s + dy * c) / np.float32(frame["ry"])
    field = _sstep(0, FIELD_TOP, np.clip(1 - u * u - v * v, 0, 1))
    return (field * opaque).astype(np.float32)


def _palette(values, count):
    values = np.ascontiguousarray(values[:: max(1, len(values) // 12000)], np.float32)
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


def _distance(colours, values):
    result = np.full(values.shape[:-1], np.inf, np.float32)
    for colour in colours:
        result = np.minimum(result, np.linalg.norm(values - colour, axis=-1))
    return result


def _crop(image, alpha, pad=3):
    ys, xs = np.where(alpha > 0)
    h, w = alpha.shape
    x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
    x1, y1 = min(int(xs.max()) + pad + 1, w), min(int(ys.max()) + pad + 1, h)
    return image[y0:y1, x0:x1], [x0, y0, x1 - x0, y1 - y0]


def cut(base, rig, soft_hair, tools, excluded, given_hair=None):
    """The head layers of `base` (RGBA, the built base), or the reason they cannot be cut.

    soft_hair is the built hair mask (blurred, 0..255); tools gives build-layers' polygon_mask, ellipse_mask and
    skin_samples; excluded marks what is neither hair nor face skin whatever its colour (hand, garments).
    given_hair is the hair mask as it was given to build-layers (--hairmask), if any: the built mask leaves the eye
    regions out, so only the given one can say that hair hangs over an eye.
    Returns ({name: (image, rect)}, None) or (None, reason).
    """
    shape = base.shape[:2]
    frame = face_frame(rig)
    if frame is None:
        return None, "the eyes and the mouth are too close together to place the face"
    opaque = base[..., 3] > 200
    field = field_of(shape, frame, (base[..., 3] > 128).astype(np.float32))
    moving = field > MOVING
    if not np.any(moving):
        return None, "the face is outside the drawing"

    lab = cv2.cvtColor(np.ascontiguousarray(base[..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
    core = (soft_hair > 200) & opaque & ~excluded
    if int(core.sum()) < MIN_PIXELS:
        return None, "too little hair in the hair mask to tell its colour"
    core_lab = lab[core]
    hair_colours = _palette(core_lab, HAIR_COLOURS)
    eye_regions = [tools.polygon_mask(shape, eye["roi"]) > 0 for eye in rig["eyes"]]
    skin_colours = []
    for region in eye_regions:
        colour, _ = tools.skin_samples(base, region.astype(np.uint8) * 255)
        skin_colours.append(cv2.cvtColor(np.uint8([[colour]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32))
    d_hair, d_skin = _distance(hair_colours, lab), _distance(np.array(skin_colours), lab)
    tolerance = max(16.0, min(40.0, float(np.percentile(_distance(hair_colours, core_lab), 97)) * 1.6))
    near = cv2.dilate((soft_hair > 24).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    coloured = (d_hair < tolerance) & (d_hair < d_skin) & near & opaque & ~excluded

    # The eyes with their lashes, the nose and the mouth are face whatever their colour.
    features = np.zeros(shape, bool)
    for region in eye_regions:
        features |= cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for key in ("mouth", "nose"):
        features |= tools.ellipse_mask(shape, rig["face"][key])
    # A brow is a stroke of hair colour on skin; hair hanging over the brow line passes through it. Inside the brow
    # region, what hangs together with hair above the region is hair, the rest is the brow and stays on the face.
    brow = tools.ellipse_mask(shape, rig["face"]["brow"])
    hair = coloured & ~features & ~brow
    around = cv2.dilate(brow.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    _, pieces = cv2.connectedComponents((coloured & ~features & around).astype(np.uint8))
    above = hair & around & ~brow & (np.mgrid[: shape[0], : shape[1]][0] < rig["face"]["brow"]["cy"])
    through = np.unique(pieces[above])
    hair |= np.isin(pieces, through[through != 0]) & brow
    # thin strands stay (no opening); specks go: only what hangs together with the core hair is hair
    hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    hair &= ~features
    _, labels = cv2.connectedComponents(hair.astype(np.uint8))
    kept = np.unique(labels[core & hair])
    hair = np.isin(labels, kept[kept != 0])

    # Hair over an eye: it belongs to the eye's own layers, which slide with the face, and would be torn off the rest.
    if given_hair is not None:
        for i, region in enumerate(eye_regions):
            covered = int(((given_hair > 127) & region).sum())
            if covered > EYE_COVER_LIMIT * int(region.sum()):
                return None, f"hair hangs over eye {i} ({covered * 100 // int(region.sum())}% of it)"
    reach = cv2.dilate(hair.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for i, region in enumerate(eye_regions):
        inside = coloured & region
        if not np.any(inside):
            continue
        _, parts = cv2.connectedComponents(inside.astype(np.uint8))
        joined = np.unique(parts[inside & reach])
        covered = int(np.isin(parts, joined[joined != 0]).sum())
        if covered > EYE_COVER_LIMIT * int(region.sum()):
            return None, f"hair hangs over eye {i} ({covered * 100 // int(region.sum())}% of it)"

    front_core = hair & moving
    if not np.any(front_core):
        # nothing over the face: the field alone turns the face inside its contour
        cut_field, rect = _crop(np.clip(field * 255, 0, 255).astype(np.uint8), moving)
        return {"face_field": (cut_field, rect)}, None

    # The hair layer reaches well past the moving face and is fully opaque over everything the skin patch can cover;
    # it fades out over its outermost pixels only, where it lies exactly on the base's own, unmoved hair.
    size = 2 * HAIR_REACH + 1
    support = cv2.dilate(moving.astype(np.uint8), np.ones((size, size), np.uint8))
    front = hair & (support > 0)
    edge = cv2.GaussianBlur(front.astype(np.float32), (0, 0), 0.8)
    fade = 2 * HAIR_FADE + 1
    margin = np.clip(cv2.GaussianBlur(cv2.erode(support, np.ones((fade, fade), np.uint8)).astype(np.float32), (0, 0), 3) * 1.5, 0, 1)
    hair_layer = base.copy()
    hair_layer[..., 3] = np.clip(base[..., 3].astype(np.float32) * edge * margin, 0, 255).astype(np.uint8)

    # Skin under the hair: a normalised blur of skin pixels only, widening until every covered pixel is reached. The
    # hair around the hole must not bleed in, so there is no solve against the hole's border.
    grow = 2 * PATCH_GROW + 1
    hole = cv2.dilate(front_core.astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
    skin = (d_skin < d_hair) & (d_skin < 28) & opaque & ~hole & ~features & ~brow & ~excluded & moving
    if int(skin.sum()) < MIN_PIXELS:
        return None, "too little skin next to the hair to continue it under the hair"
    rgb = base[..., :3].astype(np.float32)
    known = skin.astype(np.float32)
    filled, done = rgb.copy(), skin.copy()
    for sigma in (2, 4, 8, 16, 32, 64, 128):
        num = cv2.GaussianBlur(rgb * known[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(known, (0, 0), sigma)
        reached = ~done & (den > 0.05)
        filled[reached] = (num / np.maximum(den, 1e-4)[..., None])[reached]
        done |= reached
    smooth = cv2.GaussianBlur(filled, (0, 0), 3)
    patch = base.copy()
    patch[..., :3] = np.clip(smooth, 0, 255).astype(np.uint8)
    patch[..., 3] = np.clip(base[..., 3].astype(np.float32) * cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.2), 0, 255).astype(np.uint8)

    layers = {}
    layers["face_field"] = _crop(np.clip(field * 255, 0, 255).astype(np.uint8), moving)
    layers["face_skin"] = _crop(patch, patch[..., 3])
    layers["hair_front"] = _crop(hair_layer, hair_layer[..., 3])
    return layers, None
