"""Cheek layers for build-layers.py (--cheek-pull): cheeks that the runtime can pull past the face's contour.

For each side of the face (left and right in the image) the visible outer contour of the face is traced from below the
eyes to above the chin, and the tool writes
  cheek_left.png / cheek_right.png  the skin of that cheek from the contour inwards, with the contour's drawn line, cut
                  from the face as it is under the hair in front of it (the base with face_skin over it when there are
                  head layers, the base otherwise);
  layers.json "cheeks"  {"left": {"edge": [[x, y], ...], "reach": [px, ...], "travel": px, "free": [...]}, "right": ...}:
                  the contour from top to bottom, how far into the face a pull reaches at each of its points (short of
                  the eyes, the nose, the mouth and the chin), the largest pull of the contour in source pixels and,
                  where hair lies over the contour or, in front of the face, beside it, how much of the pull each point
                  takes (1 elsewhere; left out when 1 everywhere).
The runtime draws a cheek layer only while that cheek is pulled: over the base, under the eyes, the drawn mouths and the
hair in front of the face. It moves the layer's contour only outwards, so the old contour and the hair beside the face
stay covered. base.png is not changed, so runtimes that do not know these layers draw the avatar as before.

With head layers the contour's own drawn line can sit in hair_front (where the plate or the colours took it for hair).
Hair in front of the face stays where it is while the cheek moves under it, so the line would stay behind: the pixels of
hair_front in a narrow band along the contour that are not part of hair reaching beyond that band move to face_skin (the
two over the base are the same picture), and the cheek layer carries them. So do pieces of hair_front further out beside
the contour, as far as a pull reaches, that are residue of the matting (small, or thin and staying there): the pulled
cheek covers them. Other hair in front lying beside the contour (a lock beside it, the dark gap between the jaw and the
hair behind it that a plate's matting lifts) stays in front, and the pull is held short of it: pulled into it, the cheek
would pass under it and come out where it ends in a spike or a notch. Hair in front thinner than THICK px (lock tips lying
across the cheek) does not hide the contour: the face is seen beyond it, and the layer is not cut along it. Where hair
lies over the contour the contour does not move at all; wherever the pull is held back it eases back in like a smoothstep
over a travel of contour.

A side is left out, and the reason returned, when its contour cannot be traced. Without head layers, hair lying over a
cheek would be dragged along with it, so the travel shrinks with the share of the cheek such hair covers.

Loaded by build-layers.py from its own folder only when --cheek-pull is given.
"""

import math

import cv2
import numpy as np

SIDES = ("left", "right")
SPAN_TOP = 0.32  # the contour starts this far below the eye line, in L (eye line to mouth)
SPAN_BOTTOM = 1.5  # and ends here at the latest
NARROW = 0.45  # or where the face is narrower than this share of its widest row (the jaw turning to the chin)
CHIN_CLEAR = 0.15  # in L: the contour ends at least this far above the chin
MIN_SPAN = 0.35  # in L: a shorter contour is left out
TRAVEL = 0.11  # largest pull of the contour, in face widths
REACH = (0.06, 0.42)  # how far into the face a pull reaches, in face widths
CLEAR = 0.8  # a pull reaches this share of the way to the nearest eye, the nose, the mouth or the chin
STROKE = 3  # px beyond the skin's edge that go with the layer: the contour's drawn line
EDGE_STEP = 18  # Lab distance from the skin just inside at which a pixel is past the edge
SKIN_LIMIT = 48  # Lab distance from the cheek's own skin at which a pixel is not skin at all
FRONT_SKIN = 20  # beyond hair that ends a row, pixels this close to the cheek's skin mean the hair lies over the face
OUTLIER = 0.06  # in D: rows farther than this from the smooth contour (an ear, a strand) take the smooth contour
HAIR_COVER = 0.3  # share of a cheek under hair that has no layer of its own at which the pull is gone
MIN_TRAVEL = 0.25  # a side whose travel shrinks below this share of the full travel is left out
POINTS = 24  # contour points written per side
HIDDEN = 4  # px: a row whose seen face ends this far inside the contour has hair over the contour
COVER_EASE = 0.05  # in L: the pull stays off this far beside hair over the contour and eases in over twice that
LINE = 6  # px either side of the contour in which hair_front pixels can be the contour's own drawn line
LINE_HAIR = 16  # hair_front within this many px (through hair_front) of hair beyond the band stays hair
OUTER = 10  # px outside the contour to which that band reaches
THICK = 5  # px: in the band, hair_front narrower than this is the contour's line, through which no hair reaches in
BESIDE = (4, 17)  # px outside the contour (from, to) where hair in front of the face lies beside it, past its own line
SMOOTH = 0.03  # in L: the traced contour is smoothed over this many rows (a pulled contour must turn smoothly)
SWEEP = 4  # px beyond the largest pull and the layer's stroke: how far beside the contour a pulled cheek can reach
SPECK = 40  # px: a piece of hair_front beside the contour this small is residue of the matting, not hair
ENDS = 0.15  # the runtime's pull eases in and out over this share of the contour at both ends (cheek.ts CHEEK_ENDS)
CENTRES = (0.35, 0.65)  # where along the contour the runtime can centre its pull (cheek.ts cheekPose)
HOLD_EASE = 1.0  # in travels: where the pull is held back, it eases back in over this many rows of contour


def _sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def frame(rig):
    """The face frame: eye midpoint, unit vectors down the face and across it (to the image right), L and D."""
    a, b = rig["face"]["eyeA"], rig["face"]["eyeB"]
    eyes = np.array([(a["cx"] + b["cx"]) / 2, (a["cy"] + b["cy"]) / 2], np.float64)
    mouth = np.array([rig["mouth"]["cx"], rig["mouth"]["cy"]], np.float64)
    along = float(np.linalg.norm(mouth - eyes))
    between = math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])
    if along < 8 or between < 8:
        return None
    down = (mouth - eyes) / along
    return {"eyes": eyes, "down": down, "across": np.array([down[1], -down[0]]), "L": along, "D": between}


def bump(t):
    """Raised cosine over the contour (0 at both ends, 1 in the middle): about the shape of the runtime's pull."""
    t = np.clip(t, 0, 1)
    return 0.5 - 0.5 * np.cos(2 * np.pi * t)


def peak(s):
    """The largest share of the pull that the contour takes at s (0..1) wherever the runtime centres it: its raised cosine
    (half the contour wide either side of a centre in CENTRES) eased over ENDS at both ends (cheek.ts cheekBump)."""
    s = np.asarray(s, np.float64)
    off = np.clip(np.maximum(CENTRES[0] - s, s - CENTRES[1]), 0, None)
    shape = np.where(off < 0.5, 0.5 + 0.5 * np.cos(2 * np.pi * off), 0.0)
    return shape * _sstep(0, ENDS, s) * _sstep(0, ENDS, 1 - s)


def ease_under(cap, length):
    """Per row shares (0..1) no larger than `cap` anywhere and rising from each row at most as fast as a smoothstep that
    reaches 1 after `length` rows: a pull held back at some rows turns into them gently instead of in a step."""
    cap = np.asarray(cap, np.float64)
    out = cap.copy()
    if length <= 0:
        return out
    k = np.arange(len(cap))
    for j in np.nonzero(cap < 1)[0]:
        out = np.minimum(out, cap[j] + (1 - cap[j]) * _sstep(0, length, np.abs(k - j)))
    return out


def _smooth(values, sigma):
    radius = max(1, int(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _trace(lab, stop, skin_distance, row, start, step, reach):
    """Scan one row outwards from `start`: the skin's outer edge (column, fractional) and whether hair in front of the
    face ends it (skin again just beyond that hair), or (nan, False)."""
    n = lab.shape[1]
    i, moved = start, 0
    while 0 <= i < n and (stop[row, i] or skin_distance[row, i] > SKIN_LIMIT):
        i += step
        moved += 1
        if moved > reach:
            return math.nan, False
    if not 0 <= i < n:
        return math.nan, False
    recent = [lab[row, i]]
    reference = lab[row, i]
    last = i
    while True:
        i += step
        if not 0 <= i < n:
            return math.nan, False
        if stop[row, i]:
            break
        far = np.linalg.norm(lab[row, i] - reference) > EDGE_STEP or skin_distance[row, i] > SKIN_LIMIT
        if far:
            if abs(i - last) >= 2:
                break
            continue
        last = i
        recent.append(lab[row, i])
        if len(recent) > 9:
            recent.pop(0)
        reference = np.median(np.array(recent), axis=0)
    edge = last + step * 0.5
    # hair that ends with skin a little further out lies over the face, it is not beside it
    front, run = False, 0
    for j in range(i, i + step * max(4, int(reach * 0.3)), step):
        if not 0 <= j < n:
            break
        if not stop[row, j] and skin_distance[row, j] < FRONT_SKIN:
            run += 1
            if run >= 4:
                front = True
                break
        else:
            run = 0
    return edge, front


def _runs_with(mask, seeds):
    """The runs of consecutive True in `mask` (1D) that contain a True of `seeds`."""
    out = np.zeros(len(mask), bool)
    start = None
    for k in range(len(mask) + 1):
        if k < len(mask) and mask[k]:
            start = k if start is None else start
        elif start is not None:
            if seeds[start:k].any():
                out[start:k] = True
            start = None
    return out


def _contour(v, e, D, sigma):
    """The traced edge distances made smooth: rows far from a robust cubic fit (an ear, a strand over the edge) take the
    fit. Also returns the rows that ended well inside the fit: hair lying over the contour."""
    ok = np.isfinite(e)
    if ok.sum() < 8:
        return None
    x, y = v[ok], e[ok]
    weights = np.ones_like(y)
    scale = max(OUTLIER * D / 2, 1.0)
    for _ in range(5):
        coefficients = np.polyfit(x, y, 3, w=np.sqrt(weights))
        residual = y - np.polyval(coefficients, x)
        weights = np.maximum(np.clip(1 - (residual / (4.685 * scale)) ** 2, 0, None) ** 2, 1e-4)
    fit = np.polyval(coefficients, v)
    kept = np.where(ok & (np.abs(np.nan_to_num(e) - fit) <= OUTLIER * D), e, fit)
    return _smooth(kept, sigma), ok & (np.nan_to_num(e) < fit - OUTLIER * D)


def _point(f, u, v):
    return f["eyes"] + u * f["across"] + v * f["down"]


def _thick(alpha):
    """Where hair_front (8-bit alpha) is at least THICK px across, with its soft edge."""
    core = cv2.morphologyEx((alpha >= 128).astype(np.uint8), cv2.MORPH_OPEN, np.ones((THICK, THICK), np.uint8))
    return (cv2.dilate(core, np.ones((5, 5), np.uint8)) > 0) & (alpha > 0)


def _thin(alpha):
    """hair_front pixels (8-bit alpha) that are not part of hair at least THICK px across: lock tips, single strands."""
    return (alpha > 0) & ~_thick(alpha)


def over(below, above):
    """`above` over `below` (straight 8-bit RGBA)."""
    a = above[..., 3:4].astype(np.float32) / 255
    b = below[..., 3:4].astype(np.float32) / 255
    alpha = a + b * (1 - a)
    colour = (above[..., :3] * a + below[..., :3] * b * (1 - a)) / np.maximum(alpha, 1e-6)
    return np.concatenate([np.clip(np.rint(colour), 0, 255), np.clip(np.rint(alpha * 255), 0, 255)], -1).astype(np.uint8)


def _beside(f, half, rect_shape, shape, traced, rows, end, travel, grow=6):
    """Image masks: where a pulled cheek can reach beside its contour (from the contour out to the largest pull with the
    layer's stroke, along the span), and the face more than LINE px inside a contour."""
    sweep_r, deep_r = np.zeros(rect_shape, bool), np.zeros(rect_shape, bool)
    out = int(math.ceil(travel + STROKE + SWEEP))
    columns = np.arange(rect_shape[1], dtype=np.float64) - half
    for sign, smooth, *_ in traced.values():
        e, span = smooth[: end + 1], rows[: end + 1]
        r0, r1 = max(int(span[0]) - grow, 0), min(int(span[-1]) + grow, rect_shape[0] - 1)
        ext = np.interp(np.arange(r0, r1 + 1), span, e)[:, None]
        band = np.zeros(rect_shape, np.uint8)
        cv2.polylines(band, [np.rint(np.stack([half + sign * e, span], 1)).astype(np.int32)], False, 255, 2 * out + 1)
        u = sign * columns[None, :]
        sweep_r[r0 : r1 + 1] |= (band[r0 : r1 + 1] > 0) & (u >= ext - 1)
        deep_r[r0 : r1 + 1] |= (u >= 0) & (u < ext - LINE)
    # the face frame's rows and columns back to the picture
    to_frame = np.array([[f["across"][0], f["across"][1], half - float(f["eyes"] @ f["across"])],
                         [f["down"][0], f["down"][1], -float(f["eyes"] @ f["down"])]], np.float64)

    def back(mask):
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        return cv2.warpAffine(mask.astype(np.uint8) * 255, to_frame, (shape[1], shape[0]), flags=flags, borderValue=0) > 0

    return back(sweep_r), back(deep_r)


def hold(alpha, f, sign, e, span, travel):
    """Per contour row (`e` at rows `span`) the share of the pull that keeps the pulled contour, with the layer's stroke,
    short of hair in front of the face beside it (`alpha`: hair_front's, full size) along the contour's outward normal,
    wherever the runtime centres its pull, eased so that it turns in gently: the pulled cheek does not pass under that
    hair to come out where it ends in a step or a notch. Hair at least THICK px across holds the pull anywhere it can
    reach; thinner hair only within LINE + OUTER px of the contour (an ear's outline, a strand along the contour): a strand
    further out may cross the pulled cheek in front of it."""
    de = np.gradient(e) if len(e) > 1 else np.zeros_like(e)
    norm = np.hypot(1.0, de)
    normal = (sign / norm)[:, None] * f["across"] + (-de / norm)[:, None] * f["down"]
    base = f["eyes"] + (sign * e)[:, None] * f["across"] + span[:, None] * f["down"]
    ts = np.arange(1, int(math.ceil(travel + STROKE + SWEEP)) + 1, dtype=np.float64)
    pts = np.rint(base[:, None, :] + normal[:, None, :] * ts[None, :, None]).astype(int)
    h, w = alpha.shape
    at = (np.clip(pts[..., 1], 0, h - 1), np.clip(pts[..., 0], 0, w - 1))
    hit = _thick(alpha)[at] | ((alpha[at] >= 128) & (ts[None, :] <= LINE + OUTER))
    gap = np.where(hit.any(axis=1), ts[np.argmax(hit, axis=1)], np.inf)
    # how far the contour may move out at each row (all of the travel where nothing is near), and no more than at the rows
    # within half the spacing of the POINTS written, as the runtime interpolates between them
    room = np.clip(gap - STROKE - 1, 0, travel) / travel
    spacing = int(math.ceil((len(e) - 1) / (POINTS - 1) / 2))
    room = cv2.erode(room.astype(np.float32)[:, None], np.ones((2 * spacing + 1, 1), np.uint8))[:, 0].astype(np.float64)
    # rising from the held rows like a smoothstep over HOLD_EASE travels: the pulled contour meets the still one there
    # without a corner. As a share of the pull the contour takes at each row wherever it is centred
    allowed = ease_under(room, max(4.0, HOLD_EASE * travel))
    steps = np.hypot(*np.diff(base, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(steps)]) / max(float(steps.sum()), 1e-9)
    return np.clip(allowed / np.maximum(peak(s), 1e-9), 0, 1)


def cut(drawn, hair, rig, excluded, polygon_mask, head=None):
    """Cheek layers of the face drawn as `drawn` (RGBA: the base the runtime draws, eyes filled). `hair` (8 bit) is the
    hair mask, `excluded` parts that are not the face (hands, accessories, limbs, garments), `head` None or the head
    layers at full size: {"hair": hair_front, "skin": face_skin} (RGBA).

    Returns ({layer name: (image, rect)}, {side: data}, [notes], head layers): sides that cannot be traced are left out
    with a note; the head layers come back with the contour's drawn line moved from the hair to the skin (or None).
    """
    notes = []
    layered = head is not None
    if layered:
        front = head["hair"][..., 3].astype(np.float32) / 255
        hair = np.clip(np.rint(hair.astype(np.float32) * (1 - front)), 0, 255).astype(np.uint8)
        face = over(drawn, head["skin"])
    else:
        face = drawn
    f = frame(rig)
    if f is None:
        return {}, {}, ["the eyes and the mouth do not give a face frame"], head
    L, D = f["L"], f["D"]
    h, w = face.shape[:2]
    half = int(math.ceil(1.7 * D))
    us = np.arange(-half, half + 1, dtype=np.float64)
    vs = np.arange(0, int(math.ceil((SPAN_BOTTOM + 0.3) * L)) + 1, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    mx = (f["eyes"][0] + uu * f["across"][0] + vv * f["down"][0]).astype(np.float32)
    my = (f["eyes"][1] + uu * f["across"][1] + vv * f["down"][1]).astype(np.float32)

    def warp(image, interpolation=cv2.INTER_LINEAR):
        return cv2.remap(image, mx, my, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def failed(note):
        return {}, {}, notes + [note], head

    rect_face = warp(np.ascontiguousarray(face))
    lab = cv2.cvtColor(np.ascontiguousarray(rect_face[..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
    rect_hair = warp(np.ascontiguousarray(hair))
    if layered:
        # where the face is seen: the hair in front of it stops a row there, where the face under it goes on. Hair in front
        # thinner than THICK px (lock tips lying across the cheek, single strands) does not: the face goes on beyond it,
        # so it neither hides the contour nor cuts the layer along it
        thin = _thin(head["hair"][..., 3])
        thick_hair = head["hair"].copy()
        thick_hair[thin, 3] = 0
        seen = warp(np.ascontiguousarray(over(face, thick_hair)))
        seen_lab = cv2.cvtColor(np.ascontiguousarray(seen[..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
        seen_front = warp(np.ascontiguousarray(thick_hair[..., 3]))
        near_thin = warp(cv2.dilate(thin.astype(np.uint8) * 255, np.ones((5, 5), np.uint8)), cv2.INTER_NEAREST) > 0
        seen_stop = (seen[..., 3] < 128) | (seen_front >= 128) | ((rect_hair >= 128) & ~near_thin)
    stop = (rect_face[..., 3] < 128) | (rect_hair >= 128) | (warp(excluded.astype(np.uint8) * 255, cv2.INTER_NEAREST) > 0)

    # where each cheek is: the rig's blush points, by their side of the face
    cheeks = sorted(
        ((float(np.dot(np.array(p, np.float64) - f["eyes"], f["across"])), float(np.dot(np.array(p, np.float64) - f["eyes"], f["down"])))
         for p in rig.get("cheeks", [])[:2]),
    )
    if len(cheeks) != 2 or not cheeks[0][0] < 0 < cheeks[1][0]:
        cheeks = [(-0.5 * D, 0.6 * L), (0.5 * D, 0.6 * L)]

    # what a pull must not reach: the eyes, the nose, the mouth, the chin and the middle of the face
    protected = np.zeros(lab.shape[:2], np.uint8)

    def rectified(point):
        d = np.array(point, np.float64) - f["eyes"]
        return float(np.dot(d, f["across"])) + half, float(np.dot(d, f["down"]))

    for eye in rig["eyes"]:
        poly = np.array([rectified(p) for p in eye["roi"]], np.int32)
        cv2.fillPoly(protected, [poly], 255)
    turn = math.degrees(math.atan2(f["across"][1], f["across"][0]))
    nose = rig["face"]["nose"]
    cx, cy = rectified((nose["cx"], nose["cy"]))
    cv2.ellipse(protected, (int(round(cx)), int(round(cy))), (max(1, int(nose["rx"] * 0.6)), max(1, int(nose["ry"] * 0.6))), 0, 0, 360, 255, -1)
    area = rig["mouth"]["area"]
    cx, cy = rectified((area["cx"], area["cy"]))
    cv2.ellipse(protected, (int(round(cx)), int(round(cy))), (max(1, int(area["rx"])), max(1, int(area["ry"]))),
                math.degrees(area.get("angle", 0)) - turn, 0, 360, 255, -1)
    jaw = rig["face"]["jaw"]
    chin = np.array([jaw["cx"], jaw["cy"]], np.float64) + 0.9 * jaw["ry"] * f["down"]
    chin_v = float(np.dot(chin - f["eyes"], f["down"]))
    cx, cy = rectified(chin)
    cv2.circle(protected, (int(round(cx)), int(round(cy))), max(2, int(0.15 * D)), 255, -1)
    protected[:, half - int(0.1 * D) : half + int(0.1 * D) + 1] = 255
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8), iterations=max(1, int(0.02 * D)))
    clearance = cv2.distanceTransform((protected == 0).astype(np.uint8), cv2.DIST_L2, 5)

    top = int(round(SPAN_TOP * L))
    last = int(min(len(vs) - 1, round(min(SPAN_BOTTOM * L, chin_v - CHIN_CLEAR * L))))
    if last - top < MIN_SPAN * L:
        return failed("the chin is too close to the eyes for a cheek")
    rows = np.arange(top, last + 1)
    traced, widths = {}, {}
    for side, sign, (cu, cv) in (("left", -1, cheeks[0]), ("right", 1, cheeks[1])):
        r = max(4, int(0.1 * D))
        ci, cj = int(round(cv)), int(round(cu)) + half
        patch = (slice(max(0, ci - r), ci + r + 1), slice(max(0, cj - r), cj + r + 1))
        samples = lab[patch][~stop[patch]]
        if len(samples) < 30:
            notes.append(f"{side}: no skin at the cheek")
            continue
        skin = np.median(samples, axis=0)
        skin_distance = np.linalg.norm(lab - skin, axis=-1)
        start = half + sign * int(max(0.2 * D, 0.55 * abs(cu)))
        edges, fronts = np.full(len(rows), np.nan), np.zeros(len(rows), bool)
        for k, row in enumerate(rows):
            edge, front = _trace(lab, stop, skin_distance, row, start, sign, int(0.5 * D))
            edges[k], fronts[k] = abs(edge - half) if math.isfinite(edge) else math.nan, front
        if np.isfinite(edges).sum() < 0.5 * len(rows):
            notes.append(f"{side}: the contour is not found below the eyes")
            continue
        contour = _contour(rows.astype(np.float64), edges, D, max(2.0, SMOOTH * L))
        if contour is None:
            notes.append(f"{side}: the contour is not found below the eyes")
            continue
        smooth, inward = contour

        def at_hair(mask, row, edge):
            """Whether hair (8-bit mask, rectified) lies just outside a traced edge."""
            column = int(round(half + sign * edge))
            return any(0 <= column + sign * j < mask.shape[1] and mask[row, column + sign * j] >= 128 for j in range(1, 4))

        # rows that end well inside the contour because hair lies over it (not a shadow or a line on the cheek)
        inward = np.array([inward[k] and math.isfinite(edges[k]) and at_hair(rect_hair, row, edges[k]) for k, row in enumerate(rows)])
        hidden, seen_edges = np.zeros(len(rows), bool), np.full(len(rows), np.inf)
        beside = np.zeros(len(rows), bool)
        if layered:
            seen_distance = np.linalg.norm(seen_lab - skin, axis=-1)
            for k, row in enumerate(rows):
                edge, _ = _trace(seen_lab, seen_stop, seen_distance, row, start, sign, int(0.5 * D))
                hidden[k] = math.isfinite(edge) and abs(edge - half) < smooth[k] - HIDDEN and at_hair(seen_front, row, abs(edge - half))
                if hidden[k]:
                    seen_edges[k] = abs(edge - half)
            # hair in front of the face that hides the contour and goes on right beside it (an ear the plate took for hair,
            # cut where the hair layer ends): pulled there the contour only slides under that hair, and where the hair
            # ends the pulled cheek comes out from under it in a step. Such rows count as hidden too
            beside = np.array([any(0 <= int(round(half + sign * (smooth[k] + j))) < seen_front.shape[1]
                                   and seen_front[row, int(round(half + sign * (smooth[k] + j)))] >= 128 for j in range(*BESIDE))
                               for k, row in enumerate(rows)])
            beside = _runs_with(hidden | beside, hidden) & ~hidden
            # and the pull eases in only past the end of that hair
            ease = 2 * int(COVER_EASE * L) + 1
            beside = cv2.dilate(beside.astype(np.uint8)[:, None], np.ones((ease, 1), np.uint8))[:, 0] > 0
        # without head layers, hair that ends a row with skin beyond it lies over the cheek; with them that hair is lifted,
        # and only hair over the contour itself (rows ending well inside it) is left to be dragged along
        traced[side] = (sign, smooth, (fronts if head is None else np.zeros_like(fronts)) | inward, skin, hidden | inward | beside, seen_edges)
        widths[side] = smooth
    if not traced:
        return {}, {}, notes, head

    # the span ends where the jaw turns towards the chin
    width_rows = np.mean([widths[s] for s in widths], axis=0)
    widest = int(np.argmax(width_rows))
    narrow = np.nonzero(width_rows[widest:] < NARROW * width_rows[widest])[0]
    end = widest + int(narrow[0]) - 1 if len(narrow) else len(rows) - 1
    if rows[end] - rows[0] < MIN_SPAN * L:
        return failed("the cheeks are too short below the eyes")
    mouth_row = int(np.clip(round(L) - rows[0], 0, end))
    face_width = float(sum(widths[s][mouth_row] for s in widths) * (2 / len(widths)))
    travel = TRAVEL * face_width

    layers, data = {}, {}
    if layered:
        # the contour's drawn line in hair_front moves to face_skin: hair_front pixels near the contour that are not
        # part of hair reaching out of that band (hair crossing the contour stays in front). The band reaches further
        # out than in (OUTER px): the line is drawn outside the skin's edge, and can lie a few px off the traced one
        band = np.zeros((h, w), np.uint8)
        for sign, smooth, *_ in traced.values():
            pts = np.rint(np.array([_point(f, sign * x, y) for x, y in zip(smooth, rows)])).astype(np.int32)
            cv2.polylines(band, [pts], False, 255, 2 * LINE + 1)
            out = np.rint(np.array([_point(f, sign * (x + (LINE + OUTER) / 2), y) for x, y in zip(smooth, rows)])).astype(np.int32)
            cv2.polylines(band, [out], False, 255, OUTER - LINE + 1)
        alpha = head["hair"][..., 3]
        reached = ((alpha > 64) & ~(cv2.dilate(band, np.ones((7, 7), np.uint8)) > 0)).astype(np.uint8)
        within = (alpha > 12).astype(np.uint8)
        # but not along the contour's own line: a thin run of hair_front in the band (the line itself, which can touch
        # real hair at its ends) carries nothing from beyond the band into it
        thick = cv2.morphologyEx(within, cv2.MORPH_OPEN, np.ones((THICK, THICK), np.uint8))
        within[(band > 0) & (thick == 0)] = 0
        for _ in range(LINE_HAIR):
            reached = cv2.dilate(reached, np.ones((3, 3), np.uint8)) & within
        moved = (band > 0) & (alpha > 0) & (reached == 0)
        # beside each contour, as far out as a pull with the layer's stroke can reach: pieces of hair_front there that are
        # residue of the matting (no larger than SPECK px, or thinner than THICK px and staying beside the contour, and
        # neither reaching more than LINE px into the face) go to face_skin as well, so a pulled cheek covers them instead
        # of passing under them (a dark dot or a short line on the moved cheek). Larger hair beside the contour stays in
        # front, and the pull is held short of it (hold)
        sweep, deep = _beside(f, half, lab.shape[:2], (h, w), traced, rows, end, travel)
        pieces = (alpha > 12) & ~moved
        count, labels, stats, _ = cv2.connectedComponentsWithStats(pieces.astype(np.uint8), connectivity=8)
        if count > 1:
            def tally(mask):
                return np.bincount(labels[mask & pieces], minlength=count)

            area = stats[:, cv2.CC_STAT_AREA]
            near = cv2.dilate((sweep | (band > 0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            thin_piece = (tally(_thick(alpha)) == 0) & (tally(near) == area)
            residue = (tally(sweep) > 0) & (tally(deep) == 0) & ((area <= SPECK) | thin_piece)
            residue[0] = False
            if residue.any():
                speck = residue[labels]
                faint = (cv2.dilate(speck.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (alpha > 0) & (alpha <= 12)
                moved |= speck | faint
        if moved.any():
            hair_layer, skin_layer = head["hair"].copy(), head["skin"].copy()
            skin_layer[moved] = over(skin_layer[moved][None], hair_layer[moved][None])[0]
            hair_layer[moved] = 0
            head = {"hair": hair_layer, "skin": skin_layer}
            face = over(drawn, skin_layer)
    span = rows[: end + 1]
    t = (span - span[0]) / max(1, span[-1] - span[0])
    along = bump(t)
    front_alpha = head["hair"][..., 3] if layered else None
    for side, (sign, smooth, fronts, skin, covered, seen_edges) in traced.items():
        e = smooth[: end + 1]
        # the contour stays where hair lies over it (the hair in front of the face, or any hair without head layers):
        # pulled there it would come out from under that hair
        near = cv2.dilate(covered[: end + 1].astype(np.uint8)[:, None], np.ones((2 * int(COVER_EASE * L) + 1, 1), np.uint8))[:, 0]
        # none of it there, then eased in over about a travel of contour, so the pulled contour turns out of the still one
        # gently (smoothing the rows over leaked a little of the pull into them: a step where the layer ends at that hair)
        free = ease_under(np.where(near > 0, 0.0, 1.0), max(4.0, HOLD_EASE * travel))
        cols = half + sign * e
        inner = np.clip(np.rint(cols - sign * 2), 0, lab.shape[1] - 1).astype(int)
        raw = clearance[span, inner]
        reach = np.minimum(CLEAR * raw, REACH[1] * face_width)
        reach = np.maximum(_smooth(reach, 4), 4.0)
        if np.average(reach, weights=along + 1e-6) < REACH[0] * face_width:
            notes.append(f"{side}: the cheek is too narrow beside the eyes, the nose and the mouth")
            continue
        # hair over the cheek without a layer of its own would be dragged along: shrink the travel by its share
        weight_sum, covered = 0.0, 0.0
        for k, row in enumerate(span):
            if along[k] < 1e-3:
                continue
            c0 = int(round(cols[k]))
            depth = np.arange(0, int(reach[k]) + 1)
            cs = np.clip(c0 - sign * depth, 0, lab.shape[1] - 1)
            wts = along[k] * (1 - _sstep(0, reach[k], depth))
            weight_sum += wts.sum()
            covered += (wts * (rect_hair[row, cs] >= 128)).sum()
        share = max(covered / max(weight_sum, 1e-6), float(np.average(fronts[: end + 1], weights=along + 1e-6)))
        side_travel = travel * max(0.0, 1 - share / HAIR_COVER)
        if side_travel < MIN_TRAVEL * travel:
            notes.append(f"{side}: hair lies over {share:.0%} of the cheek")
            continue
        if layered:
            # hair in front of the face lying beside the contour (a lock beside it, the dark gap between the jaw and the
            # hair that the plate's matting took for hair in front): pulled into it, the cheek would pass under that hair
            # and come out where it ends in a step or a notch. The pull stays short of it and eases back in
            free = np.minimum(free, hold(front_alpha, f, sign, e, span, side_travel))
        # the layer: from the contour's drawn line inwards past the reach, a little longer than the span
        grow = 6
        vv_edge = np.concatenate([np.full(grow, span[0]) - np.arange(grow, 0, -1), span, span[-1] + np.arange(1, grow + 1)])
        ee = np.concatenate([np.full(grow, e[0]), e, np.full(grow, e[-1])])
        rr = np.concatenate([np.full(grow, reach[0]), reach, np.full(grow, reach[-1])])
        # where hair in front of the face hides the contour the layer ends at that hair: what lies under it is never shown
        seen_out = np.concatenate([np.full(grow, seen_edges[0]), seen_edges[: end + 1], np.full(grow, seen_edges[end])])
        outer = [_point(f, sign * (min(x, s) + STROKE), y) for x, s, y in zip(ee, seen_out, vv_edge)]
        inside = [_point(f, sign * max(x - r - 8, 0.0), y) for x, y, r in zip(ee, vv_edge, rr)]
        region = polygon_mask((h, w), np.rint(np.array(outer + inside[::-1])).astype(np.int32))
        soft = np.maximum(region.astype(np.float32) / 255, cv2.GaussianBlur(region.astype(np.float32) / 255, (0, 0), 1.0))
        soft[excluded > 0] = 0
        layer = face.copy()
        layer[..., 3] = np.clip(np.rint(face[..., 3].astype(np.float32) * soft), 0, 255).astype(np.uint8)
        ys, xs = np.nonzero(layer[..., 3])
        if not len(xs):
            notes.append(f"{side}: the cheek has no pixels")
            continue
        x0, y0 = max(int(xs.min()) - 2, 0), max(int(ys.min()) - 2, 0)
        x1, y1 = min(int(xs.max()) + 3, w), min(int(ys.max()) + 3, h)
        layers[f"cheek_{side}"] = (np.ascontiguousarray(layer[y0:y1, x0:x1]), [x0, y0, x1 - x0, y1 - y0])
        picks = np.unique(np.rint(np.linspace(0, len(span) - 1, POINTS)).astype(int))
        data[side] = {
            "edge": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in (_point(f, sign * e[k], span[k]) for k in picks)],
            "reach": [round(float(reach[k]), 1) for k in picks],
            "travel": round(float(side_travel), 1),
            **({"free": [round(float(free[k]), 2) for k in picks]} if free[picks].min() < 0.99 else {}),
        }
    return layers, data, notes, head
