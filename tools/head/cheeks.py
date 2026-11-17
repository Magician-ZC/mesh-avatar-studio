"""Cheek layers for build-layers.py (--cheek-pull): cheeks that the runtime can pull past the face's contour.

For each side of the face (left and right in the image) the visible outer contour of the face is traced from below the
eyes to above the chin, and the tool writes
  cheek_left.png / cheek_right.png  the skin of that cheek from the contour inwards, with the contour's drawn line, cut
                  from the face as it is under the hair in front of it (the base with face_skin over it when there are
                  head layers, the base otherwise);
  layers.json "cheeks"  {"left": {"edge": [[x, y], ...], "reach": [px, ...], "travel": px, "free": [...]}, "right": ...}:
                  the contour from top to bottom, how far into the face a pull reaches at each of its points (short of
                  the eyes, the nose, the mouth and the chin), the largest pull of the contour in source pixels and,
                  where hair lies over the contour, how much of the pull each point takes (1 elsewhere; left out when 1
                  everywhere).
The runtime draws a cheek layer only while that cheek is pulled: over the base, under the eyes, the drawn mouths and the
hair in front of the face. It moves the layer's contour only outwards, so the old contour and the hair beside the face
stay covered. base.png is not changed, so runtimes that do not know these layers draw the avatar as before.

With head layers the contour's own drawn line can sit in hair_front (where the plate or the colours took it for hair).
Hair in front of the face stays where it is while the cheek moves under it, so the line would stay behind: the pixels of
hair_front in a narrow band along the contour that are not part of hair reaching beyond that band move to face_skin (the
two over the base are the same picture), and the cheek layer carries them.

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
SMOOTH = 0.03  # in L: the traced contour is smoothed over this many rows (a pulled contour must turn smoothly)


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


def over(below, above):
    """`above` over `below` (straight 8-bit RGBA)."""
    a = above[..., 3:4].astype(np.float32) / 255
    b = below[..., 3:4].astype(np.float32) / 255
    alpha = a + b * (1 - a)
    colour = (above[..., :3] * a + below[..., :3] * b * (1 - a)) / np.maximum(alpha, 1e-6)
    return np.concatenate([np.clip(np.rint(colour), 0, 255), np.clip(np.rint(alpha * 255), 0, 255)], -1).astype(np.uint8)


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
        # where the face is seen: the hair in front of it stops a row there, where the face under it goes on
        seen = warp(np.ascontiguousarray(over(face, head["hair"])))
        seen_lab = cv2.cvtColor(np.ascontiguousarray(seen[..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
        seen_front = warp(np.ascontiguousarray(head["hair"][..., 3]))
        seen_stop = (seen[..., 3] < 128) | (seen_front >= 128) | (rect_hair >= 128)
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
        hidden = np.zeros(len(rows), bool)
        if layered:
            seen_distance = np.linalg.norm(seen_lab - skin, axis=-1)
            for k, row in enumerate(rows):
                edge, _ = _trace(seen_lab, seen_stop, seen_distance, row, start, sign, int(0.5 * D))
                hidden[k] = math.isfinite(edge) and abs(edge - half) < smooth[k] - HIDDEN and at_hair(seen_front, row, abs(edge - half))
        # without head layers, hair that ends a row with skin beyond it lies over the cheek; with them that hair is lifted,
        # and only hair over the contour itself (rows ending well inside it) is left to be dragged along
        traced[side] = (sign, smooth, (fronts if head is None else np.zeros_like(fronts)) | inward, skin, hidden | inward)
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
        # part of hair reaching out of that band (hair crossing the contour stays in front)
        band, line = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
        for sign, smooth, *_ in traced.values():
            pts = np.rint(np.array([_point(f, sign * x, y) for x, y in zip(smooth, rows)])).astype(np.int32)
            cv2.polylines(band, [pts], False, 255, 2 * LINE + 1)
            cv2.polylines(line, [pts], False, 255, 1)
        alpha = head["hair"][..., 3]
        distance = cv2.distanceTransform(255 - line, cv2.DIST_L2, 5)
        reached = ((alpha > 64) & (distance > LINE + 3)).astype(np.uint8)
        within = (alpha > 12).astype(np.uint8)
        for _ in range(LINE_HAIR):
            reached = cv2.dilate(reached, np.ones((3, 3), np.uint8)) & within
        moved = (band > 0) & (alpha > 0) & (reached == 0)
        if moved.any():
            hair_layer, skin_layer = head["hair"].copy(), head["skin"].copy()
            skin_layer[moved] = over(skin_layer[moved][None], hair_layer[moved][None])[0]
            hair_layer[moved] = 0
            head = {"hair": hair_layer, "skin": skin_layer}
            face = over(drawn, skin_layer)
    span = rows[: end + 1]
    t = (span - span[0]) / max(1, span[-1] - span[0])
    along = bump(t)
    for side, (sign, smooth, fronts, skin, covered) in traced.items():
        e = smooth[: end + 1]
        # the contour stays where hair lies over it (the hair in front of the face, or any hair without head layers):
        # pulled there it would come out from under that hair
        near = cv2.dilate(covered[: end + 1].astype(np.uint8)[:, None], np.ones((2 * int(COVER_EASE * L) + 1, 1), np.uint8))[:, 0]
        free = np.clip(1 - _smooth(near.astype(np.float64), max(2.0, 2 * COVER_EASE * L)) * 1.6, 0, 1)
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
        # the layer: from the contour's drawn line inwards past the reach, a little longer than the span
        grow = 6
        vv_edge = np.concatenate([np.full(grow, span[0]) - np.arange(grow, 0, -1), span, span[-1] + np.arange(1, grow + 1)])
        ee = np.concatenate([np.full(grow, e[0]), e, np.full(grow, e[-1])])
        rr = np.concatenate([np.full(grow, reach[0]), reach, np.full(grow, reach[-1])])
        outer = [_point(f, sign * (x + STROKE), y) for x, y in zip(ee, vv_edge)]
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
