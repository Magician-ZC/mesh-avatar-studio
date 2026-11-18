"""What lies behind an accessory cut by an outline (a handbag hanging over a skirt): the garment goes on there, with its
print, its folds and its seams, so the base behind the accessory is filled with the garment's own texture.

Where the hole is wide (the bag's body), the garment beside it is copied whole: the one shift that lays a piece of the
garment over the hole so that the ring round the hole matches the drawing best, with every pixel of the piece on the
garment. A tiered skirt beside a bag copies along its tiers, so the seams, the print and the folds go on behind the bag
in one piece; what still differs along the ring (light, shade) is spread smoothly across the piece.

The narrow rest (a strap) is an exemplar fill (Criminisi, Pérez and Toyama, 2004): from the edge inwards, the place where
the most is known and a drawn line or a seam runs into the hole goes first; it takes the patch of the garment round it
that best matches what is already there and copies what is missing. Both searches run on a grid coarser by a whole
factor (a hole as large as a bag stays quick); each pixel then copies the full-resolution pixel its coarse pixel came
from, so the fill keeps the drawing's own detail, and the seams between patches are blended by letting neighbours vote.

Its silhouette is the figure's outline interpolated across the hole (masked averages down an image pyramid of the
alpha): the garment goes on behind the accessory as far as the figure does around it and stops at a smooth edge. It never
goes behind the accessory's own soft edge onto the background (the picture's pixels there are not solid: the inside of a
strap loop, the matte's antialiased rim): the accessory's layer draws those pixels whole, colour and coverage, so anything
behind them would show at rest; the base stays transparent there, as under a limb's antialiased silhouette.

Only numpy and OpenCV, no learned model: the same picture always gives the same fill.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

PATCH = 22  # px at full resolution: the side of a copied patch
SEARCH = 360  # px at full resolution: how far from a patch its source is looked for
COARSE = 50_000  # most hole pixels the search runs on (the grid is coarser by a whole factor until it is no more)
EDGE = 4  # px: source patches stay this far inside the figure (its own edge and line art are not the garment)
SOLID = 250  # picture alpha from which a pixel inside the hole is solid, not the accessory's soft edge onto the background
VOTE = 3  # px: neighbours that vote on each filled pixel at full resolution
PRIORITY_FLOOR = 0.05  # added to the data term, so a flat front still fills from where most is known
WIDE = 40  # px: the hole's parts wider than a disk of this radius are copied whole (a strap, beads and all, is not)
RING = 16  # px: the ring round a wide part that its copy must match
DETAIL = 3.0  # weight of the drawing's detail (seams, folds) against its colours when a copy is matched
COVER = 0.98  # of a wide part: the share whose copy must lie on the garment
DRIFT = 0.15  # of a wide part's height (and RING more): how far up or down its copy may come from. A garment hangs down
# and wraps round the body: along a tier its print and folds repeat, down it they change (seams, the hem)


def pushpull(values, known):
    """Values for the pixels that are not `known`, smoothly from the known ones round them (masked averages down an image
    pyramid, filled back up). `values` is H x W x C float32."""
    levels = [(values * known[..., None], known.astype(np.float32))]
    while min(levels[-1][1].shape) > 8 and levels[-1][1].min() == 0:
        colour, weight = levels[-1]
        down = cv2.pyrDown(colour)
        levels.append((down.reshape(down.shape[0], down.shape[1], -1), cv2.pyrDown(weight)))
    colour, weight = levels[-1]
    estimate = colour / np.maximum(weight, 1e-6)[..., None]
    for colour, weight in reversed(levels[:-1]):
        up = cv2.pyrUp(estimate, dstsize=(weight.shape[1], weight.shape[0]))
        estimate = up.reshape(weight.shape[0], weight.shape[1], -1)
        share = np.clip(weight, 0, 1)[..., None]
        estimate = share * (colour / np.maximum(weight, 1e-6)[..., None]) + (1 - share) * estimate
    return estimate


def exemplar(image, target, source, patch, search):
    """Fill `target` of a small RGB image (float32) from patches lying wholly in `source`. Returns the filled image and,
    for each pixel, the pixel it was copied from ((-1, -1) for pixels not filled from a patch)."""
    h, w = target.shape
    r = patch // 2
    image = image.copy()
    known = ~target
    confidence = known.astype(np.float32)
    origin = np.full((h, w, 2), -1, np.int32)
    valid = cv2.erode(source.astype(np.uint8), np.ones((patch, patch), np.uint8), borderType=cv2.BORDER_CONSTANT,
                      borderValue=0) > 0
    if not valid.any():
        return image, origin
    grey = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ys, xs = np.nonzero(target)
    # priorities are only ever needed round the hole
    zone = (slice(max(0, int(ys.min()) - r - 2), min(h, int(ys.max()) + r + 3)),
            slice(max(0, int(xs.min()) - r - 2), min(w, int(xs.max()) + r + 3)))
    oy, ox = zone[0].start, zone[1].start
    ring = np.ones((3, 3), np.uint8)
    while True:
        k = known[zone]
        front = (cv2.dilate(k.astype(np.uint8), ring) > 0) & ~k
        fy, fx = np.nonzero(front)
        if not len(fy):
            break
        share = cv2.boxFilter(confidence[zone], -1, (patch, patch), normalize=True, borderType=cv2.BORDER_CONSTANT)
        kf = k.astype(np.float32)
        inner = cv2.erode(kf, ring)
        g = np.where(k, grey[zone], 0)
        gx = cv2.dilate(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) * inner, ring)
        gy = cv2.dilate(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) * inner, ring)
        nx, ny = cv2.Sobel(kf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(kf, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.hypot(nx[fy, fx], ny[fy, fx]) + 1e-6
        # isophotes running into the hole: the known gradient across the front's normal
        data = np.abs(gy[fy, fx] * nx[fy, fx] - gx[fy, fx] * ny[fy, fx]) / norm / 255.0
        best = int(np.argmax(share[fy, fx] * (data + PRIORITY_FLOOR)))
        py, px = int(fy[best]) + oy, int(fx[best]) + ox
        y0, y1, x0, x1 = py - r, py + r + 1, px - r, px + r + 1
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            # at the image's edge: the pixel takes the mean of its known neighbours
            around = (slice(max(0, py - 1), py + 2), slice(max(0, px - 1), px + 2))
            image[py, px] = image[around][known[around]].mean(axis=0)
            known[py, px], confidence[py, px] = True, share[py - oy, px - ox]
            continue
        seen = known[y0:y1, x0:x1]
        reach = search
        while True:
            sy0, sy1 = max(0, py - reach), min(h, py + reach + 1)
            sx0, sx1 = max(0, px - reach), min(w, px + reach + 1)
            ok = valid[sy0 + r : sy1 - r, sx0 + r : sx1 - r]
            if ok.any() or reach > max(h, w):
                break
            reach *= 2
        if not ok.any():
            break
        cost = cv2.matchTemplate(image[sy0:sy1, sx0:sx1], image[y0:y1, x0:x1], cv2.TM_SQDIFF,
                                 mask=np.repeat(seen[..., None], 3, axis=2).astype(np.float32))
        cost = np.where(ok[: cost.shape[0], : cost.shape[1]], cost, np.inf)
        cy, cx = np.unravel_index(int(np.argmin(cost)), cost.shape)
        qy, qx = sy0 + int(cy), sx0 + int(cx)  # the source patch's top-left corner
        missing = ~seen
        image[y0:y1, x0:x1][missing] = image[qy : qy + patch, qx : qx + patch][missing]
        grey[y0:y1, x0:x1][missing] = cv2.cvtColor(
            np.clip(image[y0:y1, x0:x1], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32)[missing]
        my, mx = np.nonzero(missing)
        origin[y0 + my, x0 + mx] = np.stack([qy + my, qx + mx], axis=1)
        confidence[y0:y1, x0:x1][missing] = share[py - oy, px - ox]
        known[y0:y1, x0:x1] |= missing
    return image, origin


def shift_for(rgb, part, ring, source, step, drift=None):
    """The shift (dy, dx) that lays the garment over `part` best: its ring (`ring`) matches the drawing round it, and
    COVER of `part` and most of the ring fall on `source`. Searched on a grid `step` times coarser, then refined; None
    when no shift does."""
    ys, xs = np.nonzero(part | ring)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = part.shape
    small = lambda a, kind=cv2.INTER_AREA: cv2.resize(a, (max(1, w // step), max(1, h // step)), interpolation=kind)  # noqa: E731
    image = small(rgb.astype(np.float32))
    # the detail - seams, folds, the print's edges - counts beside the colours, so a seam meets its own continuation
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = small(DETAIL * (grey - cv2.GaussianBlur(grey, (0, 0), 3)))
    src = small(source.astype(np.float32))
    box = (slice(y0 // step, max(y0 // step + 1, y1 // step)), slice(x0 // step, max(x0 // step + 1, x1 // step)))
    body = small(part.astype(np.float32))[box]
    band = (small(ring.astype(np.float32))[box] > 0.5).astype(np.float32)
    if band.sum() < 8 or body.sum() < 1:
        return None
    cost = cv2.matchTemplate(image, np.ascontiguousarray(image[box]), cv2.TM_SQDIFF,
                             mask=np.ascontiguousarray(np.repeat(band[..., None], 3, axis=2)))
    cost += 3 * cv2.matchTemplate(detail, np.ascontiguousarray(detail[box]), cv2.TM_SQDIFF, mask=np.ascontiguousarray(band))
    covered = cv2.matchTemplate(src, np.ascontiguousarray(body), cv2.TM_CCORR) / body.sum()
    held = cv2.matchTemplate(src, np.ascontiguousarray(band), cv2.TM_CCORR) / band.sum()
    good = (covered >= COVER) & (held >= 0.9)
    oy, ox = box[0].start, box[1].start
    good[max(0, oy - 1) : oy + 2, max(0, ox - 1) : ox + 2] = False  # not where it already is
    if drift is not None:  # along the garment's tiers
        rows = np.arange(good.shape[0])[:, None]
        good &= np.abs(rows - oy) * step <= drift
    if not good.any():
        return None
    cy, cx = np.unravel_index(int(np.argmin(np.where(good, cost, np.inf))), cost.shape)
    coarse = ((int(cy) - oy) * step, (int(cx) - ox) * step)
    # refined at full resolution round the coarse shift
    fine = DETAIL * (grey - cv2.GaussianBlur(grey, (0, 0), 3))
    ry, rx = np.nonzero(ring)
    py, px = np.nonzero(part)
    best = None
    for dy in range(coarse[0] - step, coarse[0] + step + 1):
        for dx in range(coarse[1] - step, coarse[1] + step + 1):
            qy, qx = py + dy, px + dx
            inside = (qy >= 0) & (qy < h) & (qx >= 0) & (qx < w)
            if inside.mean() < COVER or source[qy[inside], qx[inside]].sum() < COVER * len(py):
                continue
            sy, sx = ry + dy, rx + dx
            on = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            on[on] &= source[sy[on], sx[on]]
            if on.sum() < 8:
                continue
            value = float(np.mean((rgb[ry[on], rx[on]].astype(np.float32) - rgb[sy[on], sx[on]]) ** 2)) * 3 + 3 * float(
                np.mean((fine[ry[on], rx[on]] - fine[sy[on], sx[on]]) ** 2))
            if best is None or value < best[0]:
                best = (value, (dy, dx))
    return None if best is None else best[1]


def copy_whole(rgb, alpha, part, source, step, align_tiers=False):
    """Copy the garment over a wide part of the hole by one shift, with what differs along its ring spread smoothly
    across it. Returns (colours for the part's pixels, which of them were copied, the shift), or None."""
    figure = alpha >= 128
    ring = cv2.dilate(part.astype(np.uint8), np.ones((2 * RING + 1, 2 * RING + 1), np.uint8)) > 0
    ring &= ~part & source & figure
    ys = np.flatnonzero(part.any(axis=1))
    drift = 2 * RING if align_tiers else DRIFT * (ys.max() - ys.min() + 1) + RING
    shift = shift_for(rgb, part, ring, source, step, drift)
    if shift is None:
        shift = shift_for(rgb, part, ring, source, step)
    if shift is None:
        return None
    dy, dx = shift
    h, w = part.shape
    py, px = np.nonzero(part)
    qy, qx = np.clip(py + dy, 0, h - 1), np.clip(px + dx, 0, w - 1)
    copied = source[qy, qx]
    # the difference along the ring next to the part, spread across it
    edge = ring & (cv2.dilate(part.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    ey, ex = np.nonzero(edge)
    sy, sx = np.clip(ey + dy, 0, h - 1), np.clip(ex + dx, 0, w - 1)
    keep = source[sy, sx]
    difference = np.zeros((h, w, 3), np.float32)
    difference[ey[keep], ex[keep]] = rgb[ey[keep], ex[keep]].astype(np.float32) - rgb[sy[keep], sx[keep]]
    known = np.zeros((h, w), bool)
    known[ey[keep], ex[keep]] = True
    if known.any():
        correction = pushpull(difference, known)[py, px]
    else:
        correction = np.zeros((len(py), 3), np.float32)
    colours = np.clip(rgb[qy, qx].astype(np.float32) + correction, 0, 255).astype(np.uint8)
    return colours, copied, shift


def vote(rgb, target, sy, sx, ty, tx, radius):
    """Each filled pixel becomes the mean of what its filled neighbours' sources say it is: where neighbours were copied
    together their votes agree and the copy stays sharp; at a seam between two patches it is blended."""
    h, w = target.shape
    oy = np.full((h, w), -1, np.int64)
    ox = np.full((h, w), -1, np.int64)
    oy[ty, tx], ox[ty, tx] = sy, sx
    total = np.zeros((len(ty), 3), np.float64)
    count = np.zeros(len(ty), np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = ty + dy, tx + dx
            inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            ny, nx = np.where(inside, ny, 0), np.where(inside, nx, 0)
            qy, qx = oy[ny, nx] - dy, ox[ny, nx] - dx
            use = inside & target[ny, nx] & (oy[ny, nx] >= 0) & (qy >= 0) & (qy < h) & (qx >= 0) & (qx < w)
            qy, qx = np.where(use, qy, 0), np.where(use, qx, 0)
            total += rgb[qy, qx] * use[:, None]
            count += use
    return np.where(count[:, None] > 0, total / np.maximum(count, 1)[:, None], rgb[ty, tx]).astype(np.uint8)


def continue_seams(rgba, filled, hole, garment):
    """Match both the colour and direction of the folds entering a pasted panel.
    The hem mask can end at a handle; the opaque cloth right against the cut still supplies its colour."""
    yy, xx = np.nonzero(hole)
    width = min(np.ptp(xx) + 1, np.ptp(yy) + 1)
    radius = max(PATCH, min(100, round(0.19 * width)))
    h, w = hole.shape
    window = (slice(max(0, yy.min() - radius), min(h, yy.max() + radius + 1)),
              slice(max(0, xx.min() - radius), min(w, xx.max() + radius + 1)))
    target = hole[window]
    src, out = rgba[window], filled[window]
    solid = (src[..., 3] >= SOLID) & ~target
    if not solid.any():
        return
    distance, labels = cv2.distanceTransformWithLabels((~solid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    sy, sx = np.nonzero(solid)
    inner = target & (out[..., 3] > 0)
    rim = inner & (distance < 4)
    delta = np.zeros((*target.shape, 3), np.float32)
    index = labels[rim] - 1
    delta[rim] = src[sy[index], sx[index], :3].astype(np.float32) - out[rim, :3]
    # Correct illumination over a wide neighbourhood and edge detail close to the rim.
    correction = np.zeros_like(delta)
    for spread in (radius, radius * 0.4, radius * 0.15, 3):
        weight = cv2.GaussianBlur(rim.astype(np.float32), (0, 0), spread)
        change = cv2.GaussianBlur(delta, (0, 0), spread) / np.maximum(weight[..., None], 1e-6)
        mix = np.exp(-0.5 * (distance / spread) ** 2)[..., None]
        correction += (change - correction) * mix
    matched = np.clip(out[..., :3] + correction, 0, 255)

    # A flared garment's long folds run obliquely into the hole from above and below.
    # Trace them in the direction measured on the garment beside the accessory.
    grey = cv2.GaussianBlur(cv2.cvtColor(src[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 6)
    gx = cv2.Sobel(grey, -1, 1, 0)
    gy = cv2.Sobel(grey, -1, 0, 1)
    sample = solid & garment[window] & (np.abs(gx) > 2 * np.abs(gy))
    slope = np.clip(-np.sum(gx[sample] * gy[sample]) / max(1, np.sum(gx[sample] ** 2)), -0.5, 0.5)
    y, x = np.indices(target.shape)
    columns = np.rint(x - slope * y).astype(np.int32)
    columns -= columns.min()
    visible = np.zeros((len(y), columns.max() + 1), bool)
    known = solid & (cv2.distanceTransform((~inner).astype(np.uint8), cv2.DIST_L2, 5) > 3)
    visible[y, columns] = known
    rows = np.arange(len(y))[:, None]
    ends = (np.maximum.accumulate(np.where(visible, rows, -2 * h), axis=0),
            np.minimum.accumulate(np.where(visible, rows, 2 * h)[::-1], axis=0)[::-1])
    total, weight = np.zeros_like(delta), np.zeros(target.shape, np.float32)
    for sign, edge in zip((-1, 1), ends):
        donor_y = (edge[y, columns] + sign * 6).astype(np.float32)
        donor_x = (x + slope * (donor_y - y)).astype(np.float32)
        length = np.abs(donor_y - y)
        share = np.exp(-(length / radius) ** 2) * np.minimum(1, (distance + 15) / (length + 1)) ** 2
        share *= cv2.remap(known.astype(np.uint8), donor_x, donor_y, cv2.INTER_NEAREST)
        total += cv2.remap(src[..., :3], donor_x, donor_y, cv2.INTER_LINEAR) * share[..., None]
        weight += share
    weight = cv2.GaussianBlur(weight, (0, 0), 2)
    folds = cv2.GaussianBlur(total, (0, 0), 2) / np.maximum(weight[..., None], 1e-6)
    mix = np.minimum(weight, 1) * np.minimum(distance / 10, 1)
    out[inner, :3] = np.rint(matched[inner] + (folds[inner] - matched[inner]) * mix[inner, None]).astype(np.uint8)


def fill(rgba, hole, source, align_tiers=False):
    """The garment behind `hole` (bool): RGBA for the whole image, the picture's own pixels outside the hole. `source`
    (bool) is where the garment's patches may come from (opaque, outside the hole)."""
    out = rgba.copy()
    if not hole.any():
        return out
    h, w = hole.shape
    ys, xs = np.nonzero(hole)
    pad = SEARCH + PATCH
    if align_tiers:
        # The donor must fit the wider backing and match the tier beside the bag.
        # A narrow search forces the copy down the skirt to the next row of ruffles.
        pad = max(pad, int(np.ptp(xs)) + SEARCH // 2)
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    window = (slice(y0, y1), slice(x0, x1))
    part, here, region = hole[window], rgba[window], out[window]
    # the silhouette behind the accessory: the figure's alpha interpolated across the hole, a smooth edge at its half
    level = pushpull(here[..., 3:4].astype(np.float32) / 255.0, ~part)[..., 0]
    behind = part & (level >= 0.5)
    soft = cv2.GaussianBlur(np.where(part, behind, here[..., 3] >= 128).astype(np.float32), (0, 0), 0.7)
    region[..., 3][part] = np.clip(np.round(soft[part] * 255), 0, 255).astype(np.uint8)
    # nothing behind the accessory's soft edge onto the background: its layer draws those pixels whole
    region[..., 3][part & (here[..., 3] < SOLID)] = 0
    figure = cv2.erode((here[..., 3] >= 128).astype(np.uint8), np.ones((2 * EDGE + 1, 2 * EDGE + 1), np.uint8)) > 0
    source = source[window] & ~part & figure
    # colours where no patch reaches, and under the soft edge
    smooth = pushpull(here[..., :3].astype(np.float32), source if source.any() else ~part)
    region[..., :3][part] = np.clip(smooth[part], 0, 255).astype(np.uint8)
    if not behind.any() or not source.any():
        return out
    wh, ww = part.shape
    step = max(1, math.ceil(math.sqrt(behind.sum() / COARSE)))
    rgb = here[..., :3]
    # the wide parts: the garment beside them copied whole
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WIDE + 1, 2 * WIDE + 1))
    wide = cv2.morphologyEx(behind.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0
    # each wide part with the hole round it up to the hole's edge, so that its copy meets the drawing there (wide parts
    # that meet are one piece: one copy, no seam between them)
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WIDE + 5, 2 * WIDE + 5))
    grown = (cv2.dilate(wide.astype(np.uint8), reach) > 0) & behind
    count, parts = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    rest = behind.copy()
    for i in np.unique(parts[wide]):
        piece = parts == i
        whole = copy_whole(rgb, here[..., 3], piece, source, max(2, step), align_tiers)
        if whole is None:
            continue
        colours, copied, _ = whole
        py, px = np.nonzero(piece)
        region[..., :3][py[copied], px[copied]] = colours[copied]
        rest[py[copied], px[copied]] = False
    if not rest.any():
        return out
    # the narrow rest: an exemplar fill, round what is drawn and what was copied
    ch, cw = -(-wh // step), -(-ww // step)
    pad_to = ((0, ch * step - wh), (0, cw * step - ww))
    current = np.where(behind[..., None] & ~rest[..., None], region[..., :3], rgb)
    small = cv2.resize(np.pad(current, pad_to + ((0, 0),), mode="edge"), (cw, ch), interpolation=cv2.INTER_AREA)
    blocks = lambda mask: np.pad(mask, pad_to).reshape(ch, step, cw, step)  # noqa: E731
    target = blocks(rest).any(axis=(1, 3))
    known_source = blocks(source).all(axis=(1, 3)) & ~target
    patch = max(7, int(round(PATCH / step)) | 1)
    filled, origin = exemplar(small.astype(np.float32), target, known_source, patch, max(patch, SEARCH // step))
    # back at full resolution: each pixel copies the pixel its coarse pixel came from (the same place within it)
    ty, tx = np.nonzero(rest)
    o = origin[ty // step, tx // step]
    sy, sx = o[:, 0] * step + ty % step, o[:, 1] * step + tx % step
    good = (o[:, 0] >= 0) & (sy < wh) & (sx < ww)
    sy, sx = np.where(good, sy, 0), np.where(good, sx, 0)
    good &= source[sy, sx]
    coarse = cv2.resize(np.clip(filled, 0, 255).astype(np.uint8), (cw * step, ch * step), interpolation=cv2.INTER_LINEAR)
    sy, sx = np.where(good, sy, ty), np.where(good, sx, tx)
    copied = np.where(good[:, None], rgb[sy, sx], coarse[ty, tx])
    region[..., :3][ty, tx] = copied
    voted = vote(np.where(behind[..., None], region[..., :3], rgb), rest, sy, sx, ty, tx, VOTE)
    region[..., :3][ty, tx] = np.where(good[:, None], voted, copied)
    return out
