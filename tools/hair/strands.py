"""Strand layers for build-layers.py (--strand-layers): a held lock moves on its own layer over the picture filled in
behind it, instead of bending the base.

Without them a held strand bends the base like its sway does: the hair round the strand's line follows it with a soft
fall-off, so a pull of more than its sway (`max`) stretches the lock's edges and drags what lies beside it. For each
strand with a `reach` (M11 rigs) two layers are cut:

  strand_<i>.png        the lock: the hair along the strand (the hair mask inside LOCK sigmas of its line, below its
                        root, without the mask's pixels that are the colour of what lies round it rather than of the
                        hair along the line, and the hair hanging from it out to REGION sigmas, told by colour: thin
                        hairs and tips that end inside that corridor; a garment's line that goes on beyond it stays).
                        Each pixel of its outline (and of the line drawn round it, its thin hairs) holds only its share
                        of hair over what lies behind it; where the hair goes on beside it (above its root, beside a
                        cut through more hair) it fades in over FEATHER px
  strand_<i>_under.png  the plate: the drawn base (base_bare, base_cloth or base) as far round the strand as it bends it
                        (REGION sigmas), with the lock taken out: what lies behind it filled in smoothly from the opaque
                        pixels round it that are not hair, transparent where the lock hung over the background. Where
                        the lock fades into the hair that stays the plate keeps the drawn hair, and it goes over to the
                        fill only under the opaque lock (MARGIN px further on, over RAMP px), so no cut edge of the lock
                        lies over anything but the drawing however a display filters the two textures (a light line of
                        the fill under the cut otherwise). The runtime draws it in place of the base there, so the
                        base's own bend never shows

At rest the lock over the plate draws the drawing (within TOLERANCE levels where the colours were told apart).

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
EDGE = 3  # px the lock's outline may lie outside the hair mask (and inside it: the mask's edge is not the drawing's)
OUTLINE = 2  # px round the lock and its thin hairs that go with it too: the line drawn round hair, its light rim
FEATHER = 6  # px over which the lock fades in from the hair that stays (above its root, beside a cut through the hair)
MARGIN = 8  # px beyond the fade where the plate still keeps the drawn hair: the reach of a display's mip levels
RAMP = 10  # px over which the plate then goes over to what lies behind the lock
TOLERANCE = 4  # levels the lock over the plate may differ from the drawing where the colours were told apart
GAP = 4  # px apart that dark pixels by the lock are still one piece (a hair's tip, a garment's lines)
HAIR = 0.35  # share of the hair's colour (over what lies round it) that a pixel of the hair mask needs to be the lock
CENTRE = 0.5  # sigmas from the strand's line: the lock's own colour
THIN = 0.2  # share of the lock's hair that makes a pixel by the lock part of a thin hair hanging from it
OFF = 22.0  # RGB distance a pixel of hair over what lies behind may lie off the way from the one to the other
CONTRAST = 24.0  # RGB distance below which the lock's hair and what lies behind cannot be told apart
MIN_PIXELS = 200
SPECK = 0.1  # of sigma squared: the most a part the lock closes round may cover and still go with it
PALETTE = 4  # colours of the lock and of what lies round it
LAB_MARGIN = 6.0  # Lab units a thin hair pixel is nearer the lock's colours than the others (and a pixel the fill uses
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


def hair_like(lab, core, other, candidates, margin=LAB_MARGIN):
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


def pushpull(values, known):
    """Values (H x W x C float32) for the pixels that are not `known`, smoothly from the known ones round them."""
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


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def distance_to(mask):
    """Distance (px) of every pixel to the nearest pixel of mask (large where it is empty)."""
    if not mask.any():
        return np.full(mask.shape, 1e6, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def share_of_hair(colour, behind, hair):
    """The share of each pixel's colour (n x 3, 0..255) that is the lock's hair (n x 3) over what lies behind it (n x 3):
    its position on the way from the one to the other (0 where the two cannot be told apart), and how far the colour is
    off that way (a garment's line or a ribbon is not hair over what lies behind it)."""
    way = hair - behind
    length = (way * way).sum(axis=1)
    share = np.clip(((colour - behind) * way).sum(axis=1) / np.maximum(length, 1e-6), 0, 1)
    off = np.linalg.norm(colour - behind - share[:, None] * way, axis=1)
    return np.where(length > CONTRAST * CONTRAST, share, 0), off


def least_share(colour, behind):
    """The least share of the lock with which the colour (within TOLERANCE) is a colour over `behind` (n x 3)."""
    gap = colour - behind
    gap = np.sign(gap) * np.maximum(np.abs(gap) - TOLERANCE, 0)
    near = behind + gap
    darker = np.where(near < behind, (behind - near) / np.maximum(behind, 1e-6), 0)
    lighter = np.where(near > behind, (near - behind) / np.maximum(255 - behind, 1e-6), 0)
    return np.maximum(darker, lighter).max(axis=1), near


def cut_one(here, distance, along, parsed, sigma, line):
    """(lock RGBA uint8, plate RGBA uint8, region, unsure) over one strand's window, or None when too little hair.

    here: the drawn base (RGBA uint8); distance, along: corridor(); parsed: the hair mask (bool); line: the strand's line
    from a tenth of its length drawn into the window (uint8), which the lock hangs together with."""
    alpha = here[..., 3]
    opaque, solid = alpha > 0, alpha > 200
    rgb = np.ascontiguousarray(here[..., :3])
    lock = (distance <= LOCK * sigma) & (along > ROOT) & parsed & opaque
    # the hair mask's pixels that are not the colour of the hair along the strand's line but of what lies round it (a
    # coat's weave the parser took for hair) are not the lock; specks of the lock's own light inside it stay in it
    centre = lock & solid & (distance <= CENTRE * sigma)
    plain = solid & ~parsed
    if centre.sum() >= PALETTE and plain.sum() >= MIN_PIXELS:
        tone = pushpull(rgb.astype(np.float32), centre)
        around = pushpull(rgb.astype(np.float32), plain)
        share, _ = share_of_hair(rgb[lock].astype(np.float32), around[lock], tone[lock])
        told = np.linalg.norm(tone[lock] - around[lock], axis=1) > CONTRAST
        coloured = lock.copy()
        coloured[lock] = ~told | (share >= HAIR)
        count, labels, stats, _ = cv2.connectedComponentsWithStats((~coloured).astype(np.uint8), connectivity=4)
        enclosed = np.zeros(lock.shape, bool)
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] <= SPECK * sigma * sigma:
                enclosed |= labels == i
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        enclosed &= ~np.isin(labels, border)
        lock = coloured | (enclosed & lock)
    # thin hair the mask leaves out (tips, loose hairs) hangs together with the lock: told by colour
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    near = (distance <= REGION * sigma) & (along > ROOT) & opaque
    lock |= hair_like(lab, lock & solid, near & ~parsed & solid & (distance > LOCK * sigma * 0.5), near & ~lock)
    # only the hair that hangs together with the strand's line
    _, labels = cv2.connectedComponents(lock.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[(line > 0) & lock])
    lock = np.isin(labels, keep[keep > 0])
    if lock.sum() < MIN_PIXELS:
        return None
    # the hair that stays: the hair mask outside the lock that goes on into the hair round the strand (above its root,
    # beyond its corridor); bits of the mask left inside the corridor are the lock's own and are told by colour
    rest = parsed & ~lock
    _, labels = cv2.connectedComponents(rest.astype(np.uint8), connectivity=8)
    outer = np.unique(labels[rest & ~((distance <= REGION * sigma) & (along > ROOT))])
    stays = np.isin(labels, outer[outer > 0]) & rest
    # where the lock fades in from the hair that stays, and where the plate goes over to what lies behind the lock
    from_stay = distance_to(stays)
    fade = smoothstep(0, FEATHER, from_stay)
    fill = smoothstep(FEATHER + MARGIN, FEATHER + MARGIN + RAMP, from_stay)
    # the lock's body (deep in the hair mask: all of it hair) and round it the pixels told apart by colour: its
    # outline, the drawn line round it, its thin hairs, the bits of the mask in the corridor
    body = lock & parsed & (alpha == 255) & (distance_to(~lock & ~stays) > EDGE)
    # the thin hairs hanging from the lock that the mask and the palettes leave out (fine strands fanning out over the
    # skin, the tips below it): darker than what lies round them towards the lock's hair, hanging from the lock and
    # ending inside its corridor (a garment's line that runs under the lock goes on beyond it, and stays)
    corridor_in = (distance <= REGION * sigma) & (along > ROOT)
    plain = solid & ~parsed & ~lock & ~hair_like(lab, lock & solid, solid & ~parsed & ~lock, solid & ~lock, -LAB_MARGIN)
    dark = np.zeros(lock.shape, bool)
    if plain.sum() >= MIN_PIXELS:
        around = pushpull(rgb.astype(np.float32), plain)
        tone = pushpull(rgb.astype(np.float32), lock & solid)
        pick = opaque & ~lock & ~stays
        share, off = share_of_hair(rgb[pick].astype(np.float32), around[pick], tone[pick])
        dark[pick] = (share >= THIN) & (off <= OFF)
        strokes = dark.copy()  # every dark stroke in the window: what lies behind the lock is not filled from them
        dark &= corridor_in
        dark |= rest & ~stays & corridor_in  # bits of the hair mask in the corridor
        # dark pixels this close are one piece: a garment's lines and seams make one piece with the garment's
        # further lines, which goes on beyond the corridor; a hair over the skin is a piece of its own
        joined = cv2.dilate(dark.astype(np.uint8), disk(GAP // 2)) > 0
        _, labels = cv2.connectedComponents(joined.astype(np.uint8), connectivity=8)
        rim = (distance > REGION * sigma - 3) | (along <= ROOT) | (cv2.dilate(stays.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        rim[0, :] = rim[-1, :] = rim[:, 0] = rim[:, -1] = True
        beyond = np.unique(labels[joined & rim])
        touching = np.unique(labels[joined & (cv2.dilate(lock.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)])
        thin = np.isin(labels, np.setdiff1d(touching[touching > 0], beyond)) & dark
        # and the lock's own outline, however its strokes go on
        thin |= dark & (cv2.dilate(lock.astype(np.uint8), disk(EDGE)) > 0)
    else:
        thin = np.zeros(lock.shape, bool)
        strokes = dark
    # what goes with the lock: its hair, the thin hairs hanging from it and the line drawn round them (each pixel told
    # apart by colour from what lies behind it); every other pixel stays as drawn, a stray hair whole
    zone = (cv2.dilate((lock | thin).astype(np.uint8), disk(OUTLINE)) > 0) & ~stays & (along > ROOT * 0.5)
    zone |= body
    # and what the lock closes round (a speck of its shade between its strands)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~zone).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    for i in range(1, count):
        if i not in border and stats[i, cv2.CC_STAT_AREA] <= SPECK * sigma * sigma:
            zone |= (labels == i) & ~stays
    # what lies behind: colours from the solid pixels round the zone that are not hair (the hair mask, the dark strokes
    # left as they are), so the fill meets the drawing round it; the figure's outline interpolated across it
    known = solid & ~zone & ~parsed & ~strokes
    if known.sum() < MIN_PIXELS:
        known = solid & ~zone
    figure = cv2.inpaint(np.ascontiguousarray(alpha), zone.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA) >= 128
    figure = np.where(zone, figure, alpha >= 128)
    behind = np.clip(pushpull(rgb.astype(np.float32), known), 0, 255)
    hair_colour = pushpull(rgb.astype(np.float32), body if body.sum() >= PALETTE else lock & solid)
    # premultiplied (0..255): the drawing, what lies behind (opaque on the figure, nothing off it), the plate, the lock
    a = alpha.astype(np.float32)[..., None] / 255
    drawing = np.concatenate([rgb.astype(np.float32) * a, a * 255], axis=2)
    under = np.concatenate([behind * figure[..., None], figure[..., None] * 255.0], axis=2)
    plate = drawing.copy()
    out = np.zeros_like(drawing)
    # the lock's body: all of it, faded in from the hair that stays (where the plate keeps the drawing)
    w = fill[body][:, None]
    plate[body] = drawing[body] * (1 - w) + under[body] * w
    out[body] = drawing[body] * fade[body][:, None]
    # round it: each pixel's share of the lock's hair (s), the rest of it what lies behind: on the figure, or as far as
    # the drawing covers the pixel (its own antialiased outline over the background stays that, in the colour behind)
    edge = zone & ~body
    w = fill[edge][:, None]
    c, ca = drawing[edge], a[edge]
    colour = rgb[edge].astype(np.float32)
    s, _ = share_of_hair(colour, behind[edge], hair_colour[edge])
    hair = (w[:, 0] > 0) & (ca[:, 0] > 0)
    clear = ca[:, 0] < 0.999
    under_edge = np.concatenate([behind[edge] * figure[edge][:, None], 255 * figure[edge][:, None]], axis=1)
    edge_plate = c * (1 - w) + under_edge * w
    # a pixel the drawing leaves partly clear (the outline over the background) is split as it is: the lock takes its
    # hair share of it, the plate keeps the rest
    part = (w[:, 0] * s)[:, None]
    edge_plate[clear] = ((1 - part) * c / np.maximum(1 - part * ca, 1e-6))[clear]
    p = edge_plate[:, 3:4] / 255
    # over an opaque plate the lock is the pixel's hair share against it (at least what keeps the drawing within
    # TOLERANCE); over a plate that is partly clear it takes what the plate leaves of the pixel
    least, near = least_share(colour, edge_plate[:, :3])
    share = np.clip(np.maximum(s, least), 0, 1)[:, None]
    on_lock = np.concatenate([np.clip(near - (1 - share) * edge_plate[:, :3], 0, 255 * share), 255 * share], axis=1)
    taken = np.clip((ca - p) / np.maximum(1 - p, 1e-6), 0, 1)
    off_lock = np.concatenate([np.clip(c[:, :3] - (1 - taken) * edge_plate[:, :3], 0, 255 * taken), 255 * taken], axis=1)
    edge_lock = np.where(p >= 0.999, on_lock, off_lock)
    edge_lock[clear] = (part * c)[clear]
    # a pixel that is not the lock's hair stays as drawn
    plate[edge] = np.where(hair[:, None], edge_plate, c)
    out[edge] = np.where(hair[:, None], edge_lock, 0)
    # the share of the lock's outline that borders the background: there the drawing does not tell how far the
    # figure goes on behind the lock (beside a dress's outline the fill may reach past it, or a notch may open)
    shown = out[..., 3] > 0
    ring = (cv2.dilate(shown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~shown
    unsure = float((ring & ~opaque).sum()) / float(max(1, ring.sum()))
    region = ((distance <= REGION * sigma) & (along >= ROOT * 0.5)) | zone
    return straight(out), straight(plate), region, unsure


def straight(premultiplied):
    """RGBA uint8 with straight colours from premultiplied float RGBA (0..255)."""
    a = premultiplied[..., 3:4]
    colour = np.where(a > 0, premultiplied[..., :3] * 255 / np.maximum(a, 1e-6), 0)
    return np.clip(np.rint(np.concatenate([colour, a], axis=2)), 0, 255).astype(np.uint8)


def cut(drawn, hair, rig):
    """The strand layers of the strands with a reach: ({name: (image, rect)}, {strand name: {lock, under}}, [notes]).

    drawn: the base the runtime draws (RGBA, source size); hair: 8-bit hair mask (source size, white is hair)."""
    h, w = hair.shape
    layers, data, notes = {}, {}, []
    for index, strand in enumerate(rig.get("strands", [])):
        if not strand.get("reach"):
            continue
        nodes = [(float(x), float(y)) for x, y in strand["nodes"]]
        sigma = float(strand["sigma"])
        pad = int(np.ceil(REGION * sigma)) + OUTLINE + EDGE + 2
        xs, ys = [p[0] for p in nodes], [p[1] for p in nodes]
        box = (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad), min(w, int(max(xs)) + pad + 1), min(h, int(max(ys)) + pad + 1))
        x0, y0, x1, y1 = box
        distance, along = corridor(nodes, box)
        window = (slice(y0, y1), slice(x0, x1))
        line = np.zeros(distance.shape, np.uint8)
        start = max(1, int(np.ceil(0.1 * (len(nodes) - 1))))
        cv2.polylines(line, [np.rint(np.asarray(nodes[start - 1 :]) - (x0, y0)).astype(np.int32)], False, 1, 3)
        result = cut_one(drawn[window], distance, along, hair[window] >= 128, sigma, line)
        if result is None:
            notes.append(f"{strand['name']}: too little hair along the strand")
            continue
        lock_image, plate, region, unsure = result
        rows, cols = np.nonzero(region)
        ry0, ry1, rx0, rx1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        lrows, lcols = np.nonzero(lock_image[..., 3] > 0)
        ly0, ly1, lx0, lx1 = lrows.min(), lrows.max() + 1, lcols.min(), lcols.max() + 1
        name = f"strand_{index}"
        layers[name] = (np.ascontiguousarray(lock_image[ly0:ly1, lx0:lx1]), [int(x0 + lx0), int(y0 + ly0), int(lx1 - lx0), int(ly1 - ly0)])
        layers[f"{name}_under"] = (np.ascontiguousarray(plate[ry0:ry1, rx0:rx1]), [int(x0 + rx0), int(y0 + ry0), int(rx1 - rx0), int(ry1 - ry0)])
        data[strand["name"]] = {"lock": name, "under": f"{name}_under", "unsure": round(unsure, 4)}
    return layers, data, notes
