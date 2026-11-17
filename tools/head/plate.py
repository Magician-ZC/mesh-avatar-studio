"""Head layers from a clean plate (build-layers.py --head-plate): the hair lifted off a picture of the same head without it.

`tools/head/layers.py` tells hair from skin by colour, which fails where hair hangs over the eyes and brows and cannot
know what the forehead under a fringe looks like. A clean plate removes both problems: a picture of the same head, pixel
aligned with the source, in which the hair in front of the face is gone and the face is whole (an image model's layer
decomposition gives one; so does an artist). With it

  alpha   comes from the source itself: source = alpha * hair + (1 - alpha) * plate, solved per pixel against the local
          hair colour (difference matting), inside the footprint the plate's maker gave for the hair;
  hair    is the source's own colour with the plate taken out of its soft edges, so hair over plate is the source again;
  under   is the plate where hair covered it, and the source everywhere else.

The resting picture is therefore the source, to rounding, and nothing of the plate's own rendering of the hair is used.

Only the hair that a head turn can uncover becomes a layer: inside the face field's reach. The rest of the hair stays
in the base, where it lies exactly under the edge of the layer.
"""

import cv2
import numpy as np

REACH = 30  # px the hair layer reaches beyond the moving face (as in layers.py)
FADE = 8  # of which the outermost pixels fade out, onto hair that is the base's own
FOOTPRINT_GROW = 9  # px the given hair footprint is grown: its edge is where the maker drew it, not where the source has it
UNDER_GROW = 5  # px the plate reaches beyond the hair it was under (the hair's antialiased rim)
UNDER_FEATHER = 4  # px over which the plate hands over to the source
MIN_CONTRAST = 28.0  # hair and plate closer than this (RGB distance) cannot be told apart: the footprint decides
SOLID = 0.9  # alpha from which a pixel counts as hair entirely
SAME = 16.0  # source and plate closer than this (RGB distance) are the same pixel: it is the plate
NOISE = 0.08  # alpha below this is plate noise
KEEP = 8  # 8-bit levels: where nothing is lifted, the plate may stand in for the source only this close to it
EDGE = 16  # 8-bit levels: a soft hair pixel whose hair over the layer below is further from the source is left to the source
LIFT = 0.1  # a pixel the plate would hide is hair if its source lies at least this far from the plate towards the hair colour
PLAUSIBLE = 90  # RGB distance from the local hair colour within which a hair pixel's colour is one hair can have
EXPLAIN = 2  # px: by the eyes, a pixel the plate shows this close by is the plate drawn a little off, not hair
ISLAND = 8  # px around an eye's zone from which hair there must hang down (or be no hair)
RIM = 4  # px: how far into the hair by an eye the plate may explain a pixel
BESIDE = 1.5  # px (blur sigma): the solid hair whose colour a soft hair pixel by the eyes takes


def _blur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)


def _local_colour(rgb, where, sigma):
    """The mean colour of `where` around each pixel (normalised blur), and how much of it there is."""
    weight = where.astype(np.float32)
    total = _blur(weight, sigma)
    colour = _blur(rgb * weight[..., None], sigma) / np.maximum(total, 1e-4)[..., None]
    return colour, total


def align(source_rgb, plate_rgb, region):
    """The plate shifted so that it lies on the source inside `region` (a mask of features both show, e.g. the lower
    face): translation only, to a tenth of a pixel. Returns (plate, (dx, dy), correlation)."""
    ys, xs = np.where(region)
    if len(xs) < 400:
        return plate_rgb, (0.0, 0.0), 0.0
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    a = cv2.cvtColor(source_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(plate_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(b, a, window)
    if abs(dx) > 12 or abs(dy) > 12:  # not a small misregistration: the plate is of something else there
        return plate_rgb, (0.0, 0.0), float(response)
    shift = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(plate_rgb, shift, (plate_rgb.shape[1], plate_rgb.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return moved, (float(dx), float(dy)), float(response)


BACKDROP = 238  # a plate pixel at least this light in every channel may be the plate's own (white) backdrop
BACKDROP_MIN = 400  # px a light region must have, and it must touch the plate's border, to count as backdrop


def backdrop(plate_rgb, valid):
    """The plate's own backdrop: light regions that reach the border of where the plate is defined. Hair in front of it
    was in front of nothing, so it is not lifted (the plate has nothing to show there)."""
    light = (plate_rgb.min(-1) >= BACKDROP) & valid
    count, labels, stats, _ = cv2.connectedComponentsWithStats(light.astype(np.uint8))
    edge = valid & ~(cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    touching = np.unique(labels[edge & light])
    keep = [i for i in touching if i != 0 and stats[i, cv2.CC_STAT_AREA] >= BACKDROP_MIN]
    return np.isin(labels, keep)


def explained(src, plate, where, hair, radius=EXPLAIN):
    """The pixels of `where` whose source colour the plate has within `radius` px (closer than SAME): the plate's own
    drawing a little off, not hair. Only plate pixels unlike the local hair colour `hair` explain anything (a dark lash
    line explains no dark lock lying across it). src, plate and hair are float RGB."""
    found = np.zeros(where.shape, bool)
    ys, xs = np.nonzero(where)
    if len(xs) == 0:
        return found
    h, w = where.shape
    y0, y1, x0, x1 = max(0, ys.min() - radius), min(h, ys.max() + radius + 1), max(0, xs.min() - radius), min(w, xs.max() + radius + 1)
    box, near_hair = src[y0:y1, x0:x1], hair[y0:y1, x0:x1]
    padded = cv2.copyMakeBorder(plate[y0:y1, x0:x1], radius, radius, radius, radius, cv2.BORDER_REPLICATE)
    nearest = np.full(box.shape[:2], np.inf, np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                moved = padded[radius + dy : radius + dy + box.shape[0], radius + dx : radius + dx + box.shape[1]]
                distance = np.linalg.norm(box - moved, axis=-1)
                distance[np.linalg.norm(moved - near_hair, axis=-1) <= MIN_CONTRAST] = np.inf
                nearest = np.minimum(nearest, distance)
    found[y0:y1, x0:x1] = where[y0:y1, x0:x1] & (nearest < SAME)
    return found


def matte(source_rgb, plate_rgb, footprint, valid, features=None, eyes=None):
    """Hair alpha (0..1) of the source over the plate inside the grown footprint, and the hair's local colour.

    footprint: where the plate's maker says the hair is (0..1). valid: where the plate is defined.
    features: the eye regions and the brow, where only hair on the footprint itself counts (grown by two pixels) -
    lashes and brows are as dark as hair, and a plate that draws them a pixel off would otherwise leave slivers of them
    on the hair layer, which stay behind when the face turns. eyes: the part of features around the eyes (default all).
    Returns (alpha, hair colour, reason): alpha is None with the reason when the plate does not allow it.
    """
    src = source_rgb.astype(np.float32)
    plate = plate_rgb.astype(np.float32)
    grow = 2 * FOOTPRINT_GROW + 1
    near = cv2.dilate((footprint > 0.05).astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
    # not in front of the plate's own backdrop (grown a little: its edge is soft)
    near &= valid & ~(cv2.dilate(backdrop(plate_rgb, valid).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    # the hair's own colour near each pixel: source pixels deep in the footprint that differ clearly from the plate
    apart = np.linalg.norm(src - plate, axis=-1)
    core = (cv2.erode((footprint > 0.9).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (apart > MIN_CONTRAST) & valid
    if int(core.sum()) < 200:
        return None, None, "too little hair in the footprint differs from the plate"
    hair, amount = _local_colour(src, core, 14)
    far, _ = _local_colour(src, core, 60)  # where no core hair is near, the wider neighbourhood's
    hair = np.where((amount > 0.02)[..., None], hair, far)
    direction = hair - plate
    contrast = np.linalg.norm(direction, axis=-1)
    solved = np.clip(np.sum((src - plate) * direction, axis=-1) / np.maximum(contrast * contrast, 1.0), 0, 1)
    # where hair and plate have the same colour the equation says nothing: the footprint decides
    trust = np.clip((contrast - MIN_CONTRAST) / MIN_CONTRAST, 0, 1)
    alpha = solved * trust + footprint * (1 - trust)
    alpha[~near] = 0
    alpha[alpha < NOISE] = 0
    # a pixel that is the plate is not hair, whatever the footprint says (skin between strands, a footprint drawn too wide)
    alpha[apart < SAME] = 0
    # inside the footprint a pixel that is clearly not the plate is hair entirely: highlights and lighter strands solve to
    # less than 1 against the local (darker) hair colour, and would let the plate shine through
    inside = cv2.erode((footprint > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    alpha[inside & near & (apart >= SAME) & ((solved > 0.3) | (trust < 0.5))] = 1
    alpha[alpha > SOLID] = 1
    if features is not None:
        alpha[features & ~(cv2.dilate((footprint > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)] = 0
        eyes = features if eyes is None else eyes
        # nor, at the hair's edge by an eye, a pixel the plate shows a pixel or two away: the plate drew the eyelid's band
        # or the lash line a little off, and lifted, that band would stay on the hair layer when the eye closes or the
        # face turns (the hair's own soft rim is left alone)
        rim = eyes & (alpha >= 0.5) & (cv2.dilate((alpha <= 0).astype(np.uint8), np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8)) > 0)
        alpha[explained(src, plate, rim, hair)] = 0
        # and what is left of that band ends in a clean edge: no single pixels sticking out of the hair
        thin = (alpha > 0) & ~(cv2.morphologyEx((alpha > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0)
        alpha[thin & rim] = 0
    # specks: only what hangs together with solid hair - by the eyes, with solid hair beyond them (what is left of the
    # eyelid's band once the plate has explained the rest is no hair hanging over the eye)
    _, labels = cv2.connectedComponents((alpha > 0).astype(np.uint8))
    anchor = alpha >= 1
    if features is not None:
        anchor &= ~(cv2.dilate(eyes.astype(np.uint8), np.ones((2 * ISLAND + 1, 2 * ISLAND + 1), np.uint8)) > 0)
    kept = np.unique(labels[anchor])
    alpha[~np.isin(labels, kept[kept != 0])] = 0
    return alpha.astype(np.float32), hair, None


def lift(source, plate_rgb, alpha, hair_colour, moving, valid, features=None):
    """The hair layer and the under layer of `source` (RGBA) for hair alpha `alpha` over `plate_rgb`.

    moving: where a head turn slides the picture (the face field > 0). Returns (hair RGBA, under RGBA, under weight).
    hair_colour is the local hair colour from matte(). Two nested zones around the moving face keep the resting picture
    the source: the hair layer fades in over the outermost FADE px of its reach, where the layer below is still the
    source itself (hair on its own hair); only further in, where the hair layer is complete, does the layer below
    become the plate. Solid hair pixels reproduce the source exactly, soft ones as closely as a colour hair can have
    allows. The plate never hides what the matte did not lift: such a pixel is lifted after all where it is hair, and
    keeps the source below where it is not, and by the eyes (features: where only hair on the footprint is lifted, and a
    lash put on the hair layer would stay behind when the face turns).
    """
    src = source[..., :3].astype(np.float32)
    plate = plate_rgb.astype(np.float32)
    size = 2 * REACH + 1
    support = cv2.dilate(moving.astype(np.uint8), np.ones((size, size), np.uint8))
    depth = cv2.distanceTransform(support, cv2.DIST_L2, 5)  # px inside the reach
    margin = np.clip(depth / FADE, 0, 1)
    a = alpha * margin
    # the plate under the hair (and a rim around it, for the hair's antialiased edge), from where the hair layer is complete
    grow = 2 * UNDER_GROW + 1
    covered = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((grow, grow), np.uint8)).astype(np.float32)
    rim = np.clip(_blur(covered, UNDER_FEATHER / 2), 0, 1)
    rim[alpha > 0] = 1
    weight = rim * np.clip((depth - FADE - 2) / UNDER_FEATHER, 0, 1)
    # never the plate's own backdrop, nor anything outside the plate
    weight[(cv2.dilate(backdrop(plate_rgb, valid).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) | ~valid] = 0
    # where nothing is lifted, the plate stands in for the source only if it is the source to within KEEP: a strand tip the
    # matte left out (a speck of soft alpha, or outside the footprint) or a lash would vanish from the resting picture.
    # By the eyes (the features, where only hair on the footprint is lifted) the source stays below; elsewhere such a
    # pixel is lifted after all, below
    near_eyes = features if features is not None else np.zeros(alpha.shape, bool)
    hidden = (alpha <= 0) & (weight > 0) & (np.abs(src - plate).max(-1) > KEEP)
    weight[hidden & near_eyes] = 0
    hidden &= ~near_eyes
    under = source.copy()
    under_rgb = src * (1 - weight[..., None]) + plate * weight[..., None]
    under[..., :3] = np.clip(np.rint(under_rgb), 0, 255).astype(np.uint8)
    # the hair's colour: the source where the pixel is hair entirely; in soft pixels the source with the layer below taken
    # out, so that hair over under is the source again - unless that gives a colour no hair has (a nearly transparent
    # pixel does, and it would show once the face has moved): then the hair's local colour
    under_drawn = under[..., :3].astype(np.float32)
    if np.any(hidden):
        # on the line from the layer below to the local hair colour, the point nearest to the source
        towards = hair_colour - under_drawn
        t = np.clip(np.sum((src - under_drawn) * towards, axis=-1) / np.maximum(np.sum(towards * towards, axis=-1), 1.0), 0, 1)
        # one that is no hair (no nearer the hair colour than the plate: skin detail the plate lacks) keeps the source below
        skin = hidden & (t < LIFT)
        weight = np.where(skin, 0.0, weight)
        under[skin, :3] = source[skin, :3]
        under_drawn = under[..., :3].astype(np.float32)
        a = np.where(hidden & ~skin, t * margin, a)
    exact = (src - (1 - a[..., None]) * under_drawn) / np.maximum(a[..., None], 0.05)
    plausible = (np.linalg.norm(exact - hair_colour, axis=-1) < PLAUSIBLE) & (exact.min(-1) >= 0) & (exact.max(-1) <= 255)
    colour = np.where((a >= 0.999)[..., None], src, np.where(plausible[..., None], exact, hair_colour))
    # where the hair's local colour stands in, its alpha is solved again against that colour (the point on the line from
    # the layer below to it nearest to the source), so hair over the layer below stays as close to the source as that
    # colour allows; over the eyes never more hair than the matte found, or it would stay behind when the face turns
    swapped = (a > 0) & (a < 0.999) & ~plausible
    if np.any(swapped):
        # the plausible colour nearest to the exact one: towards the local hair colour until within PLAUSIBLE of it
        off_hair = exact - hair_colour
        reach = np.linalg.norm(off_hair, axis=-1, keepdims=True)
        nearest = np.clip(hair_colour + off_hair * np.minimum(1.0, PLAUSIBLE / np.maximum(reach, 1e-6)), 0, 255)
        colour = np.where(swapped[..., None], nearest, colour)
        towards = nearest - under_drawn
        refit = np.clip(np.sum((src - under_drawn) * towards, axis=-1) / np.maximum(np.sum(towards * towards, axis=-1), 1.0), 0, 1)
        if features is not None:
            refit = np.where(features, np.minimum(refit, a), refit)
        a = np.where(swapped, refit, a)
    # by the eyes the hair's soft edge takes the colour of the solid hair beside it, its alpha solved again against that
    # colour (never more than before): solved exactly, the edge carries what lies under it - a light rim over the dark
    # lash line - which shows once the eye closes or the face turns; and no eye colour goes onto the hair layer
    edge = near_eyes & (a > 0) & (a < 0.999)
    if np.any(edge):
        local, amount = _local_colour(src, (a >= 0.999) & (alpha > 0), BESIDE)
        local = np.where((amount > 0.01)[..., None], local, hair_colour)
        towards = local - under_drawn
        refit = np.minimum(np.clip(np.sum((src - under_drawn) * towards, axis=-1) / np.maximum(np.sum(towards * towards, axis=-1), 1.0), 0, 1), a)
        # where that hair colour cannot give the source back, the pixel keeps its colour
        fits = np.abs(under_drawn * (1 - refit[..., None]) + local * refit[..., None] - src).max(-1) <= EDGE
        edge &= fits
        colour = np.where(edge[..., None], local, colour)
        a = np.where(edge, refit, a)
    # by the eyes, a soft pixel that no hair colour and alpha give back (a highlight lighter than both sides, another hue)
    # is left to the source: no hair there, the source below - a speck that moves with the face rather than a wrong
    # colour in the resting picture next to the eye
    drawn = under_drawn * (1 - a[..., None]) + colour * a[..., None]
    off = near_eyes & (alpha > 0) & (a < 0.999) & (np.abs(drawn - src).max(-1) > EDGE)
    if np.any(off):
        a = np.where(off, 0.0, a)
        weight = np.where(off, 0.0, weight)
        under[off, :3] = source[off, :3]
    hair = source.copy()
    hair[..., :3] = np.clip(np.rint(colour), 0, 255).astype(np.uint8)
    hair[..., 3] = np.clip(np.rint(source[..., 3].astype(np.float32) * a), 0, 255).astype(np.uint8)
    return hair, under, weight


def rest_error(source, hair, under):
    """How far hair over under is from the source where the hair layer is drawn (max and mean, 8-bit levels)."""
    a = hair[..., 3:4].astype(np.float32) / 255 * (source[..., 3:4] > 0)
    drawn = under[..., :3].astype(np.float32) * (1 - a) + hair[..., :3].astype(np.float32) * a
    error = np.abs(drawn - source[..., :3].astype(np.float32)).max(-1)
    where = (hair[..., 3] > 0) & (source[..., 3] > 200)
    if not np.any(where):
        return 0.0, 0.0
    return float(error[where].max()), float(error[where].mean())


ALIGN_LIMIT = 4.0  # px the plate may lie off the source before it is not a plate of this picture
REST_LIMIT = 6.0  # mean 8-bit error of hair over under against the source, over the moving face
MIN_LIFTED = 0.01  # share of the moving face the hair layer must cover to be worth a layer


def cut(source, rig, plate_rgba, footprint, frame_of, field_of, moving_floor, outside):
    """The head layers of `source` (RGBA) from a clean plate, or the reason they cannot be made.

    plate_rgba: the picture without the hair in front of the face, source size; its alpha says where it is defined.
    footprint: where that hair is according to the plate's maker (8-bit, source size).
    frame_of, field_of, moving_floor: the face frame, the face field and its threshold of layers.py (the same field).
    outside: what is neither hair nor face (hand, accessories, limbs, garments).
    Returns ({"field", "hair", "under", "weight", "report"}, None) or (None, reason).
    """
    shape = source.shape[:2]
    frame = frame_of(rig)
    if frame is None:
        return None, "the eyes and the mouth are too close together to place the face"
    field = field_of(shape, frame, (source[..., 3] > 128).astype(np.float32))
    moving = field > moving_floor
    if not np.any(moving):
        return None, "the face is outside the drawing"
    valid = (plate_rgba[..., 3] > 128) & (source[..., 3] > 128) & ~outside
    if not np.any(valid & moving):
        return None, "the plate does not cover the face"
    plate_rgb = np.ascontiguousarray(plate_rgba[..., :3])
    # register on the lower face (nose to chin), which hair in front of the face does not cover
    nose, mouth = rig["face"]["nose"], rig["mouth"]
    low = np.zeros(shape, bool)
    half = max(40, int(round(1.6 * rig["face"]["mouth"]["rx"])))
    low[max(0, int(nose["cy"] - nose["ry"])) : int(mouth["cy"] + 0.8 * half), max(0, int(mouth["cx"] - half)) : int(mouth["cx"] + half)] = True
    plate_rgb, shift, response = align(np.ascontiguousarray(source[..., :3]), plate_rgb, low & valid)
    if max(abs(shift[0]), abs(shift[1])) > ALIGN_LIMIT:
        return None, f"the plate lies {max(abs(shift[0]), abs(shift[1])):.1f} px off the picture"
    eyes = np.zeros(shape, bool)
    for eye in rig["eyes"]:
        region = np.zeros(shape, np.uint8)
        cv2.fillPoly(region, [np.array(eye["roi"], np.int32)], 1)
        eyes |= cv2.dilate(region, np.ones((9, 9), np.uint8)) > 0
    features = eyes.copy()
    brow = rig["face"].get("brow")
    if brow:
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        features |= ((xx - brow["cx"]) / (brow["rx"] + 8)) ** 2 + ((yy - brow["cy"]) / (brow["ry"] + 8)) ** 2 <= 1
    alpha, hair_colour, reason = matte(np.ascontiguousarray(source[..., :3]), plate_rgb, footprint.astype(np.float32) / 255, valid, features, eyes)
    if alpha is None:
        return None, reason
    hair, under, weight = lift(source, plate_rgb, alpha, hair_colour, moving, valid, features)
    lifted = float(((hair[..., 3] > 0) & moving).sum() / max(1, int(moving.sum())))
    if lifted < MIN_LIFTED:
        return None, "no hair in front of the face"
    a = hair[..., 3:4].astype(np.float32) / 255
    rest = under[..., :3].astype(np.float32) * (1 - a) + hair[..., :3].astype(np.float32) * a
    error = np.abs(rest - source[..., :3].astype(np.float32)).max(-1)[moving & (source[..., 3] > 200)]
    mean = float(error.mean()) if len(error) else 0.0
    if mean > REST_LIMIT:
        return None, f"the layers do not give the picture back (mean error {mean:.1f})"
    report = {"shift": [round(shift[0], 2), round(shift[1], 2)], "restMean": round(mean, 2), "restP99": round(float(np.percentile(error, 99)), 1) if len(error) else 0.0, "lifted": round(lifted, 3)}
    return {"field": field, "hair": hair, "under": under, "weight": weight, "report": report}, None

