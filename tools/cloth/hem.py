"""Hem band (`build-layers.py --cloth-under hem`): what a swinging garment uncovers.

With `--cloth-under keep` the base under a garment keeps the garment itself, so wherever the garment swings away the
base shows a second, static copy of it, and the travel has to stay so small that the copy hardly shows. With `hem` the
base keeps the garment too, except a band along the garment's free edges, which is what the garment uncovers when it
swings:

- the band follows the garment's own travel (`cloth[*].max`, default 20 px): (max * BAND_K + BAND_PAD) px deep at the
  garment's lowest point, narrowing linearly to nothing at its waist line. The waist never opens;
- the garment's drawn edge (line art, soft edge), which lies a few px outside the outline, goes into the garment layer
  where the band is, so no line of the garment stays behind; the layer is cut hard there, and the band lies under it,
  so at rest the two are the picture again;
- in the band the base continues only what is visible outside the garment: under the hem, in a column where a leg comes
  out below it (`legs`, a mask the caller gives, e.g. the parser's skin), the leg's own colour as it is just below the
  hem, carried up and shaded (a little darker at the hem, more towards the garment's inside); everywhere else the band
  is transparent.

Nothing is generated: no model is asked, and no pixel holds anything but a copy of the leg below or transparency. The
band never reaches above the waist line and is never wider than the garment's travel allows.
"""

import cv2
import numpy as np

DEFAULT_MAX = 20.0  # px: the engine's default garment travel (cloth[*].max)
BAND_K = 1.25  # of the travel: how deep the band reaches at the garment's lowest point
BAND_PAD = 3.0  # px more, for the soft edge and the mesh rounding
RING = 10  # px outside the outline that may still belong to the garment's drawn edge (line art, soft edge, an outline
# traced from a coarse parse); legs are never taken into it
LOOK = (8, 20)  # px below a column's lowest garment pixel where a leg coming out under the hem is read
MIN_LEG = 0.6  # share of that window that must be leg for the column to continue it
SHADE = 0.35  # how much darker the continued leg is at the band's inner end (the garment's shadow)
SHADE_AT_HEM = 0.08  # ... and right at the hem: the band never repeats the picture exactly, so a quality check that looks
# for the garment's static copy (base pixels equal to the garment-kept base) does not take a continued leg for it
OPAQUE = 250  # a band pixel continues a leg only where the picture is opaque (soft edges lie over the background)


def depth_of(travel):
    """How deep (px) the band reaches at the hem for a garment with this travel."""
    return float(travel) * BAND_K + BAND_PAD


def band(rgba, polygon, garment, legs=None, keep_out=None):
    """The hem band of one garment.

    rgba: the picture (full size); polygon: the garment's outline as a mask (uint8 or bool, full size); garment: its rig
    entry (`waist`, optional `max`); legs: optional mask (white = a leg that may continue under the hem); keep_out:
    optional mask of pixels that belong to other layers (other garments, hand, accessories), left alone.
    Returns (layer, band, fill, report): the garment layer's mask (outline and its drawn edge where the band is), the
    band, what the base holds there (RGBA uint8, full size; transparent unless a leg continues) and what was done.
    """
    h, w = rgba.shape[:2]
    polygon = polygon > 0
    keep_out = np.zeros((h, w), bool) if keep_out is None else keep_out > 0
    travel = float(garment.get("max", DEFAULT_MAX))
    depth_at_hem = depth_of(travel)
    rows = np.nonzero(polygon.any(1))[0]
    if not len(rows):
        return polygon, np.zeros((h, w), bool), np.zeros((h, w, 4), np.uint8), {"band": 0.0, "pixels": 0, "legs": 0, "top": None}
    waist = float(np.mean([point[1] for point in garment["waist"]]))
    lowest = float(rows.max())
    ys = np.arange(h, dtype=np.float32)[:, None]
    reach = depth_at_hem * np.clip((ys - waist) / max(lowest - waist, 1.0), 0, 1)  # band depth per row
    reach = np.broadcast_to(reach, (h, w))

    grown = cv2.dilate(polygon.astype(np.uint8), np.ones((2 * RING + 1, 2 * RING + 1), np.uint8)) > 0
    ring = grown & ~polygon & (rgba[..., 3] > 0) & (reach >= 2) & ~keep_out
    if legs is not None:
        ring &= ~(legs > 127)
    region = polygon | ring
    inside = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    banded = ((region & (inside <= reach)) | ring) & ~keep_out

    fill = np.zeros((h, w, 4), np.uint8)
    continued = np.zeros((h, w), bool)
    if legs is not None:
        legs = legs > 127
        columns = np.nonzero(region.any(0))[0]
        bottom = np.full(w, -1)
        bottom[columns] = h - 1 - np.argmax(region[::-1, columns], axis=0)
        colour = np.zeros((w, 4), np.float32)
        for x in columns:
            y0, y1 = bottom[x] + LOOK[0], min(h, bottom[x] + LOOK[1])
            if y0 >= y1:
                continue
            window = rgba[y0:y1, x].astype(np.float32)
            if legs[y0:y1, x].mean() >= MIN_LEG and np.median(window[:, 3]) >= 128:
                colour[x] = np.median(window, 0)
        found = colour[:, 3] > 0
        if found.any():
            # smooth along the hem, among leg columns only (no colour bleeds in from the gaps between the legs)
            weight = cv2.GaussianBlur(found.astype(np.float32)[None], (0, 0), sigmaX=3, sigmaY=0)[0]
            smooth = cv2.GaussianBlur((colour * found[:, None])[None], (0, 0), sigmaX=3, sigmaY=0)[0]
            colour = np.where(found[:, None], smooth / np.maximum(weight[:, None], 1e-6), 0)
            up = (bottom[None, :] - ys) / max(depth_at_hem, 1.0)  # 0 at the column's hem, 1 a band depth up
            continued = banded & found[None, :] & (up >= 0) & (up <= 1) & (rgba[..., 3] >= OPAQUE)
            cy, cx = np.nonzero(continued)
            shade = 1 - SHADE_AT_HEM - (SHADE - SHADE_AT_HEM) * np.clip(up[cy, cx], 0, 1)
            fill[cy, cx, :3] = np.clip(np.rint(colour[cx, :3] * shade[:, None]), 0, 255).astype(np.uint8)
            fill[cy, cx, 3] = np.clip(np.rint(colour[cx, 3]), 0, 255).astype(np.uint8)
    rows_banded = np.nonzero(banded.any(1))[0]
    report = {
        "band": round(depth_at_hem, 1),
        "pixels": int(banded.sum()),
        "legs": int(continued.sum()),
        "top": int(rows_banded.min()) if len(rows_banded) else None,
    }
    return region, banded, fill, report
