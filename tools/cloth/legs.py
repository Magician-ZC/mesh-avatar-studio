"""Recover the two leg surfaces underneath a skirt from its owner's visible legs.

All coordinates are source pixels. Scanlines are fitted to the visible silhouettes, including
their offset from the pose detector. Each surface borrows its own two-dimensional thigh/calf
texture in cross-section coordinates. Empty pixels between and outside the legs stay empty.
"""

import cv2
import numpy as np


def trace_hem(rgba, region):
    """Trace the ink/alpha boundary nearest the fitted lower edge, through adjoining columns."""
    height, width = region.shape
    end = height - 1 - region[::-1].argmax(axis=0)
    active = region.any(axis=0) & (end > height * 0.48)
    if not active.any():
        return region
    radius = max(5, round(min(height * .10, width * .05)))
    radius = min(radius, 75)
    top = max(0, int(end[active].min()) - radius)
    bottom = min(height, int(end[active].max()) + radius + 2)
    area = rgba[top:bottom].astype(np.float32)
    area[..., :3] *= area[..., 3:] / 255
    smooth = cv2.GaussianBlur(area, (5, 5), 1)
    r, g, b, a = cv2.split(smooth)
    feature = np.stack((3 * (r - b), 3 * (g - b), .42 * (r + g + b), 1.4 * a), axis=-1)
    ys = np.arange(top, bottom)
    xx = np.arange(width)
    local = end - top
    prior = []
    for sign in (-1, 1):
        samples = np.stack([feature[np.clip(local + sign * d, 0, len(ys) - 1), xx]
                            for d in np.linspace(.72 * radius, radius, 12).astype(int)])
        prior.append(np.median(samples, axis=0))
    costs = np.linalg.norm(feature - prior[0], axis=2) - np.linalg.norm(feature - prior[1], axis=2)
    # A hem separates two materials. A specular streak inside leather or a cast shadow
    # on a calf may have a stronger gradient, but has the same material on both sides.
    sep = np.maximum(np.linalg.norm(prior[0] - prior[1], axis=1), 1)
    costs = np.clip(costs / sep, -1, 1)
    energy = np.cumsum(costs, axis=0)
    edges = np.linalg.norm(np.diff(smooth, axis=0, append=smooth[-1:]), axis=2)
    score = -energy * 2 + np.minimum(edges, 70) * .055 - .035 * np.abs(ys[:, None] - end)
    score[np.abs(ys[:, None] - end) > radius] = -1e5
    traced = region.copy()
    for x0, x1 in np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2):
        tracks = np.zeros((x1 - x0, len(ys)), np.int8)
        value = score[:, x0].copy()
        for x in range(x0 + 1, x1):
            padded = np.pad(value, 4, constant_values=-1e6)
            jumps = np.stack([padded[i:i + len(ys)] - abs(i - 4) * 1.1 for i in range(9)])
            tracks[x - x0] = np.argmax(jumps, axis=0) - 4
            value = jumps.max(axis=0) + score[:, x]
        row = int(value.argmax())
        for x in range(x1 - 1, x0 - 1, -1):
            last = int(ys[row])
            # The strongest gradient sits in the center of the anti-aliased outline.
            while last + 1 < height and rgba[last, x, 3] < 240 and rgba[last + 1, x, 3] > 0:
                last += 1
            if last > end[x]:
                traced[end[x] + 1:last + 1, x] = rgba[end[x] + 1:last + 1, x, 3] > 0
            else:
                traced[last + 1:end[x] + 1, x] = False
            row += int(tracks[x - x0, row])
    return traced


def side_occluders(rgba, rect, region, span, hip_y):
    """Find coat/bag pixels that a coarse side outline included with the skirt."""
    x, y, w, h = rect
    near = max(5, round(.12 * span))
    blocked = np.zeros_like(region)
    yy = np.flatnonzero(region.any(1) & (y + np.arange(h) >= hip_y))
    if not len(yy):
        return blocked
    steps = np.arange(-near, near + 1)
    for reverse in (False, True):
        edge = w - 1 - region[yy, ::-1].argmax(1) if reverse else region[yy].argmax(1)
        direction = -1 if reverse else 1
        xx = np.clip(x + edge[:, None] + direction * steps, 0, rgba.shape[1] - 1)
        rgb = rgba[y + yy[:, None], xx].astype(np.float32)
        r, g, b = (rgb[..., i] for i in range(3))
        f = np.stack((2 * (r-b), 2 * (g-b), (r+g+b)*.22), axis=2)
        window = max(2, near // 3)
        outside, inside = np.median(f[:, :window], axis=1), np.median(f[:, -window:], axis=1)
        contrast = np.linalg.norm(inside-outside, axis=1)
        cost = np.linalg.norm(f-outside[:, None], axis=2) - np.linalg.norm(f-inside[:, None], axis=2)
        energy = np.cumsum(np.clip(cost / np.maximum(contrast[:, None], 1), -1, 1), axis=1)
        border = np.argmin(energy + .04 * np.abs(steps), axis=1) - near
        # A fixed neighboring object must be visible just outside this side in the source.
        # A ruffled or curved bottom eventually runs almost horizontally across a thigh.
        # It is the hem, even though it becomes the last nonempty pixel of that scanline.
        side = edge > .80 * w if reverse else edge < .20 * w
        vertical = np.abs(np.gradient(edge.astype(float))) < 1.8 if len(edge) > 1 else side
        valid = side & vertical & (rgb[:, :window, 3].mean(1) > 240) & (contrast > 16)
        border = np.where(valid, np.clip(border, 0, near-1), 0)
        boundary = edge + direction * border
        for row, start, stop in zip(yy, edge, boundary):
            if start != stop:
                blocked[row, min(start, stop):max(start, stop)+1] = True
    return blocked & region


def sections(rgba, bones, span, last):
    """Paired source-silhouette boundaries just below the hem, before sock or shoe changes."""
    joints = np.asarray([bones[k] for k in ('hip', 'knee', 'ankle')], float)
    step = max(1, round(span / 140))
    start = max(0, round(last - .08 * span))
    stop = min(rgba.shape[0], round(joints[2, 1] - .12 * (joints[2, 1] - joints[1, 1])))
    rows = []
    for y in range(start, stop, step):
        center = np.interp(y, joints[:, 1], joints[:, 0])
        runs = np.flatnonzero(np.diff(np.r_[False, rgba[y, :, 3] > 100, False])).reshape(-1, 2)
        for left, right in runs:
            if left <= center < right and .18 * span < right - left < .88 * span:
                # A separate, stable silhouette excludes back panels and dangling hem petals.
                if rows and (y - rows[-1][0] > 2 * step or
                             max(abs(left - rows[-1][1]), abs(right - rows[-1][2])) > .045 * span):
                    rows = []
                rows.append((y, left, right - 1))
                break
        if len(rows) >= max(5, round(.54 * span / step)):
            break
    if len(rows) < 4:
        return None
    return np.asarray(rows, np.float64)


def leg_surface(rgba, rect, outer, reach, bones, span):
    x, y, w, h = rect
    hip, knee, ankle = (np.asarray(bones[k]) for k in ('hip', 'knee', 'ankle'))
    probe_x = int(np.clip(np.interp(y + h, [hip[1], knee[1], ankle[1]],
                                    [hip[0], knee[0], ankle[0]]) - x, 0, w - 1))
    lower = np.where(outer[:, probe_x])[0]
    if not len(lower):
        return None
    sample = sections(rgba, bones, span, y + lower[-1])
    if sample is None:
        return None
    ys, left, right = sample.T
    source_centers, source_radii = (left + right) / 2, (right - left) / 2
    joints_y = [hip[1], knee[1], ankle[1]]
    joints_x = [hip[0], knee[0], ankle[0]]
    # Smooth over the first few boundary pixels of the visible section.
    n = min(10, len(ys))
    anchor = float(ys[:n].mean())
    radius = float(source_radii[:n].mean())
    offset = float(np.mean(source_centers[:n] - np.interp(ys[:n], joints_y, joints_x)))
    length = knee[1] - hip[1]
    target_y = y + np.arange(h, dtype=np.float32)
    target_x = x + np.arange(w, dtype=np.float32)
    frac = np.clip((target_y - hip[1]) / length, 0, 2)
    anchor_frac = (anchor - hip[1]) / length
    # Thigh, knee and calf widths vary smoothly, with scale taken from the same leg.
    knots = [0, .22, .57, .92, 1.04, 1.28, 1.54, 1.85, 2]
    shape = [.98, 1.04, .90, .67, .64, .76, .78, .52, .37]
    radii = radius * np.interp(frac, knots, shape) / np.interp(anchor_frac, knots, shape)
    # Join the actual silhouette tangent near the first exposed row.
    slope_radius = float(np.polyfit(ys - anchor, source_radii, 1)[0])
    slope_center = float(np.polyfit(ys - anchor, source_centers, 1)[0])
    centers = np.interp(target_y, joints_y, joints_x) + offset * np.clip(1 - (anchor - target_y) / length, 0, 1)
    tangent_mix = np.exp(-np.maximum(anchor - target_y, 0) / (.20 * length))
    radii = radii * (1 - tangent_mix) + (radius + slope_radius * (target_y - anchor)) * tangent_mix
    center_at_anchor = np.interp(anchor, ys, source_centers)
    centers = centers * (1 - tangent_mix) + (center_at_anchor + slope_center * (target_y - anchor)) * tangent_mix
    centers = np.where(target_y >= ys[0], np.interp(target_y, ys, source_centers), centers)
    radii = np.where(target_y >= ys[0], np.interp(target_y, ys, source_radii), radii)
    radii = cv2.GaussianBlur(radii[:, None], (1, 0), max(.035 * length, 1))[:, 0]
    centers = cv2.GaussianBlur(centers[:, None], (1, 0), max(.025 * length, 1))[:, 0]
    # Use unshadowed visible rows away from the stationary cast shadow along the original hem.
    first, last = float(ys[0]), float(ys[-1])
    period = max(last - first, 1.)
    phase = np.mod(first - target_y, 2 * period)
    donor_y = first + np.minimum(phase, 2 * period - phase)
    donor_radius = np.interp(donor_y, ys, source_radii)
    donor_center = np.interp(donor_y, ys, source_centers)
    u = (target_x[None, :] - centers[:, None]) / np.maximum(radii[:, None], 1)
    sample_x = donor_center[:, None] + np.clip(u, -1, 1) * (donor_radius[:, None] - 1)
    # Keep cv2 input dimensions local to the few rows being sampled.
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    sample_y = np.broadcast_to((donor_y - y0)[:, None], (h, w)).astype(np.float32)
    swatch = rgba[y0:y1, :, :3]
    # Recover the unshadowed transverse light profile from the visible segment. Repeating
    # its raw rows would stamp the old skirt's shadow onto the thigh at regular intervals.
    table_u = np.linspace(-1, 1, 257)
    table_x = source_centers[:, None] + table_u * (source_radii[:, None] - 1)
    table_y = np.broadcast_to((ys-y0)[:, None], table_x.shape)
    swatches = cv2.remap(swatch, table_x.astype(np.float32), table_y.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    profile = np.percentile(swatches, 68, axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile[None], (7, 1), 0)
    lit = cv2.remap(profile, (np.clip(u, -1, 1) * 128 + 128).astype(np.float32),
                    np.zeros((h, w), np.float32), cv2.INTER_LINEAR)
    fine = swatch.astype(np.float32) - cv2.GaussianBlur(swatch.astype(np.float32), (5, 5), 0)
    grain = cv2.remap(fine, sample_x.astype(np.float32), sample_y, cv2.INTER_LINEAR)
    colors = lit + np.clip(grain, -9, 9)
    # Blend back to the untouched calf below the old contact shadow. The source is sampled
    # in two dimensions, avoiding the vertical streaks of stretching hem-coloured columns.
    end = h - 1 - outer[::-1].argmax(axis=0)
    distance = np.maximum(np.arange(h)[:, None] - end[None, :], 0)
    mix = np.clip((distance / reach - .45) / .45, 0, 1)
    mix = mix * mix * (3 - 2 * mix)
    original = rgba[y:y + h, x:x + w]
    colors = np.clip(colors * (1-mix[..., None]) + original[..., :3] * mix[..., None], 0, 255)
    alpha = np.clip((1 - np.abs(u)) * radii[:, None] + .5, 0, 1)
    # Below the skirt the original silhouette is known, including its subpixel edge.
    near = np.abs(u) < 1.20
    alpha = np.where(near, alpha * (1-mix) + (original[..., 3] / 255) * mix, alpha)
    silhouette = (outer | (distance <= reach)) & outer.any(0) & (original[..., 3] == 255)
    sprite = np.zeros((h, w, 4), np.uint8)
    sprite[..., 3] = np.rint(alpha * silhouette * 255).astype(np.uint8)
    sprite[..., :3] = colors
    sprite[sprite[..., 3] == 0] = 0
    return sprite


def recover(rgba, rect, mask, legs):
    """Return the true skirt coverage and the two fitted silhouettes, in skirt-local coordinates."""
    x, y, w, h = rect
    span = abs(legs['right']['hip'][0] - legs['left']['hip'][0])
    # Include the last few painted hem pixels if the fitted sprite stopped above them.
    outer = trace_hem(rgba[y:y + h, x:x + w], mask)
    if not span:
        return None
    # The upper waistband stays anchored. Legs begin at their hip joints.
    hip = min(legs[side]['hip'][1] for side in ('left', 'right'))
    fixed = side_occluders(rgba, rect, outer, span, hip)
    outer &= ~fixed
    clear = outer & (y + np.arange(h)[:, None] >= hip)
    reach = max(6, round(.43 * span))
    surfaces = {side: leg_surface(rgba, rect, outer, reach, legs[side], span) for side in ('left', 'right')}
    if any(surface is None for surface in surfaces.values()):
        return None
    end = h - 1 - outer[::-1].argmax(axis=0)
    distance = np.arange(h)[:, None] - end[None, :]
    source = rgba[y:y + h, x:x + w]
    for pixels in surfaces.values():
        pixels[~(clear | (distance > 0))] = 0
    body = (surfaces['left'][..., 3] == 255) | (surfaces['right'][..., 3] == 255)
    # A narrow skirt fringe can end outside the leg. Clean through its last visible ink
    # if there is stage directly below it, without erasing neighboring sleeves or back panels.
    fringe = max(4, round(span * .048))
    below = source[np.minimum(h - 1, end + fringe + 1), np.arange(w), 3]
    outside = (below < 8) & (distance <= fringe)
    extension = (distance > 0) & outer.any(0) & (body | outside) & ~fixed
    for pixels in surfaces.values():
        pixels[fixed] = 0
    # Leave an overlap on the opaque original calf where its texture already matches.
    # Independently minified base/leg textures otherwise sample transparency across the
    # same cut from both sides, producing a pale seam in the small desktop window.
    overlap = body & (distance > .72 * reach) & (source[..., 3] == 255)
    clear |= extension & ~overlap
    contact = np.zeros_like(source)
    contact[extension] = source[extension]
    return outer, clear, surfaces, contact


def coat_back(rgba, rect, outer, clear, surfaces, legs):
    """Extend a rear coat only if its hem is actually visible in the gap between the legs."""
    x, y, w, h = rect
    source = rgba[y:y + h, x:x + w]
    xx = np.arange(w)
    hips = [legs[side]['hip'][0] - x for side in ('left', 'right')]
    span = abs(hips[1] - hips[0])
    center = np.mean(hips)
    end = h - 1 - outer[::-1].argmax(axis=0)
    at = end + max(3, round(span * .014))
    depth = max(4, round(span * .35))
    offsets = np.arange(depth)
    rows = np.minimum(at[None, :] + offsets[:, None], h - 1)
    open_air = source[rows, xx, 3] < 16
    length = open_air.argmax(0)
    body = (surfaces['left'][..., 3] > 12) | (surfaces['right'][..., 3] > 12)
    visible = (np.abs(xx - center) < span * .35) & outer.any(0) & open_air.any(0)
    visible &= (length > span * .045) & ~body[np.minimum(at + length // 2, h - 1), xx]
    runs = np.flatnonzero(np.diff(np.r_[False, visible, False])).reshape(-1, 2)
    run = next(((l, r) for l, r in runs if l <= center <= r and r-l > .07 * span), None)
    if run is None:
        return None
    left, right = run
    # Fold sampling coordinates into the actual rear-panel swatch, keeping its soft hem
    # solely at the original height. Never continue the front skirt into this opening.
    width = right - left - 1
    dx = (xx - left) % (2 * width)
    sx = left + np.minimum(dx, 2 * width - dx)
    lo = at[sx] + np.rint(length[sx] * .12).astype(int)
    band = np.maximum(2, np.rint(length[sx] * .62).astype(int))
    dy = (end - np.arange(h)[:, None]) % (2 * band)
    sy = lo + np.minimum(dy, 2 * band - dy)
    sprite = np.zeros_like(source)
    # Extend the rear coat behind the side of the skirt wherever the source shows an
    # opaque side panel next to that row. This joins the back to the coat at both thighs.
    row = np.arange(h)
    left_edge = outer.argmax(1)
    right_edge = w - 1 - outer[:, ::-1].argmax(1)
    margin = max(3, round(.024 * span))
    left_coat = rgba[y + row, np.maximum(0, x + left_edge - margin), 3] > 240
    right_coat = rgba[y + row, np.minimum(rgba.shape[1]-1, x + right_edge + margin), 3] > 240
    rear = (np.abs(xx - center) < .45 * span) | np.where(xx < center, left_coat[:, None], right_coat[:, None])
    keep = outer & clear & rear & (source[..., 3] == 255)
    sprite[keep] = source[np.minimum(sy, h - 1), sx][keep]
    return sprite
