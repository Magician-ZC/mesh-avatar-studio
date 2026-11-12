// Checks of rig.limbs beyond its field types (ER4): unique layer-safe names, joints and outline inside the image, a
// polygon with area, separate joints that do not fold back on themselves, the swing and bend limits, and no limb that is
// also bound as the hanging `arms` or planted `legs`.
import type { Point, Rig } from '../rig/types';
import { LIMB_LIMITS } from './chain.ts';

const NAME = /^[A-Za-z0-9_]+$/;

function inside(rig: Rig, [x, y]: Point): boolean {
  return x >= 0 && y >= 0 && x <= rig.image.width && y <= rig.image.height;
}

function area(outline: Point[]): number {
  let sum = 0;
  for (let i = 0; i < outline.length; i++) {
    const [ax, ay] = outline[i], [bx, by] = outline[(i + 1) % outline.length];
    sum += ax * by - ay * bx;
  }
  return Math.abs(sum) / 2;
}

/** Point in polygon (even-odd rule). */
export function containsPoint(outline: Point[], [x, y]: Point): boolean {
  let hit = false;
  for (let i = 0, j = outline.length - 1; i < outline.length; j = i++) {
    const [xi, yi] = outline[i], [xj, yj] = outline[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

export function limbErrors(rig: Rig): string[] {
  const errors: string[] = [];
  const names = new Set<string>();
  const bound: Point[] = [];
  if (rig.arms) for (const side of ['left', 'right'] as const) bound.push(rig.arms[side].elbow, rig.arms[side].wrist);
  if (rig.legs) for (const side of ['left', 'right'] as const) bound.push(rig.legs[side].knee, rig.legs[side].ankle);
  for (const [i, limb] of (rig.limbs ?? []).entries()) {
    const path = `rig.limbs[${i}]`;
    if (!NAME.test(limb.name) || names.has(limb.name)) errors.push(`${path}.name: must be a unique name of letters, digits and underscores`);
    names.add(limb.name);
    if (limb.joints.some(p => !inside(rig, p))) errors.push(`${path}.joints: must stay inside the image`);
    const [a, b, c] = limb.joints;
    const first = Math.hypot(b[0] - a[0], b[1] - a[1]), second = Math.hypot(c[0] - b[0], c[1] - b[1]);
    if (first < 1 || second < 1) errors.push(`${path}.joints: root, middle and end must be at least 1 px apart`);
    else {
      const bisector = Math.hypot((b[0] - a[0]) / first + (c[0] - b[0]) / second, (b[1] - a[1]) / first + (c[1] - b[1]) / second);
      if (bisector < 0.1) errors.push(`${path}.joints: the two bones must not fold back onto each other`);
    }
    if (limb.outline.some(p => !inside(rig, p))) errors.push(`${path}.outline: points must stay inside the image`);
    if (!(area(limb.outline) >= 1)) errors.push(`${path}.outline: must have a positive area`);
    if ((limb.swing ?? 0) > LIMB_LIMITS.swing) errors.push(`${path}.swing: must be between 0 and ${LIMB_LIMITS.swing} degrees`);
    if ((limb.bend ?? 0) > LIMB_LIMITS.bend) errors.push(`${path}.bend: must be between 0 and ${LIMB_LIMITS.bend} degrees`);
    if (bound.some(p => containsPoint(limb.outline, p))) errors.push(`${path}.outline: the same limb must not also be bound in arms or legs`);
  }
  return errors;
}
