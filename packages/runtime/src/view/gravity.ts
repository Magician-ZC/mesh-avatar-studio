// Gravity direction (ER6, WebGL runtime only): `rig.physics.gravity` is the direction things hang in, in radians in
// source pixels (x right, y down), default pi/2: straight down the source image. A figure turned upright to be bound
// while it was lying keeps its real gravity this way: hair, buns and garments hang and sway along it, the breeze and
// the head-roll correction run across it. Absent, or exactly pi/2, physics keeps its original code path unchanged.

export const DEFAULT_GRAVITY = Math.PI / 2;

/** The unit vector along gravity (`down`) and the one across it (`across`; +x for the default). */
export interface GravityFrame { down: [number, number]; across: [number, number] }

/** The rig's gravity, or null for the default direction (the original code path). */
export function gravityFrame(rig: { physics?: { gravity?: number } }): GravityFrame | null {
  const value = rig.physics?.gravity;
  if (typeof value !== 'number' || !Number.isFinite(value) || value === DEFAULT_GRAVITY) return null;
  const c = Math.cos(value), s = Math.sin(value);
  // across is down turned by -pi/2: for the default (0, 1) it is (1, 0), the source x axis
  return { down: [c, s], across: [s, -c] };
}

/** The component of (x, y) across gravity and along it. */
export function splitByGravity(frame: GravityFrame, x: number, y: number): [number, number] {
  return [x * frame.across[0] + y * frame.across[1], x * frame.down[0] + y * frame.down[1]];
}
