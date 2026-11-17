// Interaction impulses (ER1): give a part an instantaneous velocity, in source px per second. Only velocities in the
// existing physical state change, never positions, so the existing springs, pendulums and damping let it die away.
// Nothing here runs unless `impulse` is called, so avatars render exactly as before without it.
//
//   body    the parameter springs of `bodyAngleX` and `bodyAngleZ` (motion.js)
//   head    the parameter springs of `angleX`, `angleY` and `angleZ` (motion.js)
//   strand  the node velocities of one hair strand, growing from root to tip (physics.js); a fringe strand shares it
//           with its neighbours in the fringe sheet
//   cloth   the angular velocity of the garment columns nearest `at` along the hem, falling off as a cosine (cloth.js)
//   limb    the swing velocity of a limb chain (ER4, ../limbs/runtime.ts) from the push's part across the limb
//
// Unknown targets and non-finite vectors are ignored. Bounds are in ./limits.ts.
import type { Point, Rig } from '../rig/types';
import { CLOTH_FALLOFF, CLOTH_SWING_RATE, FRINGE_FALLOFF, IMPULSE_LIMITS } from './limits.ts';

export type ImpulseTarget =
  | { kind: 'body' | 'head' }
  | { kind: 'strand' | 'cloth' | 'limb'; name: string; at?: number };

/** Motion's parameter springs (motion.js): velocity per parameter id. */
export interface ParameterSprings { vel: Record<string, number> }
/** One physics strand chain (physics.js): node velocities from root to tip. */
export interface StrandChain { v: number[][] }
/** The strand chains in the order of `rig.strands`; the first `fringe` of them move as one sheet. */
export interface StrandChains { strands: StrandChain[]; fringe?: number }
/** The cloth simulator's interaction entry (`Cloth.impulse`, cloth.js); false when nothing was pushed. */
export interface ClothSimulator { impulse(name: string, at: number | undefined, vx: number, vy: number): boolean }
/** The limb chains' interaction entry (ER4, ../limbs/runtime.ts); false for an unknown limb. */
export interface LimbSimulator { impulse(name: string, vx: number, vy: number): boolean }

/** The fields of a cloth.js garment that the impulse reads and writes. */
export interface GarmentState {
  c: { hem: Point[] };
  cols: number;
  rows: number;
  rest: ArrayLike<number>;
  pinned: boolean[];
  grid: { custom: boolean; lengths?: number[] };
  tuning: { max: number };
  angularVelocity: Float64Array;
  lowerAngularVelocity: Float64Array;
  pendulums: { velocities: number[] }[];
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const usable = (value: number) => Number.isFinite(value) && Math.abs(value) > 1e-6;

/** Adds `delta` to a velocity without raising its magnitude above max(cap, |velocity|). */
export function addCapped(velocity: number, delta: number, cap: number): number {
  const limit = Math.max(cap, Math.abs(velocity));
  return clamp(velocity + delta, -limit, limit);
}

/** Adds (dx, dy) to a [vx, vy] velocity in place without raising its speed above max(cap, current speed). */
export function addCapped2(velocity: number[], dx: number, dy: number, cap: number): void {
  const limit = Math.max(cap, Math.hypot(velocity[0], velocity[1]));
  const x = velocity[0] + dx, y = velocity[1] + dy, speed = Math.hypot(x, y);
  const k = speed > limit ? limit / speed : 1;
  velocity[0] = x * k; velocity[1] = y * k;
}

function clampLength(x: number, y: number, max: number): [number, number] {
  const length = Math.hypot(x, y);
  return length > max ? [x * max / length, y * max / length] : [x, y];
}

/**
 * Parameter velocities for a sideways push of `vx` px/s on the upper body, as seen at the chest. The speed is shared
 * between the roll about `body.pivot` and the sideways shift (rigs with legs move their hips through the legs, so
 * they only roll). The body has no vertical parameter, so the vertical component does not move it.
 */
export function bodyRates(rig: Rig, vx: number): Record<string, number> {
  const { body } = rig;
  // applyBody rotates by -bodyAngleZ / 10 * maxRoll about the pivot: a point dy below the pivot moves at -dy * rate
  const lever = body.chest.cy - body.pivotY;
  const roll = usable(lever) && usable(body.maxRoll);
  // ...and shifts by bodyAngleX / 10 * (hip.shift, or 7 + 9 at the chest centre) px when the rig has no legs
  const shift = rig.legs ? 0 : body.hip ? body.hip.shift / 10 : 1.6;
  const share = (roll ? 1 : 0) + (usable(shift) ? 1 : 0);
  const rates: Record<string, number> = {};
  if (roll) rates.bodyAngleZ = 10 * vx / share / (body.maxRoll * lever);
  if (usable(shift)) rates.bodyAngleX = vx / share / shift;
  return rates;
}

/**
 * Parameter velocities for a push of (vx, vy) px/s on the face centre: the sideways speed is shared between turning
 * (`angleX`) and rolling about `head.pivot` (`angleZ`), the vertical speed nods (`angleY`).
 */
export function headRates(rig: Rig, vx: number, vy: number): Record<string, number> {
  const { head } = rig;
  // turnOffset moves the face centre by angleX / 30 * shiftX px sideways and angleY / 30 * shiftY px up;
  // applyHead rolls by -angleZ / 30 * maxRoll about the pivot
  const lever = head.cy - head.pivotY;
  const turn = usable(head.shiftX), roll = usable(lever) && usable(head.maxRoll);
  const share = (turn ? 1 : 0) + (roll ? 1 : 0);
  const rates: Record<string, number> = {};
  if (turn) rates.angleX = 30 * vx / share / head.shiftX;
  if (roll) rates.angleZ = 30 * vx / share / (head.maxRoll * lever);
  if (usable(head.shiftY)) rates.angleY = -30 * vy / head.shiftY;
  return rates;
}

function pushSprings(springs: ParameterSprings, rates: Record<string, number>, cap: number): void {
  for (const [id, rate] of Object.entries(rates)) {
    const current = springs.vel[id];
    if (typeof current !== 'number' || !Number.isFinite(rate)) continue;
    springs.vel[id] = addCapped(current, clamp(rate, -cap, cap), cap);
  }
}

/**
 * Adds a strand impulse (times `weight`, 0..1) to every node but the pinned root, in proportion to its place from root
 * to tip.
 */
export function strandImpulse(chain: StrandChain, vx: number, vy: number, weight = 1): void {
  const [x, y] = clampLength(vx, vy, IMPULSE_LIMITS.strand);
  const last = chain.v.length - 1;
  for (let j = 1; j <= last; j++) {
    const share = j / last;
    addCapped2(chain.v[j], x * weight * share, y * weight * share, IMPULSE_LIMITS.strand * share);
  }
}

const fringeShare = (places: number) => places < FRINGE_FALLOFF ? Math.cos(Math.PI / 2 * places / FRINGE_FALLOFF) : 0;

const hemCache = new WeakMap<GarmentState, number[]>();

/** Position of each column's hem point along the hem polyline, by arc length (0 at its first point, 1 at its last). */
export function hemPositions(g: GarmentState): number[] {
  const cached = hemCache.get(g);
  if (cached) return cached;
  const hem = g.c.hem, lengths = [0];
  for (let i = 1; i < hem.length; i++) lengths.push(lengths[i - 1] + Math.hypot(hem[i][0] - hem[i - 1][0], hem[i][1] - hem[i - 1][1]));
  const total = lengths[lengths.length - 1];
  const positions = Array.from({ length: g.cols }, (_, k) => {
    if (!(total > 0)) return g.cols > 1 ? k / (g.cols - 1) : 0;
    const i = ((g.rows - 1) * g.cols + k) * 2, x = g.rest[i], y = g.rest[i + 1];
    let best = Infinity, along = 0;
    for (let s = 1; s < hem.length; s++) {
      const [ax, ay] = hem[s - 1], dx = hem[s][0] - ax, dy = hem[s][1] - ay, span = dx * dx + dy * dy;
      const u = span > 0 ? clamp(((x - ax) * dx + (y - ay) * dy) / span, 0, 1) : 0;
      const distance = Math.hypot(x - ax - dx * u, y - ay - dy * u);
      if (distance < best) { best = distance; along = (lengths[s - 1] + Math.sqrt(span) * u) / total; }
    }
    return along;
  });
  hemCache.set(g, positions);
  return positions;
}

const falloff = (columns: number) => columns < CLOTH_FALLOFF ? Math.cos(Math.PI / 2 * columns / CLOTH_FALLOFF) : 0;

/**
 * Adds angular velocity to the garment columns nearest `at` (0..1 along the hem; every column when omitted), so that
 * their hems start moving with the impulse. A whole column turns about its waist point, which leaves the shape of the
 * column unchanged. Each column is limited by `IMPULSE_LIMITS.cloth` and by `CLOTH_SWING_RATE` times its rendered limit
 * angle. Columns of a custom grid are pushed once the garment has been stepped (its pendulums exist from then on).
 */
export function clothImpulse(g: GarmentState, at: number | undefined, vx: number, vy: number): boolean {
  if (!Number.isFinite(vx) || !Number.isFinite(vy)) return false;
  const [x, y] = clampLength(vx, vy, IMPULSE_LIMITS.cloth);
  const positions = at === undefined ? null : hemPositions(g);
  const spacing = g.cols > 1 ? 1 / (g.cols - 1) : 1;
  const target = at === undefined ? 0 : clamp(at, 0, 1);
  let pushed = false;
  for (let k = 0; k < g.cols; k++) {
    if (g.pinned[k]) continue;
    const weight = positions ? falloff(Math.abs(positions[k] - target) / spacing) : 1;
    if (weight <= 0) continue;
    const top = k * 2, bottom = ((g.rows - 1) * g.cols + k) * 2;
    const rx = g.rest[bottom] - g.rest[top], ry = g.rest[bottom + 1] - g.rest[top + 1];
    const chord = rx * rx + ry * ry;
    const length = g.grid.custom ? g.grid.lengths?.[k] ?? Math.sqrt(chord) : Math.sqrt(chord);
    if (!(chord > 1) || !(length > 1)) continue;
    // The renderer turns (x, y) into (x cosθ + y sinθ, −x sinθ + y cosθ), so the hem moves at dθ/dt · (ry, −rx).
    const spin = weight * (x * ry - y * rx) / chord;
    const limitAngle = 2 * Math.asin(Math.min(0.95, g.tuning.max / (2 * length)));
    const cap = Math.min(IMPULSE_LIMITS.cloth / length, CLOTH_SWING_RATE * limitAngle);
    const delta = clamp(spin, -cap, cap);
    if (g.grid.custom) {
      const pendulum = g.pendulums[k];
      if (!pendulum) continue;
      pendulum.velocities = pendulum.velocities.map(v => addCapped(v, delta, cap));
    } else {
      g.angularVelocity[k] = addCapped(g.angularVelocity[k], delta, cap);
      g.lowerAngularVelocity[k] = addCapped(g.lowerAngularVelocity[k], delta, cap);
    }
    pushed = true;
  }
  return pushed;
}

/** The `impulse` method of a mesh avatar: dispatches to the motion springs, the hair physics and the cloth. */
export function createImpulse(rig: Rig, parts: { motion: ParameterSprings; physics: StrandChains; cloth: ClothSimulator; limbs?: LimbSimulator | null }) {
  const strands = new Map<string, number[]>();
  (rig.strands ?? []).forEach((strand, i) => strands.set(strand.name, [...(strands.get(strand.name) ?? []), i]));
  const chains = parts.physics.strands;
  const fringe = Math.max(0, Math.min(chains.length, parts.physics.fringe ?? 0));
  function pushStrand(index: number, x: number, y: number): void {
    if (index >= fringe) {
      if (chains[index]) strandImpulse(chains[index], x, y);
      return;
    }
    for (let k = 0; k < fringe; k++) {
      const share = fringeShare(Math.abs(k - index));
      if (share > 0) strandImpulse(chains[k], x, y, share);
    }
  }
  return function impulse(target: ImpulseTarget, vx: number, vy: number): void {
    if (!target || typeof target !== 'object') return;
    const x = Number(vx), y = Number(vy);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    if (target.kind === 'body') pushSprings(parts.motion, bodyRates(rig, x), IMPULSE_LIMITS.body);
    else if (target.kind === 'head') pushSprings(parts.motion, headRates(rig, x, y), IMPULSE_LIMITS.head);
    else if (target.kind === 'strand') {
      for (const i of strands.get(target.name) ?? []) pushStrand(i, x, y);
    } else if (target.kind === 'cloth' && typeof target.name === 'string') {
      const at = typeof target.at === 'number' && Number.isFinite(target.at) ? target.at : undefined;
      parts.cloth.impulse(target.name, at, x, y);
    } else if (target.kind === 'limb' && typeof target.name === 'string') parts.limbs?.impulse(target.name, x, y);
    // other kinds are not targets
  };
}
