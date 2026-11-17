// Cheek pull (T-ER12, WebGL runtime only): the cheek layers cut by tools/head/cheeks.py (`cheek_left.png`,
// `cheek_right.png` and "cheeks" in layers.json) and how a held cheek moves them.
//
// Each side has the face's outer contour on that side (`edge`, top to bottom, source px), how far into the face a pull
// reaches at each contour point (`reach`) and the largest pull of the contour (`travel`). A vertex of a cheek layer takes
// the pull by
//   - where it lies along the contour: a raised cosine over the contour, 0 at both ends, centred on the pinch;
//   - how deep it lies inside the face: 1 on the contour (and on its drawn line just outside), 0 at `reach`.
// The contour itself only ever moves outwards along its normal: a pull inwards or along the contour moves the skin inside
// it and leaves the contour where it is. The layer is drawn over the unchanged base, so the old contour and the hair
// beside the face stay covered whatever the pull. Pulling towards the camera (depth > 0) swells the skin around the pinch
// a little and lights it like a bulge (renderer.js); its outline hardly changes.
//
// An avatar without the data has no cheek layers and draws exactly as before.
import type { Point } from '../rig/types';

export type CheekSide = 'left' | 'right';
export const CHEEK_SIDES: readonly CheekSide[] = Object.freeze(['left', 'right'] as const);

/** One side of layers.json "cheeks". */
export interface CheekEdge {
  /** The face's outer contour on this side, top to bottom, source px (at least 3 points). */
  edge: Point[];
  /** How far into the face a pull reaches at each contour point, px (as many as `edge`). */
  reach: number[];
  /** Largest pull of the contour, px. */
  travel: number;
  /** Optional: how much of the pull each contour point takes (0..1, as many as `edge`; 1 when absent): less where hair lies over the contour. */
  free?: number[];
}
export type CheekData = Partial<Record<CheekSide, CheekEdge>>;

/** Bounds of a cheek pull. */
export const CHEEK_LIMITS = Object.freeze({
  /** A pull inwards moves the skin inside the contour at most this share of `travel`. */
  inward: 0.3,
  /** A pull along the contour moves the skin inside it at most this share of `travel`. */
  along: 0.5,
  /** Release speed, source px/s. */
  release: 600,
  /** At full depth the skin around the pinch swells by this share of its distance from the pinch. */
  swell: 0.1,
});

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const sstep = (a: number, b: number, x: number) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

/** Problems of layers.json "cheeks" (empty when it is valid). */
export function cheekProblems(value: unknown, path = 'cheeks'): string[] {
  if (!object(value)) return [`${path}: expected an object with "left" and/or "right"`];
  const errors: string[] = [];
  for (const [side, data] of Object.entries(value)) {
    const at = `${path}.${side}`;
    if (!(CHEEK_SIDES as readonly string[]).includes(side)) { errors.push(`${at}: unknown side (expected left or right)`); continue; }
    if (!object(data)) { errors.push(`${at}: expected { edge, reach, travel }`); continue; }
    const { edge, reach, travel } = data;
    if (!Array.isArray(edge) || edge.length < 3 || !edge.every(p => Array.isArray(p) && p.length === 2 && p.every(finite))) {
      errors.push(`${at}.edge: expected at least 3 [x, y] points`);
    } else if (edge.some((p, i) => i > 0 && Math.hypot(p[0] - edge[i - 1][0], p[1] - edge[i - 1][1]) === 0)) {
      errors.push(`${at}.edge: repeated point`);
    }
    if (!Array.isArray(reach) || !reach.every(r => finite(r) && r > 0) || (Array.isArray(edge) && reach.length !== edge.length)) {
      errors.push(`${at}.reach: expected a positive distance for every edge point`);
    }
    if (!finite(travel) || travel <= 0) errors.push(`${at}.travel: expected a positive number`);
    const { free } = data;
    if (free !== undefined && (!Array.isArray(free) || !free.every(v => finite(v) && v >= 0 && v <= 1) || (Array.isArray(edge) && free.length !== edge.length))) {
      errors.push(`${at}.free: expected a share from 0 to 1 for every edge point`);
    }
  }
  return errors;
}

/** The sides of layers.json "cheeks" that are valid; anything else is left out. */
export function parseCheeks(value: unknown): CheekData {
  if (!object(value)) return {};
  const data: CheekData = {};
  for (const side of CHEEK_SIDES) {
    if (value[side] === undefined) continue;
    if (cheekProblems({ [side]: value[side] }).length) continue;
    const { edge, reach, travel, free } = value[side] as CheekEdge;
    data[side] = { edge: edge.map(([x, y]) => [x, y] as Point), reach: [...reach], travel, ...(free ? { free: [...free] } : {}) };
  }
  return data;
}

/**
 * The pull's shape along the contour (s 0..1): a raised cosine half the contour wide on either side of `centre`, eased
 * to 0 over the last `CHEEK_ENDS` at both ends, so the pulled contour meets the still one smoothly.
 */
export function cheekBump(s: number, centre = 0.5): number {
  const t = (s - centre) / 0.5;
  if (!(s > 0 && s < 1) || Math.abs(t) >= 1) return 0;
  return (0.5 + 0.5 * Math.cos(Math.PI * t)) * sstep(0, CHEEK_ENDS, s) * sstep(0, CHEEK_ENDS, 1 - s);
}
const CHEEK_ENDS = 0.15;

/** Where each vertex of a cheek layer lies relative to its contour (computed once). */
export interface CheekField {
  /** Indices of the vertices the pull can move (all others stay). */
  index: number[];
  /** Position along the contour, 0 at the top .. 1 at the bottom. */
  along: number[];
  /** Outward normal of the contour there. */
  normal: Point[];
  /** Inward fade times the contour's `free` share there: 1 on the contour and outside it, 0 from `reach` inwards. */
  fall: number[];
  /** 0 on the contour .. 1 a little inside: the share of a move along the contour or inwards that the vertex takes. */
  inner: number[];
  /** Contour length, px. */
  length: number;
  /** The contour's mean outward normal, weighted by the pull. */
  outward: Point;
  /** From the contour's top to its bottom, unit length. */
  downward: Point;
  /** Reach at each contour point (copied), for the swell's radius. */
  reach: number[];
  /** Arc length position of each contour point, 0..1. */
  stations: number[];
}

/** Closest point of the contour to (x, y): position along it (0..1), signed depth inside (px) and the outward normal. */
export function onEdge(data: CheekEdge, side: CheekSide, x: number, y: number) {
  const { edge } = data;
  const lengths = [0];
  for (let i = 1; i < edge.length; i++) lengths.push(lengths[i - 1] + Math.hypot(edge[i][0] - edge[i - 1][0], edge[i][1] - edge[i - 1][1]));
  const total = lengths[lengths.length - 1];
  const sign = side === 'left' ? 1 : -1;
  // segment normals, outward: the left contour runs down the left of the face, so outward is to its right-hand side
  const normals = edge.slice(1).map((p, i) => {
    const dx = p[0] - edge[i][0], dy = p[1] - edge[i][1], l = Math.hypot(dx, dy);
    return [-dy / l * sign, dx / l * sign] as Point;
  });
  let best = Infinity, at = 0, depth = 0, normal: Point = normals[0], station = 0;
  for (let i = 0; i < normals.length; i++) {
    const [ax, ay] = edge[i], vx = edge[i + 1][0] - ax, vy = edge[i + 1][1] - ay, span = vx * vx + vy * vy;
    const u = clamp(((x - ax) * vx + (y - ay) * vy) / span, 0, 1);
    const px = ax + vx * u, py = ay + vy * u, distance = Math.hypot(x - px, y - py);
    if (distance < best) {
      best = distance;
      // the normal blends into the neighbouring segments' over each half, so it turns smoothly along the contour
      const prev = normals[Math.max(0, i - 1)], next = normals[Math.min(normals.length - 1, i + 1)], own = normals[i];
      const k = u < 0.5 ? 0.5 - u : u - 0.5, other = u < 0.5 ? prev : next;
      const nx = own[0] * (1 - k) + other[0] * k, ny = own[1] * (1 - k) + other[1] * k, nl = Math.hypot(nx, ny) || 1;
      normal = [nx / nl, ny / nl];
      depth = -((x - px) * own[0] + (y - py) * own[1]);
      at = total > 0 ? (lengths[i] + Math.sqrt(span) * u) / total : 0;
      station = i + u;
    }
  }
  const i = Math.min(data.reach.length - 2, Math.floor(station)), t = station - i;
  const reach = data.reach[i] + (data.reach[i + 1] - data.reach[i]) * t;
  const free = data.free ? data.free[i] + (data.free[i + 1] - data.free[i]) * t : 1;
  return { along: at, depth, normal, reach, free, length: total };
}

/** The cheek field of a layer mesh's rest vertices (source px). */
export function cheekField(data: CheekEdge, side: CheekSide, rest: ArrayLike<number>): CheekField {
  const field: CheekField = { index: [], along: [], normal: [], fall: [], inner: [], length: 0, outward: [0, 0], downward: [0, 1], reach: [...data.reach], stations: [] };
  for (let k = 0; k < rest.length / 2; k++) {
    const hit = onEdge(data, side, rest[k * 2], rest[k * 2 + 1]);
    field.length = hit.length;
    const fall = (hit.depth <= 0 ? 1 : 1 - sstep(0, hit.reach, hit.depth)) * hit.free;
    if (fall <= 0 || hit.along <= 0 || hit.along >= 1) continue;
    field.index.push(k); field.along.push(hit.along); field.normal.push(hit.normal); field.fall.push(fall);
    field.inner.push(sstep(0, 0.35 * hit.reach, Math.max(0, hit.depth)));
  }
  const lengths = [0];
  for (let i = 1; i < data.edge.length; i++) lengths.push(lengths[i - 1] + Math.hypot(data.edge[i][0] - data.edge[i - 1][0], data.edge[i][1] - data.edge[i - 1][1]));
  field.stations = lengths.map(l => l / (lengths[lengths.length - 1] || 1));
  // mean outward direction, weighted by the pull's shape
  let nx = 0, ny = 0;
  const sign = side === 'left' ? 1 : -1;
  for (let i = 1; i < data.edge.length; i++) {
    const w = cheekBump((field.stations[i] + field.stations[i - 1]) / 2);
    const dx = data.edge[i][0] - data.edge[i - 1][0], dy = data.edge[i][1] - data.edge[i - 1][1];
    nx += -dy * sign * w; ny += dx * sign * w;
  }
  const l = Math.hypot(nx, ny) || 1;
  field.outward = [nx / l, ny / l];
  const first = data.edge[0], last = data.edge[data.edge.length - 1], span = Math.hypot(last[0] - first[0], last[1] - first[1]) || 1;
  field.downward = [(last[0] - first[0]) / span, (last[1] - first[1]) / span];
  return field;
}

/** The pull of one cheek in its contour's terms, all in source px of rest space. */
export interface CheekPose {
  /** Outward move of the contour (>= 0). */
  out: number;
  /** Inward move of the skin inside the contour (>= 0); the contour stays. */
  in: number;
  /** Move of the skin inside the contour along it, positive downwards; the contour stays. */
  along: number;
  /** Centre of the pull along the contour (0..1). */
  centre: number;
  /** Depth -1..1 (towards the camera > 0) and the rest-space point and radius of its swell. */
  depth: number;
  pinch: Point;
  radius: number;
}

/** The contour's outward direction and the direction down along it. */
function axes(field: CheekField): { out: Point; down: Point } {
  const [nx, ny] = field.outward;
  const [dx, dy] = field.downward;
  return { out: [nx, ny], down: -ny * dx + nx * dy >= 0 ? [-ny, nx] : [ny, -nx] };
}

/** A rest-space offset within the cheek's travel: soft limits outwards, inwards and along the contour. */
export function limitPull(field: CheekField, travel: number, offset: Point): Point {
  const { out, down } = axes(field);
  const a = offset[0] * out[0] + offset[1] * out[1], b = offset[0] * down[0] + offset[1] * down[1];
  const soft = (v: number, m: number) => m > 0 ? m * Math.tanh(v / m) : 0;
  const n = a >= 0 ? soft(a, travel) : -soft(-a, CHEEK_LIMITS.inward * travel), t = soft(b, CHEEK_LIMITS.along * travel);
  return [out[0] * n + down[0] * t, out[1] * n + down[1] * t];
}

/** Splits a rest-space offset (within the travel, see limitPull) into the cheek's terms. */
export function cheekPose(field: CheekField, travel: number, offset: Point, pinchAlong: number, depth: number, pinch: Point, radius: number): CheekPose {
  const { out, down } = axes(field);
  const a = offset[0] * out[0] + offset[1] * out[1], b = offset[0] * down[0] + offset[1] * down[1];
  const along = clamp(b, -CHEEK_LIMITS.along * travel, CHEEK_LIMITS.along * travel);
  return {
    out: clamp(a, 0, travel),
    in: clamp(-a, 0, CHEEK_LIMITS.inward * travel),
    along,
    // the pinch's place, and a little of the pull along the contour, set where the arc is fullest
    centre: clamp(clamp(pinchAlong, 0.4, 0.6) + 0.5 * along / Math.max(field.length, 1), 0.35, 0.65),
    depth: clamp(depth, -1, 1),
    pinch,
    radius,
  };
}

/** Rest-space displacement of every moved vertex of a cheek layer for a pose, written into `out` (x, y per vertex). */
export function cheekDisplace(field: CheekField, pose: CheekPose, rest: ArrayLike<number>, out: Float64Array): void {
  out.fill(0);
  const swell = pose.depth * CHEEK_LIMITS.swell, r2 = pose.radius * pose.radius;
  for (let j = 0; j < field.index.length; j++) {
    const k = field.index[j], b = cheekBump(field.along[j], pose.centre);
    const [nx, ny] = field.normal[j], tx = -ny, ty = nx, down = tx * field.downward[0] + ty * field.downward[1] >= 0 ? 1 : -1;
    const fall = field.fall[j], inner = field.inner[j];
    let n = (pose.out - pose.in * inner) * b * fall;
    let t = pose.along * inner * b * fall * down;
    if (swell !== 0 && r2 > 0) {
      const qx = rest[k * 2] - pose.pinch[0], qy = rest[k * 2 + 1] - pose.pinch[1];
      const g = Math.exp(-(qx * qx + qy * qy) / r2) * fall * swell;
      const sn = (qx * nx + qy * ny) * g, st = (qx * tx + qy * ty) * g;
      n += sn > 0 ? sn : sn * inner;
      t += st * inner;
    }
    out[k * 2] = n * nx + t * tx;
    out[k * 2 + 1] = n * ny + t * ty;
  }
}

/** A damped spring per cheek: towards the held pull, then back to rest with the release speed. */
export class CheekSpring {
  offset: Point = [0, 0];
  velocity: Point = [0, 0];
  depth = 0;
  depthVelocity = 0;

  /** Advances by `dt` towards `target` (rest-space px) and `depth`; held cheeks follow fast, released ones settle with a little overshoot. */
  step(dt: number, target: Point, depth: number, held: boolean): void {
    const time = clamp(dt, 0, 0.05);
    if (!time) return;
    const omega = held ? 30 : 16, zeta = held ? 1 : 0.42;
    const count = Math.ceil(time * 240), h = time / count;
    for (let n = 0; n < count; n++) {
      for (const axis of [0, 1]) {
        this.velocity[axis] += (omega * omega * (target[axis] - this.offset[axis]) - 2 * zeta * omega * this.velocity[axis]) * h;
        this.offset[axis] += this.velocity[axis] * h;
      }
      this.depthVelocity += (omega * omega * (depth - this.depth) - 2 * zeta * omega * this.depthVelocity) * h;
      this.depth += this.depthVelocity * h;
    }
    if (!held && Math.hypot(...this.offset) < 0.02 && Math.hypot(...this.velocity) < 0.5 && Math.abs(this.depth) < 1e-3 && Math.abs(this.depthVelocity) < 0.01) {
      this.offset = [0, 0]; this.velocity = [0, 0]; this.depth = 0; this.depthVelocity = 0;
    }
  }

  /** At rest: nothing to draw. */
  get still(): boolean {
    return this.offset[0] === 0 && this.offset[1] === 0 && this.velocity[0] === 0 && this.velocity[1] === 0 && this.depth === 0 && this.depthVelocity === 0;
  }
}

/** Where a cheek is drawn now: a rest-space point on it, where that point is drawn and the local turn of the drawing. */
export interface CheekFrame { rest: Point; drawn: Point; angle: number }

interface Side {
  data: CheekEdge;
  field: CheekField;
  rest: ArrayLike<number>;
  spring: CheekSpring;
  held: boolean;
  offset: Point;
  depth: number;
  pinch: Point;
  pinchAlong: number;
  radius: number;
  displacement: Float64Array;
  moving: boolean;
}

const turn = (v: Point, angle: number): Point => {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c];
};

/** The cheeks of one avatar: held through the grab interface, moved by their springs, read by the cheek layers. */
export class CheekPull {
  private readonly sides = new Map<CheekSide, Side>();

  /** `rests`: the rest vertices (source px) of each side's cheek layer mesh; a side needs both its data and its mesh. */
  constructor(data: CheekData, rests: Partial<Record<CheekSide, ArrayLike<number>>>) {
    for (const side of CHEEK_SIDES) {
      const edge = data[side], rest = rests[side];
      if (!edge || !rest || !(edge.travel > 0)) continue;
      const field = cheekField(edge, side, rest);
      if (!field.index.length) continue;
      this.sides.set(side, {
        data: edge, field, rest, spring: new CheekSpring(), held: false, offset: [0, 0], depth: 0,
        pinch: [0, 0], pinchAlong: 0.5, radius: 0, displacement: new Float64Array(rest.length), moving: false,
      });
    }
  }

  /** Whether this side can be pulled. */
  has(side: CheekSide): boolean { return this.sides.has(side); }

  /** A rest-space point in the middle of the side's contour, for measuring where the cheek is drawn. */
  anchor(side: CheekSide): Point | null {
    const s = this.sides.get(side);
    if (!s) return null;
    const { edge } = s.data, i = Math.floor((edge.length - 1) / 2);
    return [edge[i][0], edge[i][1]];
  }

  /** Starts holding a cheek pressed at (x, y) (source px, as drawn). */
  hold(side: CheekSide, x: number, y: number, frame: CheekFrame): boolean {
    const s = this.sides.get(side);
    if (!s || !Number.isFinite(x) || !Number.isFinite(y)) return false;
    const back = turn([x - frame.drawn[0], y - frame.drawn[1]], -frame.angle);
    s.pinch = [frame.rest[0] + back[0], frame.rest[1] + back[1]];
    const hit = onEdge(s.data, side, s.pinch[0], s.pinch[1]);
    s.pinchAlong = hit.along;
    s.radius = Math.max(6, 0.7 * hit.reach);
    s.held = true; s.offset = [0, 0]; s.depth = 0;
    return true;
  }

  /** The held cheek's offset from the press (source px, as drawn) and depth (-1..1, towards the camera > 0). */
  move(side: CheekSide, dx: number, dy: number, depth: number): void {
    const s = this.sides.get(side);
    if (!s?.held || !Number.isFinite(dx) || !Number.isFinite(dy)) return;
    s.offset = [dx, dy];
    s.depth = Number.isFinite(depth) ? clamp(depth, -1, 1) : 0;
  }

  /** Lets go with a velocity (source px/s, as drawn; bounded by CHEEK_LIMITS.release). */
  release(side: CheekSide, vx: number, vy: number, angle: number): void {
    const s = this.sides.get(side);
    if (!s?.held) return;
    s.held = false;
    let v: Point = [Number.isFinite(vx) ? vx : 0, Number.isFinite(vy) ? vy : 0];
    const speed = Math.hypot(v[0], v[1]);
    if (speed > CHEEK_LIMITS.release) v = [v[0] * CHEEK_LIMITS.release / speed, v[1] * CHEEK_LIMITS.release / speed];
    const rest = turn(v, -angle);
    s.spring.velocity = [s.spring.velocity[0] + rest[0], s.spring.velocity[1] + rest[1]];
  }

  /** Clears every cheek at once, without springing back. */
  reset(): void {
    for (const s of this.sides.values()) { s.held = false; s.spring = new CheekSpring(); s.moving = false; s.displacement.fill(0); }
  }

  /** Advances the springs; `angles` is the local turn of each side's drawing (rest space to drawn). */
  step(dt: number, angles: Partial<Record<CheekSide, number>>): void {
    for (const [side, s] of this.sides) {
      const angle = angles[side] ?? 0;
      const target: Point = s.held ? limitPull(s.field, s.data.travel, turn(s.offset, -angle)) : [0, 0];
      s.spring.step(dt, target, s.held ? s.depth : 0, s.held);
      s.moving = !s.spring.still;
      if (!s.moving) continue;
      const pose = cheekPose(s.field, s.data.travel, s.spring.offset, s.pinchAlong, s.spring.depth, s.pinch, s.radius);
      cheekDisplace(s.field, pose, s.rest, s.displacement);
    }
  }

  /** This frame's rest-space displacement of each vertex of the side's cheek layer, or null while it is at rest. */
  displacement(side: CheekSide): Float64Array | null {
    const s = this.sides.get(side);
    return s?.moving ? s.displacement : null;
  }

  /** The swell for the renderer's light: rest-space centre, radius and depth (all 0 without one). */
  shading(side: CheekSide): [number, number, number, number] {
    const s = this.sides.get(side);
    return s?.moving && s.spring.depth !== 0 ? [s.pinch[0], s.pinch[1], s.radius, clamp(s.spring.depth, -1, 1)] : [0, 0, 0, 0];
  }
}
