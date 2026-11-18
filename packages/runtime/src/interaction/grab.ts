// Sustained grabs (T-ER12, WebGL runtime only): a part held at the pointer until it is let go. Each grab has a handle;
// moves give the pointer's offset from the press in source px (and, for cheeks, a depth), and a release hands the part
// back to its own dynamics with a bounded velocity. Nothing here runs without a grab, so avatars draw as before.
//
//   cheek      the cheek layers (./cheek.ts): the contour is pulled outwards, the skin inside follows; depth swells it
//   strand     the hair strand nearest the press: the chain from its root to the held node is placed at the pointer
//              (a position constraint before each physics step), its tip at most the strand's `reach` from where it
//              hangs; the rest of the strand follows by its own springs
//   cloth      the garment columns at `at` (or nearest the press) turn so their hem points at the pointer, within the
//              garment's `reach` and its own hem travel (`max`); the hem is never lifted beyond the turn it already makes
//              when it swings
//
// Strands and garments are held only when the rig gives them a `reach` (M11 builds); without it they are refused, and
// a drag flicks them as before (impulse).
//   limb       the limb chain swings (and bends, when held below its middle joint) towards the pointer, within its limits
//   accessory  a hanging tassel (rig.accessories) points from its knot towards the pointer, within its swing limit
//
// Unknown targets, targets this avatar does not have and non-finite input are refused or ignored.
import type { Point, Rig } from '../rig/types';
import { CLOTH_FALLOFF, IMPULSE_LIMITS } from './limits.ts';
import { hemPositions, type GarmentState, type ImpulseTarget } from './impulse.ts';
import type { CheekPull, CheekSide, CheekFrame } from './cheek.ts';

import type { SoftSurfaces, SurfaceTarget } from './soft-surfaces';

export type GrabTarget =
  | SurfaceTarget
  | { kind: 'cheek'; side: CheekSide }
  | { kind: 'strand'; name: string }
  | { kind: 'cloth'; name: string; at?: number }
  | { kind: 'limb'; name: string }
  | { kind: 'accessory'; name: string };

/** A garment of cloth.js as a grab reads and turns it. */
export interface HeldGarment extends GarmentState {
  angle: Float64Array;
  lowerAngle: Float64Array;
  pendulums: { angles: number[]; velocities: number[] }[];
}
interface StrandState { p: number[][] | null; v: number[][] }
interface TasselState { t: { name: string; pivot: number[]; tip: number[] }; dir: number[]; L1: number; L2: number; anchor: number[]; st?: { th: number[]; w: number[] } }
interface LimbChainLike { name: string; root: Point; middle: Point; end: Point; swingLimit: number; bendLimit: number }

export interface GrabParts {
  rig: Rig;
  physics: {
    strands: StrandState[]; chains: TasselState[]; hairAnchor(at: Point, P: Record<string, number>): number[];
    /** Per strand, the tip offset (px) a grab lets it show (physics.js): set while held, it fades after the release. */
    held?: number[];
  };
  /** The garment of that name (cloth.js), or null. */
  garment(name: string): HeldGarment | null;
  limbs: { chains: LimbChainLike[]; values(name: string, P: Record<string, number>): { swing: number; bend: number } | null } | null;
  cheeks: CheekPull | null;
  surfaces?: SoftSurfaces;
  /** Where a cheek is drawn now (see CheekPull.hold). */
  cheekFrame(side: CheekSide): CheekFrame | null;
  /** Where a rest-space point of the base is drawn with the latest parameters. */
  place(point: Point): Point;
  /** The parameters of the latest frame. */
  parameters(): Record<string, number>;
  impulse(target: ImpulseTarget, vx: number, vy: number): void;
}

/** Tassels swing at most this far from hanging (radians), as in physics.js; a release turns them at most this fast. */
const TASSEL_MAX = 0.55, TASSEL_RELEASE = 6;
/** A released limb settles with the limb chains' own impulse spring (../limbs/runtime.ts). */
const LIMB_SPRING = { omega: 2 * Math.PI * 1.1, zeta: 0.35 };
/** How much a held limb bends for each share its grip comes closer to the root. */
const LIMB_BEND_GAIN = 6;

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
const finite = (...values: number[]) => values.every(Number.isFinite);

/** The 4 physics nodes of a strand (physics.js resamples other node counts the same way). */
function strandNodes(nodes: Point[]): Point[] {
  if (nodes.length === 4) return nodes;
  return [0, 1, 2, 3].map(i => {
    const f = i / 3 * (nodes.length - 1), k = Math.min(nodes.length - 2, Math.floor(f)), t = f - k;
    return [nodes[k][0] + (nodes[k + 1][0] - nodes[k][0]) * t, nodes[k][1] + (nodes[k + 1][1] - nodes[k][1]) * t] as Point;
  });
}

/** Closest point of a polyline: segment index and position on it (0..1) and the distance. */
function nearest(points: number[][], x: number, y: number): { segment: number; u: number; distance: number } {
  let best = { segment: 0, u: 0, distance: Infinity };
  for (let i = 0; i < points.length - 1; i++) {
    const [ax, ay] = points[i], vx = points[i + 1][0] - ax, vy = points[i + 1][1] - ay, span = vx * vx + vy * vy;
    const u = span > 0 ? clamp(((x - ax) * vx + (y - ay) * vy) / span, 0, 1) : 0;
    const distance = Math.hypot(x - ax - vx * u, y - ay - vy * u);
    if (distance < best.distance) best = { segment: i, u, distance };
  }
  return best;
}

interface Held {
  key: string;
  target: GrabTarget;
  press: Point;
  offset: Point;
  depth: number;
  strand?: { index: number; node: number; grip: Point };
  tassel?: { index: number; length: number };
  cloth?: { garment: HeldGarment; at: number; name: string };
  /** `left`: how far the held swing and bend are from where the limb goes without the grab (for a smooth release). */
  limb?: { chain: LimbChainLike; from: Point; swing: number; bend: number; second: boolean; left: { swing: number; bend: number } };
}

/** The grab interface of one avatar. */
export function createGrabs(parts: GrabParts) {
  const { rig, physics } = parts;
  const held = new Map<number, Held>();
  // released limbs: their swing and bend still differ from the parameters and settle with the limb spring
  const settling = new Map<string, { swing: number; bend: number; vs: number; vb: number }>();
  let next = 1;

  const keyOf = (t: GrabTarget) => t.kind === 'cheek' || t.kind === 'chest' ? `${t.kind}:${t.side}` : `${t.kind}:${t.name}`;
  // the strands of that name a grab can hold: those with a reach
  const strandIndices = (name: string) => (rig.strands ?? []).flatMap((s, i) => s.name === name && s.reach! > 0 ? [i] : []);
  /** How far the named garment's hem may be held from where it hangs (its reach within its travel), or 0. */
  const hemReach = (name: string, g: HeldGarment) => {
    const reach = (rig.cloth ?? []).find(c => c.name === name)?.reach;
    return reach! > 0 ? Math.min(reach!, g.tuning.max) : 0;
  };

  function valid(target: unknown): target is GrabTarget {
    if (!target || typeof target !== 'object') return false;
    const t = target as Record<string, unknown>;
    if (t.kind === 'cheek' || t.kind === 'chest') return t.side === 'left' || t.side === 'right';
    if (!['strand', 'cloth', 'limb', 'accessory', 'fabric'].includes(t.kind as string) || typeof t.name !== 'string') return false;
    return t.kind !== 'cloth' || t.at === undefined || (typeof t.at === 'number' && Number.isFinite(t.at));
  }

  /** Whether the target can be grabbed on this avatar. */
  function can(target: unknown): boolean {
    if (!valid(target)) return false;
    switch (target.kind) {
      case 'chest': case 'fabric': return !!parts.surfaces?.can(target);
      case 'cheek': return !!parts.cheeks?.has(target.side);
      case 'strand': return strandIndices(target.name).some(i => physics.strands[i]);
      case 'cloth': { const g = parts.garment(target.name); return !!g && hemReach(target.name, g) > 0; }
      case 'limb': return !!parts.limbs?.chains.some(c => c.name === target.name);
      case 'accessory': return physics.chains.some(c => c.t.name === target.name);
    }
  }

  function grab(target: unknown, x: number, y: number): number | null {
    if (!valid(target) || !finite(x, y) || !can(target)) return null;
    const entry: Held = { key: keyOf(target), target, press: [x, y], offset: [0, 0], depth: 0 };
    if (target.kind === 'chest' || target.kind === 'fabric') {
      if (!parts.surfaces!.hold(target, x, y)) return null;
    } else if (target.kind === 'cheek') {
      const frame = parts.cheekFrame(target.side);
      if (!frame || !parts.cheeks!.hold(target.side, x, y, frame)) return null;
    } else if (target.kind === 'strand') {
      // the strand of that name nearest the press, held at its node nearest the press (never the root)
      let best: { index: number; node: number; distance: number; at: number[] } | null = null;
      for (const index of strandIndices(target.name)) {
        const state = physics.strands[index];
        const points = state.p ?? strandNodes(rig.strands![index].nodes);
        const hit = nearest(points, x, y);
        if (!best || hit.distance < best.distance) {
          const node = clamp(Math.round(hit.segment + hit.u), 1, points.length - 1);
          best = { index, node, distance: hit.distance, at: points[node] };
        }
      }
      if (!best) return null;
      entry.strand = { index: best.index, node: best.node, grip: [best.at[0] - x, best.at[1] - y] };
    } else if (target.kind === 'cloth') {
      const garment = parts.garment(target.name)!;
      entry.cloth = { garment, name: target.name, at: target.at === undefined ? hemAt(garment, x, y) : clamp(target.at, 0, 1) };
    } else if (target.kind === 'limb') {
      const chain = parts.limbs!.chains.find(c => c.name === target.name)!;
      const P = parts.parameters(), now = parts.limbs!.values(chain.name, P) ?? { swing: 0, bend: 0 };
      const root = parts.place(chain.root), middle = chain.middle;
      // held below the middle joint (along the limb), a pull towards the root also bends it
      const lx = chain.end[0] - chain.root[0], ly = chain.end[1] - chain.root[1];
      const mx = middle[0] - chain.root[0], my = middle[1] - chain.root[1];
      const second = ((x - root[0]) * lx + (y - root[1]) * ly) > (mx * lx + my * ly);
      // a limb still settling from an earlier release is held where it is now
      entry.limb = { chain, from: [x - root[0], y - root[1]], swing: now.swing, bend: now.bend, second, left: { swing: 0, bend: 0 } };
      settling.delete(chain.name);
    } else {
      const index = physics.chains.findIndex(c => c.t.name === target.name);
      const chain = physics.chains[index], anchor = chain.st ? chain.anchor : chain.t.pivot;
      entry.tassel = { index, length: clamp(Math.hypot(x - anchor[0], y - anchor[1]), 0.3 * (chain.L1 + chain.L2), chain.L1 + chain.L2) };
    }
    // a new grab of the same part takes it over
    for (const [handle, other] of held) if (other.key === entry.key) held.delete(handle);
    const handle = next++;
    held.set(handle, entry);
    return handle;
  }

  /** Position (0..1) along a garment's hem nearest to (x, y), by its columns' rest hem points. */
  function hemAt(g: HeldGarment, x: number, y: number): number {
    const positions = hemPositions(g);
    let best = Infinity, at = 0.5;
    for (let k = 0; k < g.cols; k++) {
      const i = ((g.rows - 1) * g.cols + k) * 2, d = Math.hypot(g.rest[i] - x, g.rest[i + 1] - y);
      if (d < best) { best = d; at = positions[k]; }
    }
    return at;
  }

  function move(handle: number, dx: number, dy: number, depth = 0): void {
    const entry = held.get(handle);
    if (!entry || !finite(dx, dy)) return;
    entry.offset = [dx, dy];
    entry.depth = Number.isFinite(depth) ? clamp(depth, -1, 1) : 0;
    if (entry.target.kind === 'chest' || entry.target.kind === 'fabric') parts.surfaces!.move(entry.target, dx, dy, entry.depth);
    if (entry.target.kind === 'cheek') parts.cheeks!.move(entry.target.side, dx, dy, entry.depth);
  }

  function release(handle: number, vx = 0, vy = 0): void {
    const entry = held.get(handle);
    if (!entry) return;
    held.delete(handle);
    const v: Point = [Number.isFinite(vx) ? vx : 0, Number.isFinite(vy) ? vy : 0];
    const t = entry.target;
    if (t.kind === 'chest' || t.kind === 'fabric') {
      parts.surfaces!.release(t, v[0], v[1]);
    } else if (t.kind === 'cheek') {
      parts.cheeks!.release(t.side, v[0], v[1], parts.cheekFrame(t.side)?.angle ?? 0);
    } else if (t.kind === 'strand') {
      parts.impulse({ kind: 'strand', name: t.name }, v[0], v[1]);
    } else if (t.kind === 'cloth') {
      parts.impulse({ kind: 'cloth', name: t.name, at: entry.cloth!.at }, v[0], v[1]);
    } else if (t.kind === 'limb') {
      const { chain, left } = entry.limb!;
      // the limb's impulse rule: the push's part across the limb turns it about the root
      const lx = chain.end[0] - chain.root[0], ly = chain.end[1] - chain.root[1], reach = lx * lx + ly * ly;
      const speed = Math.hypot(v[0], v[1]), k = speed > IMPULSE_LIMITS.limb ? IMPULSE_LIMITS.limb / speed : 1;
      const spin = reach > 1 && chain.swingLimit > 0 ? (lx * v[1] - ly * v[0]) * k / reach / chain.swingLimit : 0;
      settling.set(chain.name, { swing: left.swing, bend: left.bend, vs: clamp(spin, -8, 8), vb: 0 });
    } else {
      const chain = physics.chains[entry.tassel!.index];
      if (chain.st) {
        const anchor = chain.anchor, r: Point = [entry.press[0] + entry.offset[0] - anchor[0], entry.press[1] + entry.offset[1] - anchor[1]];
        const reach = r[0] * r[0] + r[1] * r[1];
        const spin = reach > 1 ? clamp((r[0] * v[1] - r[1] * v[0]) / reach, -TASSEL_RELEASE, TASSEL_RELEASE) : 0;
        chain.st.w = [spin, spin];
      }
    }
  }

  function releaseAll(): void {
    for (const handle of [...held.keys()]) release(handle);
    settling.clear();
    parts.cheeks?.reset();
  }

  /** Limb parameters while held or settling (after the motion and the caller's parameters). */
  function parameters(P: Record<string, number>): void {
    for (const entry of held.values()) {
      if (!entry.limb) continue;
      const { chain, from, swing, bend, second } = entry.limb;
      const root = parts.place(chain.root);
      const to: Point = [entry.press[0] + entry.offset[0] - root[0], entry.press[1] + entry.offset[1] - root[1]];
      const turn = Math.hypot(...from) > 1 && Math.hypot(...to) > 1 ? wrap(Math.atan2(to[1], to[0]) - Math.atan2(from[1], from[0])) : 0;
      const wantSwing = chain.swingLimit > 0 ? clamp(swing + turn / chain.swingLimit, -1, 1) : swing;
      const closer = second && Math.hypot(...from) > 1 ? 1 - Math.hypot(...to) / Math.hypot(...from) : 0;
      const wantBend = chain.bendLimit > 0 ? clamp(bend + closer * LIMB_BEND_GAIN, -1, 1) : bend;
      // the limb's own impulse spring adds to the swing parameter: hold the sum where the pointer is
      const spring = parts.limbs!.values(chain.name, {})?.swing ?? 0;
      const swingKey = `limb:${chain.name}:swing`, bendKey = `limb:${chain.name}:bend`;
      const natural = { swing: Number.isFinite(P[swingKey]) ? P[swingKey] : 0, bend: Number.isFinite(P[bendKey]) ? P[bendKey] : 0 };
      P[swingKey] = wantSwing - spring;
      P[bendKey] = wantBend;
      entry.limb.left = { swing: wantSwing - spring - natural.swing, bend: wantBend - natural.bend };
    }
    for (const [name, s] of settling) {
      P[`limb:${name}:swing`] = (Number.isFinite(P[`limb:${name}:swing`]) ? P[`limb:${name}:swing`] : 0) + s.swing;
      P[`limb:${name}:bend`] = (Number.isFinite(P[`limb:${name}:bend`]) ? P[`limb:${name}:bend`] : 0) + s.bend;
    }
  }

  /** Advances the settling limbs. */
  function step(dt: number): void {
    const time = clamp(dt, 0, 0.05);
    if (!time) return;
    const { omega, zeta } = LIMB_SPRING, count = Math.ceil(time * 240), h = time / count;
    for (const [name, s] of settling) {
      for (let n = 0; n < count; n++) {
        s.vs += (-omega * omega * s.swing - 2 * zeta * omega * s.vs) * h; s.swing += s.vs * h;
        s.vb += (-omega * omega * s.bend - 2 * zeta * omega * s.vb) * h; s.bend += s.vb * h;
      }
      if (Math.abs(s.swing) + Math.abs(s.bend) < 1e-4 && Math.abs(s.vs) + Math.abs(s.vb) < 1e-3) settling.delete(name);
    }
  }

  /** Strands and tassels at the pointer, before the physics step. */
  function beforePhysics(P: Record<string, number>): void {
    for (const entry of held.values()) {
      const goal: Point = [entry.press[0] + entry.offset[0], entry.press[1] + entry.offset[1]];
      if (entry.strand) {
        const { index, node, grip } = entry.strand, state = physics.strands[index];
        if (!state?.p) continue;
        const strand = rig.strands![index], nodes = strandNodes(strand.nodes);
        const anchor = physics.hairAnchor(nodes[node], P);
        // the held node at the pointer, no further from where it hangs than the strand's reach lets it
        let dx = goal[0] + grip[0] - anchor[0], dy = goal[1] + grip[1] - anchor[1];
        const limit = Math.max(2, strand.reach! * node / 3), length = Math.hypot(dx, dy);
        if (physics.held) physics.held[index] = strand.reach!;
        if (length > limit) { dx *= limit / length; dy *= limit / length; }
        for (let j = 1; j <= node; j++) {
          const a = j === node ? anchor : physics.hairAnchor(nodes[j], P), share = j / node;
          state.p[j] = [a[0] + dx * share, a[1] + dy * share];
          state.v[j] = [0, 0];
        }
      } else if (entry.tassel) {
        const chain = physics.chains[entry.tassel.index];
        if (!chain.st) continue;
        const r = [goal[0] - chain.anchor[0], goal[1] - chain.anchor[1]];
        if (Math.hypot(r[0], r[1]) < 1) continue;
        const angle = clamp(wrap(Math.atan2(r[1], r[0]) - Math.atan2(chain.dir[1], chain.dir[0])), -TASSEL_MAX, TASSEL_MAX);
        chain.st.th = [angle, angle];
        chain.st.w = [0, 0];
      }
    }
  }

  /** Garment columns at the pointer, before the cloth step. */
  function beforeCloth(): void {
    for (const entry of held.values()) {
      if (!entry.cloth) continue;
      const { garment: g, at, name } = entry.cloth, reach = hemReach(name, g);
      const positions = hemPositions(g), spacing = g.cols > 1 ? 1 / (g.cols - 1) : 1;
      const [dx, dy] = entry.offset;
      for (let k = 0; k < g.cols; k++) {
        if (g.pinned[k]) continue;
        const places = Math.abs(positions[k] - at) / spacing;
        const w = places < CLOTH_FALLOFF ? Math.cos(Math.PI / 2 * places / CLOTH_FALLOFF) : 0;
        if (w <= 0) continue;
        const top = k * 2, bottom = ((g.rows - 1) * g.cols + k) * 2;
        const rx = g.rest[bottom] - g.rest[top], ry = g.rest[bottom + 1] - g.rest[top + 1], chord = rx * rx + ry * ry;
        const length = g.grid.custom ? g.grid.lengths?.[k] ?? Math.sqrt(chord) : Math.sqrt(chord);
        if (!(chord > 1) || !(length > 1)) continue;
        // the turn that moves the hem by the pointer's part across the column (cloth.js: dθ moves it by dθ · (ry, −rx)),
        // within the angle at which the hem reaches the garment's reach (at most its travel)
        const limit = 0.95 * 2 * Math.asin(Math.min(0.95, reach / (2 * length)));
        const theta = clamp((dx * ry - dy * rx) / chord, -limit, limit);
        if (g.grid.custom) {
          const pendulum = g.pendulums[k];
          if (!pendulum) continue;
          pendulum.angles = pendulum.angles.map(a => a + (theta - a) * w);
          pendulum.velocities = pendulum.velocities.map(v => v * (1 - w));
        } else {
          g.angle[k] += (theta - g.angle[k]) * w; g.lowerAngle[k] += (theta - g.lowerAngle[k]) * w;
          g.angularVelocity[k] *= 1 - w; g.lowerAngularVelocity[k] *= 1 - w;
        }
      }
    }
  }

  return {
    grab, move, release, releaseAll, can, parameters, step, beforePhysics, beforeCloth,
    /** Whether anything is held or still settling (limbs). */
    get active() { return held.size > 0 || settling.size > 0; },
  };
}

export type Grabs = ReturnType<typeof createGrabs>;
