// Limb chains at run time (ER4, WebGL runtime only): parameters, idle sway, impulses and the per-frame skinning of the
// limb layers. Nothing here exists for a rig without `limbs`.
//
//   limb:<name>:swing, limb:<name>:bend   -1..1, default 0 (times the limb's swing and bend limits)
//
// While the avatar idles (any motion mode but manual) a limb whose parameters are not set gets a slow, small sway
// (scaled by the sway gain). An impulse adds swing velocity to a damped spring on top of the parameter, which settles
// within about 3 s. The final value is clamped to -1..1.
import type { Point, Rig } from '../rig/types';
import { addCapped } from '../interaction/impulse.ts';
import { IMPULSE_LIMITS, LIMB_SWING_RATE } from '../interaction/limits.ts';
import { bendParameter, limbChain, limbFrame, limbWeight, placeOnChain, swingParameter, type LimbChain, type LimbWeight } from './chain.ts';

/** The parts of the rig engine (rig.js) the limbs read. */
export interface BaseDeformer {
  baseWeights(x: number, y: number, hair?: number): unknown;
  deformBase(x: number, y: number, weights: unknown, P: Record<string, number>, phys: unknown, out: number[]): number[];
}

export interface LimbMesh { rest: ArrayLike<number>; pos: { [index: number]: number; length: number } }

/** Idle sway amplitudes, parameter units, and the impulse spring. */
export const LIMB_IDLE = Object.freeze({ swing: 0.3, bend: 0.2 });
const SPRING = { frequency: 1.1, zeta: 0.35 };
const PROBE = 0.15; // of the first bone: how far from the root the base's own turn is measured

interface Attached { mesh: LimbMesh; weights: LimbWeight[]; base: (unknown | null)[] }
interface State {
  chain: LimbChain;
  meshes: Attached[];
  probes: { at: Point; weights: unknown }[];
  offset: number;
  velocity: number;
  phase: number;
}

const wrap = (angle: number) => Math.atan2(Math.sin(angle), Math.cos(angle));
const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));

export function createLimbs(rig: Rig, engine: BaseDeformer) {
  const limbs = rig.limbs ?? [];
  if (!limbs.length) return null;
  const states: State[] = limbs.map((limb, index) => {
    const chain = limbChain(limb);
    const r = Math.max(4, PROBE * chain.first);
    const probe = (x: number, y: number) => ({ at: [x, y] as Point, weights: engine.baseWeights(x, y, 0) });
    return {
      chain, meshes: [], offset: 0, velocity: 0, phase: index * 2.39996,
      probes: [probe(...chain.root), probe(chain.root[0] + chain.u1[0] * r, chain.root[1] + chain.u1[1] * r),
        probe(chain.root[0] - chain.u1[1] * r, chain.root[1] + chain.u1[0] * r)],
    };
  });
  const byName = new Map(states.map(state => [state.chain.name, state]));
  const k = (2 * Math.PI * SPRING.frequency) ** 2, c = 2 * SPRING.zeta * Math.sqrt(k);
  const tmp = [0, 0];

  /** The base drawing at the root this frame: where the root went and how much the drawing turned there. */
  function rootPose(state: State, P: Record<string, number>, phys: unknown): { root: Point; turn: number } {
    const moved = state.probes.map(({ at, weights }) => {
      engine.deformBase(at[0], at[1], weights, P, phys, tmp);
      return [tmp[0], tmp[1]] as Point;
    });
    const [root, along, side] = moved, [r0, a0, s0] = state.probes.map(p => p.at);
    const turnA = wrap(Math.atan2(along[1] - root[1], along[0] - root[0]) - Math.atan2(a0[1] - r0[1], a0[0] - r0[0]));
    const turnB = wrap(Math.atan2(side[1] - root[1], side[0] - root[0]) - Math.atan2(s0[1] - r0[1], s0[0] - r0[0]));
    return { root, turn: wrap(turnB + wrap(turnA - turnB) / 2) };
  }

  /** The current swing and bend (-1..1) of a limb: its parameters (or idle sway) plus the impulse spring. */
  function values(state: State, P: Record<string, number>): { swing: number; bend: number } {
    const name = state.chain.name;
    return {
      swing: clamp((Number.isFinite(P[swingParameter(name)]) ? P[swingParameter(name)] : 0) + state.offset, -1, 1),
      bend: clamp(Number.isFinite(P[bendParameter(name)]) ? P[bendParameter(name)] : 0, -1, 1),
    };
  }

  return {
    names: states.map(state => state.chain.name),
    chains: states.map(state => state.chain),
    /** Skin a limb layer mesh (rest/pos in source px) to the named limb. */
    attach(name: string, mesh: LimbMesh): void {
      const state = byName.get(name);
      if (!state) return;
      const weights: LimbWeight[] = [], base: (unknown | null)[] = [];
      for (let i = 0; i < mesh.rest.length; i += 2) {
        const w = limbWeight(state.chain, mesh.rest[i], mesh.rest[i + 1]);
        weights.push(w);
        base.push(w.root < 1 ? engine.baseWeights(mesh.rest[i], mesh.rest[i + 1], 0) : null);
      }
      state.meshes.push({ mesh, weights, base });
    },
    /** Idle sway for limbs whose parameters are not set (gain 0 turns it off, as in manual mode). */
    idle(P: Record<string, number>, time: number, gain: number): void {
      for (const state of states) {
        const g = Number.isFinite(gain) ? Math.max(0, gain) : 0, t = time, f = state.phase;
        const swing = swingParameter(state.chain.name), bend = bendParameter(state.chain.name);
        if (P[swing] === undefined) P[swing] = g * LIMB_IDLE.swing * (0.6 * Math.sin(0.55 * t + f) + 0.4 * Math.sin(1.3 * t + 2.1 * f));
        if (P[bend] === undefined) P[bend] = g * LIMB_IDLE.bend * (0.6 * Math.sin(0.45 * t + 1.7 * f) + 0.4 * Math.sin(1.1 * t + f));
      }
    },
    /** Advance the impulse springs. */
    step(dt: number): void {
      const time = clamp(dt, 0, 0.05);
      if (!time) return;
      const count = Math.ceil(time * 240), h = time / count;
      for (const state of states) {
        if (!state.offset && !state.velocity) continue;
        for (let n = 0; n < count; n++) {
          state.velocity += (-k * state.offset - c * state.velocity) * h;
          state.offset += state.velocity * h;
        }
      }
    },
    /** Deform every attached limb mesh for this frame. */
    deform(P: Record<string, number>, phys: unknown): void {
      for (const state of states) {
        if (!state.meshes.length) continue;
        const { root, turn } = rootPose(state, P, phys);
        const { swing, bend } = values(state, P);
        const frame = limbFrame(state.chain, { root, turn, swing: swing * state.chain.swingLimit, bend: bend * state.chain.bendLimit });
        const out = [0, 0];
        for (const { mesh, weights, base } of state.meshes) {
          for (let v = 0; v < weights.length; v++) {
            const x = mesh.rest[v * 2], y = mesh.rest[v * 2 + 1], w = weights[v];
            placeOnChain(state.chain, frame, x, y, w.joint, out);
            if (w.root < 1) {
              engine.deformBase(x, y, base[v], P, phys, tmp);
              out[0] = tmp[0] + (out[0] - tmp[0]) * w.root;
              out[1] = tmp[1] + (out[1] - tmp[1]) * w.root;
            }
            mesh.pos[v * 2] = out[0]; mesh.pos[v * 2 + 1] = out[1];
          }
        }
      }
    },
    /**
     * Push a limb: (vx, vy) source px/s at its end becomes swing velocity about the root. False for an unknown limb or a
     * push without a sideways part.
     */
    impulse(name: string, vx: number, vy: number): boolean {
      const state = byName.get(name);
      if (!state || !Number.isFinite(vx) || !Number.isFinite(vy)) return false;
      const speed = Math.hypot(vx, vy), scale = speed > IMPULSE_LIMITS.limb ? IMPULSE_LIMITS.limb / speed : 1;
      const rx = state.chain.end[0] - state.chain.root[0], ry = state.chain.end[1] - state.chain.root[1];
      const reach = rx * rx + ry * ry;
      if (!(reach > 1) || !(state.chain.swingLimit > 0)) return false;
      // turning by w (rad/s) moves the end at w * (-ry, rx): the push's part across the limb
      const spin = (rx * vy - ry * vx) * scale / reach;
      if (Math.abs(spin) < 1e-9) return false;
      state.velocity = addCapped(state.velocity, clamp(spin / state.chain.swingLimit, -LIMB_SWING_RATE, LIMB_SWING_RATE), LIMB_SWING_RATE);
      return true;
    },
    /** Current swing and bend including the impulse spring (for tests and tools). */
    values(name: string, P: Record<string, number>): { swing: number; bend: number } | null {
      const state = byName.get(name);
      return state ? values(state, P) : null;
    },
  };
}

export type Limbs = NonNullable<ReturnType<typeof createLimbs>>;
