// Generic limb chains (ER4, WebGL runtime only): the geometry of one two-bone limb and its skinning, as pure functions.
//
// A limb is cut into its own layer. Its root joint follows the base drawing at that point (position and the drawing's
// own rotation there), the first bone turns about the root by the swing and the second bone about the middle joint by
// the bend. Every vertex is placed by both bones and blended across the middle joint along the bones' direction (the
// signed distance along the bisector of the two bones), never by image height. Near the root the limb's own motion
// fades in over ROOT_BLEND of the first bone, so the cut where the limb meets the body moves exactly like the base there.
import type { Limb, Point } from '../rig/types';

/** Default swing and bend limits in degrees. */
export const LIMB_DEFAULTS = Object.freeze({ swing: 4, bend: 6 });
/** Largest allowed limits in degrees. */
export const LIMB_LIMITS = Object.freeze({ swing: 10, bend: 12 });
/** Share of the first bone over which the chain's own motion fades in from the root. */
export const ROOT_BLEND = 0.35;
/** Half-width of the blend between the bones at the middle joint, as a share of the shorter bone. */
export const JOINT_BLEND = 0.18;
/** A bend below this angle (radians) counts as straight: positive bend then turns the second bone clockwise. */
const STRAIGHT = 5 * Math.PI / 180;

export const swingParameter = (name: string) => `limb:${name}:swing`;
export const bendParameter = (name: string) => `limb:${name}:bend`;

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const smooth = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / Math.max(1e-9, b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

export interface LimbChain {
  name: string;
  limb: Limb;
  root: Point;
  middle: Point;
  end: Point;
  /** Unit directions of the two bones and their lengths. */
  u1: Point;
  u2: Point;
  first: number;
  second: number;
  /** Unit bisector of the two bone directions: the blend across the middle joint runs along it. */
  bisector: Point;
  /** +1 or -1: positive bend increases the bend the limb is drawn with (clockwise when it is drawn straight). */
  bendSign: number;
  /** Limits in radians. */
  swingLimit: number;
  bendLimit: number;
  /** Distances in source px: the root fade-in and the half-width of the joint blend. */
  rootBlend: number;
  jointBlend: number;
}

export function limbChain(limb: Limb): LimbChain {
  const [root, middle, end] = limb.joints;
  const first = Math.hypot(middle[0] - root[0], middle[1] - root[1]);
  const second = Math.hypot(end[0] - middle[0], end[1] - middle[1]);
  const u1: Point = [(middle[0] - root[0]) / first, (middle[1] - root[1]) / first];
  const u2: Point = [(end[0] - middle[0]) / second, (end[1] - middle[1]) / second];
  const sum = Math.hypot(u1[0] + u2[0], u1[1] + u2[1]);
  const bisector: Point = [(u1[0] + u2[0]) / sum, (u1[1] + u2[1]) / sum];
  const cross = u1[0] * u2[1] - u1[1] * u2[0];
  const turn = Math.atan2(cross, u1[0] * u2[0] + u1[1] * u2[1]);
  const swing = clamp(Number.isFinite(limb.swing) ? limb.swing! : LIMB_DEFAULTS.swing, 0, LIMB_LIMITS.swing);
  const bend = clamp(Number.isFinite(limb.bend) ? limb.bend! : LIMB_DEFAULTS.bend, 0, LIMB_LIMITS.bend);
  return {
    name: limb.name, limb, root, middle, end, u1, u2, first, second, bisector,
    bendSign: Math.abs(turn) < STRAIGHT ? 1 : Math.sign(turn),
    swingLimit: swing * Math.PI / 180, bendLimit: bend * Math.PI / 180,
    rootBlend: ROOT_BLEND * first, jointBlend: JOINT_BLEND * Math.min(first, second),
  };
}

/** Per-vertex weights from the rest position: `root` 0 follows the base, 1 the chain; `joint` 0 is the first bone, 1 the second. */
export interface LimbWeight { root: number; joint: number }

export function limbWeight(chain: LimbChain, x: number, y: number): LimbWeight {
  const along = (x - chain.root[0]) * chain.u1[0] + (y - chain.root[1]) * chain.u1[1];
  const across = (x - chain.middle[0]) * chain.bisector[0] + (y - chain.middle[1]) * chain.bisector[1];
  return { root: smooth(0, chain.rootBlend, along), joint: smooth(-chain.jointBlend, chain.jointBlend, across) };
}

/** Where the chain is this frame: the root after the base deformation, the base's own turn there and the angles (radians). */
export interface LimbPose { root: Point; turn: number; swing: number; bend: number }

/** Bone transforms of one frame: v -> root + R1 (v - rest root) on the first bone, v -> middle + R2 (v - rest middle) on the second. */
export interface LimbFrame { root: Point; cos1: number; sin1: number; middle: Point; cos2: number; sin2: number }

export function limbFrame(chain: LimbChain, pose: LimbPose): LimbFrame {
  const a1 = pose.turn + pose.swing, a2 = a1 + chain.bendSign * pose.bend;
  const cos1 = Math.cos(a1), sin1 = Math.sin(a1);
  const dx = chain.middle[0] - chain.root[0], dy = chain.middle[1] - chain.root[1];
  const middle: Point = [pose.root[0] + dx * cos1 - dy * sin1, pose.root[1] + dx * sin1 + dy * cos1];
  return { root: pose.root, cos1, sin1, middle, cos2: Math.cos(a2), sin2: Math.sin(a2) };
}

/** The chain's own placement of a rest vertex (before the root fade-in), written to `out`. */
export function placeOnChain(chain: LimbChain, frame: LimbFrame, x: number, y: number, joint: number, out: number[]): number[] {
  const ax = x - chain.root[0], ay = y - chain.root[1];
  const px = frame.root[0] + ax * frame.cos1 - ay * frame.sin1, py = frame.root[1] + ax * frame.sin1 + ay * frame.cos1;
  if (joint <= 0) { out[0] = px; out[1] = py; return out; }
  const bx = x - chain.middle[0], by = y - chain.middle[1];
  const qx = frame.middle[0] + bx * frame.cos2 - by * frame.sin2, qy = frame.middle[1] + bx * frame.sin2 + by * frame.cos2;
  out[0] = px + (qx - px) * joint; out[1] = py + (qy - py) * joint;
  return out;
}
