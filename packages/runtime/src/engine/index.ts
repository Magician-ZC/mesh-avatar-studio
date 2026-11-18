import type { LightingSettings } from '../lighting/settings';
import type { Rig, ClothTuning } from '../rig/types';
import { parseRig } from '../rig/validate';
import { createMeshAvatarImpl } from './createMeshAvatar.js';
import { OVERLAY_EXPRESSIONS } from './expression-overlay.js';
import type { LipSyncSettings } from './lip-sync';
import type { ExpressionSettings } from './expression-settings';
import type { ImpulseTarget } from '../interaction/impulse';
import type { GrabTarget } from '../interaction/grab';

export type { ImpulseTarget, GrabTarget };
export type LiveExpression = keyof typeof OVERLAY_EXPRESSIONS;
export const LIVE_EXPRESSIONS = Object.keys(OVERLAY_EXPRESSIONS) as LiveExpression[];

export interface MeshAvatarOptions {
  rig: Rig;
  assetsBase?: string;
  assets?: Record<string, string>;
  manual?: boolean;
  padTop?: number;
  padSide?: number;
  fit?: 'contain' | 'cover';
  preserveMouthForm?: boolean;
  lipSync?: LipSyncSettings;
  /** Optional per-expression settings for live expressions, such as `{ surprise: { mouthOpen: 0.45 } }`. */
  expressions?: ExpressionSettings;
}
export interface MeshAvatar {
  readonly motions: { id: string; label: string; idle: boolean }[];
  setLighting(settings: Partial<LightingSettings>): void;
  getLightingStats(): { normalMs: number; computedLayers: number; cachedLayers: number };
  /** Optional per-parameter weights multiply weight and blend against underlying motion; null clears them. */
  setParameters(parameters: Record<string, number>, weight?: number, weights?: Record<string, number> | null): void;
  getParameters(): Record<string, number>;
  setVoiceLevel(value: number): void;
  /** Partial mouth-movement settings; omitted fields retain their current values. */
  setLipSync(settings: LipSyncSettings): void;
  setSpeaking(on: boolean): void;
  setEmotion(tag: string | null, options?: { playMotion?: boolean }): void;
  setExpression(name: LiveExpression): void;
  /** Show the named overlay layers from built/overlays/overlays.json and hide the rest; unknown names are ignored. */
  setOverlays(names: string[]): void;
  getExpression(): { name: LiveExpression; active: boolean };
  setTalkGain(gain: number): void;
  play(id: string): void;
  speakKana(text: string, options?: { speed?: number; loop?: boolean }): void;
  holdMouth(vowel: 'a' | 'i' | 'u' | 'e' | 'o' | 'n'): void;
  stopLipSync(): void;
  getLipSyncState(): { active: boolean; open: number; form: number };
  setAutoIdle(on: boolean): void;
  setAutoMotion(on: boolean): void;
  setSwayGain(gain: number): void;
  /** Override garment physics without rebuilding layers; null applies to every garment. */
  setClothTuning(name: string | null, tuning: Partial<ClothTuning>): void;
  onMotion(listener: (id: string | null) => void): () => void;
  /**
   * Map viewport coordinates (such as `PointerEvent.clientX/Y`) to source image pixels, following the canvas position,
   * its device pixel ratio, padding and fit. Returns null before the first frame is drawn. WebGL runtime only.
   */
  toImagePoint(clientX: number, clientY: number): [number, number] | null;
  /**
   * Map a source image pixel to viewport coordinates: the inverse of `toImagePoint`. With `view.rotation` both follow the
   * rotated drawing. Returns null before the first frame is drawn. WebGL runtime only.
   */
  toClientPoint(x: number, y: number): [number, number] | null;
  /**
   * Give a part an instantaneous velocity in source pixels per second; its springs or pendulums then let it settle.
   * Each kind of target has an upper bound, and larger values are truncated. Unknown targets are ignored; a `limb`
   * target swings that limb chain about its root. For `cloth`, `at` (0–1 along the hem) pushes the nearest columns; without
   * it the whole hem moves. WebGL runtime only.
   */
  impulse(target: ImpulseTarget, vx: number, vy: number): void;
  /**
   * Whether this avatar can pull a cheek: it was built with cheek layers (`build-layers.py --cheek-pull`) and they
   * loaded. False for other avatars, which draw exactly as before. For one side use
   * `canGrab({ kind: 'cheek', side })`. WebGL runtime only.
   */
  readonly cheekPull: boolean;
  /**
   * Hold a part at the pointer until `releaseGrab`: `x, y` is the press in source pixels (see `toImagePoint`). Returns a
   * positive handle, or null when this avatar cannot grab the target (it lacks the data, the name is unknown or the
   * input is not finite). A new grab of the same part replaces the old one, whose handle then goes stale.
   * - `cheek`: the cheek layers of that side (left/right in the image; see `cheekPull`). A pull moves the face's outer
   *   contour outwards in an arc while the eyes, nose, chin and the other side stay; depth swells and lights the skin.
   * - `strand`: the hair strand of that name nearest the press, its node nearest the press held at the pointer, its tip
   *   at most `strands[].reach` from where it hangs. Strands without a reach are not held (null).
   * - `cloth`: the garment's hem at `at` (0–1 along it; nearest the press when omitted) held towards the pointer, never
   *   beyond `cloth[].reach` and the garment's own hem travel (`cloth[].max`) and never lifted. Garments without a
   *   reach are not held (null).
   * - `chest`: one bound side of the torso, with lateral and vertical motion, depth and inertia.
   * - `fabric`: a point on a material-bound garment, with local strain relative to the sewn edge.
   * - `limb`: the limb chain swings (and bends, held below its middle joint) towards the pointer, within its limits.
   * - `accessory`: a hanging tassel (`rig.accessories`) points from its knot towards the pointer.
   * WebGL runtime only.
   */
  grab(target: GrabTarget, x: number, y: number): number | null;
  /** Whether `grab(target, ...)` would hold something on this avatar. WebGL runtime only. */
  canGrab(target: GrabTarget): boolean;
  /**
   * Move a held part: the pointer's offset from the press in source pixels and, for cheeks, chest or fabric, a depth from -1 to 1
   * (towards the camera > 0; other parts ignore it). Holding still keeps the shape. Unknown handles are ignored.
   */
  moveGrab(handle: number, dx: number, dy: number, depth?: number): void;
  /**
   * Let go: the part returns through its own dynamics, starting with the velocity (source px/s, bounded per part).
   * Unknown or stale handles are ignored; `destroy` lets go of everything.
   */
  releaseGrab(handle: number, vx?: number, vy?: number): void;
  advance(seconds: number, fps?: number): void;
  advanceParameters(seconds: number): void;
  destroy(): void;
}

export async function createMeshAvatar(
  canvas: HTMLCanvasElement,
  options: MeshAvatarOptions,
): Promise<MeshAvatar> {
  return createMeshAvatarImpl(canvas, { ...options, rig: parseRig(options.rig) });
}
