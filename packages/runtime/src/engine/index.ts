import type { LightingSettings } from '../lighting/settings';
import type { Rig, ClothTuning } from '../rig/types';
import { parseRig } from '../rig/validate';
import { createMeshAvatarImpl } from './createMeshAvatar.js';
import { OVERLAY_EXPRESSIONS } from './expression-overlay.js';
import type { LipSyncSettings } from './lip-sync';
import type { ExpressionSettings } from './expression-settings';
import type { ImpulseTarget } from '../interaction/impulse';

export type { ImpulseTarget };
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
   * Each kind of target has an upper bound, and larger values are truncated. Unknown targets are ignored, and `limb`
   * targets are reserved for limb chains. For `cloth`, `at` (0–1 along the hem) pushes the nearest columns; without
   * it the whole hem moves. WebGL runtime only.
   */
  impulse(target: ImpulseTarget, vx: number, vy: number): void;
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
