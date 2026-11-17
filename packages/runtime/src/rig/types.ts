export type Point = [number, number];
export type Band = [number, number];
export interface Ellipse { cx: number; cy: number; rx: number; ry: number }
export interface Eye {
  opening: Point[];
  roi: Point[];
  x0: number;
  x1: number;
  top: number[];
  bot: number[];
  /** Optional: the iris outline, also where the lids cover it. With it the build cuts iris layers for this eye. */
  iris?: Ellipse;
}
export interface Strand {
  name: string;
  nodes: Point[];
  sigma: number;
  k: number;
  max: number;
  /**
   * Optional (M11): how far a held strand's tip may be pulled from where it hangs, px. With it the strand can be held
   * (canGrab / grab, WebGL runtime only); without it a drag flicks the strand as before. Builds write it only when
   * the picture keeps a pull that far clean.
   */
  reach?: number;
}
export interface Accessory {
  name: string;
  pivot: Point;
  tip: Point;
  split: number;
  box: [number, number, number, number];
  color: { redness: number; minRed: number };
}
/** Optional garment that swings with the body (for example a skirt). */
export interface ClothTuning {
  responseGain?: number;
  /** Display-only exaggeration of link angles relative to the uppermost link. */
  tierGain?: number;
  /** Damping ratio multiplier for modes above the lowest mode (default 1). */
  highModeDamping?: number;
  swing: number;
  stiffness: number;
  damping: number;
  follow: number;
  cohesion: number;
  wind: number;
  max: number;
}
export interface Cloth extends Partial<ClothTuning> {
  /** Physical image scale in millimetres per source pixel; inferred if omitted. */
  scale?: number;
  name: string;
  outline: Point[];
  waist: Point[];
  hem: Point[];
  /** Ordered waist-to-hem polylines; columns follow their source arc lengths. */
  seams?: Point[][];
  /** Ordered cross-column seam lines; each defines a pendulum joint. */
  tiers?: Point[][];
  columns?: number;
  rows?: number;
  behind?: boolean;
  parent?: string;
  /**
   * Optional (M11): how far a held hem may be pulled from where it hangs, px; the hem never goes beyond the garment's
   * own travel (`max`). With it the garment can be held (canGrab / grab, WebGL runtime only); without it a drag swings
   * the hem as before.
   */
  reach?: number;
}
export interface Hand {
  outline: Point[];
  jaw: Point[];
  jawRange: Band;
  background: Point[];
  elbow: Point;
  wrist: Point;
  knuckle: Point;
  contact: Point;
  forearmShare: number;
  armBand: Band;
  wristBand: Band;
  handBand: Band;
  fingerXBand: Band;
  fingerYBand: Band;
  pinBand: Band;
}
/** A hanging arm separated from the body by a transparent gap. */
export interface Arm {
  shoulder: Point;
  elbow: Point;
  wrist: Point;
  split: Point[];
}
/** Optional two-bone arms. Left/right refer to the image; amounts are degrees. */
export interface Arms {
  left: Arm;
  right: Arm;
  swing?: number;
  bend?: number;
  inward?: number;
}
/**
 * Optional named hit zone for pointer interaction (WebGL runtime only). The outline is in source
 * pixels; array order sets priority. The name `none` is reserved for a miss. The runtime does not
 * act on zones; it validates and preserves them for applications.
 */
export interface HitZone {
  name: string;
  outline: Point[];
  side?: 'left' | 'right';
  target?: { kind: 'strand' | 'cloth' | 'limb'; name: string };
}
/**
 * Optional generic limb chain (WebGL runtime only): a separately cut arm, leg or prop with two bones, for poses that the
 * hanging `arms` and planted `legs` do not cover. Its root follows the base drawing; the parameters
 * `limb:<name>:swing` and `limb:<name>:bend` (-1..1, default 0) turn the chain about the root and bend it at the middle
 * joint. Angles are degrees.
 */
export interface Limb {
  /** Letters, digits and underscores; unique. The layer is `limb_<name>`. */
  name: string;
  kind: 'arm' | 'leg' | 'prop';
  /** Root, middle and end joint, source pixels. */
  joints: [Point, Point, Point];
  /** Cut-out polygon of the limb, source pixels. */
  outline: Point[];
  /** Draw before the base drawing instead of in front of it. */
  behind?: boolean;
  /** Swing limit about the root in degrees, 0..10, default 4. */
  swing?: number;
  /** Bend limit at the middle joint in degrees, 0..12, default 6. */
  bend?: number;
}
export interface Leg { hip: Point; knee: Point; ankle: Point }
/** Optional planted-foot weight transfer. Left/right refer to the image. */
export interface Legs {
  spine?: number;
  bob?: number;
  /** Maximum pelvis yaw in degrees, 0..15; absent/zero preserves the old projection. */
  twist?: number;
  pelvis: Point;
  shift: number;
  tilt: number;
  bend: number;
  band: Band;
  left: Leg;
  right: Leg;
  split: Point[];
}
export interface Rig {
  version: 1;
  image: { width: number; height: number };
  head: Ellipse & {
    shiftX: number; shiftY: number; pivotX: number; pivotY: number;
    maxRoll: number; weightBand: Band; turnBand: Band;
  };
  body: {
    pivotX: number; pivotY: number; maxRoll: number;
    breathBand: Band; rollBand: Band; chest: Ellipse; shoulders: Ellipse[];
    hip?: { shift: number; band: Band };
  };
  face: Record<'nose' | 'mouth' | 'eyeA' | 'eyeB' | 'earL' | 'earR', Ellipse> & {
    brow: Ellipse & { band: Band };
    jaw: Ellipse & { band: Band };
  };
  buns?: Partial<Record<'bunL' | 'bunR', Ellipse>>;
  eyes: Eye[];
  mouth: {
    cx: number; cy: number; angle: number; halfLen: number; bow: number;
    area: Ellipse & { angle: number };
  };
  cheeks: Point[];
  strands?: Strand[];
  accessories?: Accessory[];
  hand?: Hand;
  cloth?: Cloth[];
  legs?: Legs;
  arms?: Arms;
  limbs?: Limb[];
  hitZones?: HitZone[];
  mesh: {
    baseCell: number;
    fine: { x0: number; x1: number; y0: number; y1: number; cell: number };
    handCell: number; tasselCell: number; eyeBallCell: number;
    eyeCell: number; spriteCell: number; clothCell?: number;
    /** Optional limb mesh spacing, px; defaults to `baseCell`. */
    limbCell?: number;
  };
  view: {
    padTop: number; padSide: number; gazeCenter?: Point; irisTravel?: number;
    /** Optional rotation of the whole drawing about the image centre, radians (default 0). WebGL runtime only. */
    rotation?: number;
  };
  /** Optional physics settings. WebGL runtime only. */
  physics?: {
    /** Direction things hang in, radians in source pixels (default pi/2, straight down the image). */
    gravity?: number;
  };
}
