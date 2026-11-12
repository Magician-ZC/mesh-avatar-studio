// Upper bounds of interaction impulses (ER1). An impulse beyond a bound is truncated, and an impulse never raises a
// velocity above the bound (or above the velocity the part already has from its normal motion), so repeated
// impulses cannot pump a part further than a single one. Physics keeps its own stability limits on top of these.

export const IMPULSE_LIMITS = Object.freeze({
  /**
   * Head parameter springs (`angleX`, `angleY`, `angleZ`), parameter units per second. With the runtime's head springs
   * a single capped impulse peaks at about 6° of turn or nod and a fifth of `head.maxRoll`.
   */
  head: 120,
  /** Body parameter springs (`bodyAngleX`, `bodyAngleZ`, range ±10), units per second; peaks at about 2.5 units. */
  body: 24,
  /** Hair strand tip speed in source px/s (nodes nearer the root get a proportional share). */
  strand: 1600,
  /** Garment hem speed in source px/s. */
  cloth: 900,
  /** Limb end speed in source px/s (ER4); its part across the limb turns the limb about its root. */
  limb: 1200,
});

/**
 * A limb's swing parameter (-1..1, times the limb's swing limit) changes at most this much per second from an impulse,
 * so a capped push swings a limb up to about its limit and its spring brings it back within about 3 s.
 */
export const LIMB_SWING_RATE = 8;

/**
 * A garment column swings at most this many times its rendered limit angle per second (the angle at which its `max`
 * offset is reached), so a capped impulse moves a hem up to about its `max` without slamming it there.
 */
export const CLOTH_SWING_RATE = 9;

/** Columns within this many column spacings of `at` are pushed; the share falls off as a cosine. */
export const CLOTH_FALLOFF = 2;

/**
 * Physics binds the fringe strands into one sheet by averaging neighbours, which would average a single strand's
 * impulse away; fringe strands within this many places of the target share it, falling off as a cosine.
 */
export const FRINGE_FALLOFF = 3;
