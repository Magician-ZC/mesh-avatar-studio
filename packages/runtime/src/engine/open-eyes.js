// Open-eye layers (optional built data, `tools/build-layers.py --open-eyes`): `eye0_open` and `eye1_open`
// hold the source pixels around each eye, with a hole for the eye ball. The cut-out lid and lash layers only
// approximate the drawing (their lines are unblended against skin and the eye hole is painted over), so while
// an eye is fully open its open layer is drawn over them and the eye looks exactly as drawn; the ball still
// follows the gaze through the hole. The layer deforms like the base and fades out as soon as the lids move.
// An avatar without these layers renders as before; older runtimes skip layer names they do not know.

/** Openness (away from 1) and eye smile over which an open layer fades out. */
export const OPEN_EYE_FADE = 0.2;

export const openEyeLayerName = eye => `eye${eye}_open`;

const clamp01 = v => Math.min(1, Math.max(0, v));

/**
 * Opacity of an eye's open layer.
 * @param {number} open   eye openness (1 = as drawn)
 * @param {number} smile  eye smile (0 = as drawn)
 * @param {boolean} drawn the eye has drawn closed / half-closed sprites: its lids then never move, the sprites
 *                        fade in over the open eye instead
 * @param {boolean} covered a sprite fully covers the eye
 */
export function openEyeAlpha(open, smile, drawn = false, covered = false) {
  if (covered) return 0;
  const lids = drawn ? 1 : clamp01(1 - Math.abs(open - 1) / OPEN_EYE_FADE);
  const t = lids * clamp01(1 - (smile || 0) / OPEN_EYE_FADE);
  // floating-point dust at the ends of the fade must not keep a hidden layer drawing
  return t < 1e-6 ? 0 : t * t * (3 - 2 * t);
}
