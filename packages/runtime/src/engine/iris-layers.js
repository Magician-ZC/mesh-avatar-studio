// Iris layers (optional built data: `eyes[i].iris` in the rig makes `tools/build-layers.py` cut them): `eye{i}_white`
// is the eye ball without its iris and `eye{i}_iris` the iris as a sprite. The eye ball layer holds both in one
// picture and follows the gaze by warping it inside the opening, which suits a tall drawn eye with white all around
// the iris; on an eye whose iris is as tall as the opening the warp shears the iris and reads beyond the cut-out.
// With the two layers the white stays where it is and the iris moves as it is, clipped by the lids like the eye ball.
// An eye without both layers is drawn from its eye ball as before; older runtimes skip layer names they do not know.

export const irisLayerNames = eye => ({ white: `eye${eye}_white`, iris: `eye${eye}_iris` });

/**
 * Source px the white and iris layers are drawn beyond the lids. The line layers of an iris eye stay out of the
 * opening, so the eye reaches 1 px under them (the lash then covers its clipped edge, as it covers the eye ball's).
 * While an open-eye layer lies over the eye it is 3 px: that layer has a hole exactly as large as what the eye
 * layers draw, and with a clip at the lids the base would show through at the hole's edge as soon as the textures
 * are filtered (any size but 1:1).
 */
export const IRIS_RIM = [1, 3];

/** How far beyond the lids an eye's white and iris are drawn, for the opacity of its open-eye layer (0 without one). */
export const irisRim = openAlpha => IRIS_RIM[0] + (IRIS_RIM[1] - IRIS_RIM[0]) * Math.min(1, Math.max(0, openAlpha || 0));

/** An eye is drawn from its iris layers only when the build has both. */
export function hasIrisLayers(layers, eye) {
  if (!layers || typeof layers !== 'object') return false;
  const { white, iris } = irisLayerNames(eye);
  return Object.hasOwn(layers, white) && Object.hasOwn(layers, iris);
}

/** The layered parts of an eye, back to front. */
export const eyePartNames = iris => (iris ? ['white', 'iris', 'low', 'crease', 'lash'] : ['ball', 'low', 'crease', 'lash']);

/** Eye white and iris layers use the eye ball's mesh spacing; the line layers are finer. */
export const isEyeFill = part => part === 'ball' || part === 'white' || part === 'iris';

/**
 * How far each eye's iris travels for a gaze, in source px (x to the right, y down): the travel of the eye ball
 * warp at the middle of the opening. The second eye turns in a little less.
 * @returns {[[number, number], [number, number]]}
 */
export function gazeShift(gazeX, gazeY, travel) {
  return [[gazeX * travel, -gazeY * travel * 0.6], [gazeX * travel * 0.85, -gazeY * travel * 0.6]];
}
