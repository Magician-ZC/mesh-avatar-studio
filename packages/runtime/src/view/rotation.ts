// View rotation (ER6, WebGL runtime only): `rig.view.rotation` (radians, default 0) draws the whole avatar rotated by
// that angle about the source image centre. A build pipeline that turned a lying figure upright to bind it sets the
// opposite angle here, so the avatar is shown the way the original picture was. The rig, the deformers and every
// coordinate the runtime reports (`toImagePoint`) stay in the unrotated source pixels; only the final placement on the
// canvas turns. Fit and margins use the bounding box of the rotated image. Without the field nothing here runs.

/** The rig's view rotation in radians; 0 when absent or not finite (the original code path). */
export function viewRotation(rig: { view?: { rotation?: number } }): number {
  const value = rig.view?.rotation;
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** Width and height of the bounding box of a `width` x `height` rectangle rotated by `rotation`. */
export function rotatedSize(width: number, height: number, rotation: number): [number, number] {
  const c = Math.abs(Math.cos(rotation)), s = Math.abs(Math.sin(rotation));
  return [width * c + height * s, width * s + height * c];
}

/** Where the renderer puts the rotated image on a canvas buffer of `canvas` device pixels. */
export interface ViewPlacement {
  /** Source px to device px. */
  scale: number;
  /** Device position of the source origin after rotating about `centre`: device = scale * q + offset. */
  offset: [number, number];
  /** cos and sin of the rotation, and the source image centre it turns about. */
  cos: number;
  sin: number;
  centre: [number, number];
}

/**
 * Fit the rotated image like the unrotated one: `contain` keeps the whole rotated box in view and anchors it to the
 * bottom edge (a cut-off body stays on the floor), `cover` fills the canvas about its centre. `pad.top` is a share of
 * the rotated box's height above it, `pad.side` a share of its width on each side.
 */
export function placeView(canvas: [number, number], image: { w: number; h: number }, pad: { top: number; side: number },
  fit: 'contain' | 'cover', rotation: number): ViewPlacement {
  const [w, h] = canvas;
  const [boxW, boxH] = rotatedSize(image.w, image.h, rotation);
  const s = (fit === 'cover' ? Math.max : Math.min)(w / (boxW * (1 + 2 * pad.side)), h / (boxH * (1 + pad.top)));
  const left = (w - boxW * s) / 2, top = (h - boxH * s) / (fit === 'cover' ? 2 : 1);
  const centre: [number, number] = [image.w / 2, image.h / 2];
  // the box centre lands on the image centre: device = s * q + offset with q the rotated source point
  return {
    scale: s, cos: Math.cos(rotation), sin: Math.sin(rotation), centre,
    offset: [left + boxW * s / 2 - s * centre[0], top + boxH * s / 2 - s * centre[1]],
  };
}

/** A source point turned about the image centre (still in source px). */
export function rotateSource(view: Pick<ViewPlacement, 'cos' | 'sin' | 'centre'>, x: number, y: number): [number, number] {
  const dx = x - view.centre[0], dy = y - view.centre[1];
  return [view.centre[0] + dx * view.cos - dy * view.sin, view.centre[1] + dx * view.sin + dy * view.cos];
}

/** Source px to canvas device px. */
export function sourceToDevice(view: ViewPlacement, x: number, y: number): [number, number] {
  const [qx, qy] = rotateSource(view, x, y);
  return [qx * view.scale + view.offset[0], qy * view.scale + view.offset[1]];
}

/** Canvas device px to source px (the inverse of `sourceToDevice`). */
export function deviceToSource(view: ViewPlacement, x: number, y: number): [number, number] {
  const qx = (x - view.offset[0]) / view.scale - view.centre[0], qy = (y - view.offset[1]) / view.scale - view.centre[1];
  return [view.centre[0] + qx * view.cos + qy * view.sin, view.centre[1] - qx * view.sin + qy * view.cos];
}

/**
 * GLSL for the rotated vertex shaders: `aPos` (source px) is turned about the image centre before the usual
 * `* uScale + uOffset`. `uView` holds cos, sin and the centre.
 */
export function rotateVertexShader(source: string): string {
  const position = 'gl_Position = vec4(aPos * uScale + uOffset, 0.0, 1.0);';
  if (!source.includes(position)) throw new Error('view rotation: unexpected vertex shader');
  return source
    .replace('uniform vec2 uOffset;', 'uniform vec2 uOffset; uniform vec4 uView;')
    .replace(position, 'vec2 d = aPos - uView.zw; vec2 q = vec2(d.x * uView.x - d.y * uView.y, d.x * uView.y + d.y * uView.x) + uView.zw; '
      + 'gl_Position = vec4(q * uScale + uOffset, 0.0, 1.0);');
}
