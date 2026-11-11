// Viewport (client) coordinates to source image pixels (ER1). The renderer already maps canvas CSS pixels to source
// pixels (`toImage`, which also follows the canvas's device pixel ratio, padding and contain/cover fit); this adds the
// canvas position on the page.

/** The parts of a canvas element the mapping reads. */
export interface CanvasBox {
  getBoundingClientRect(): { left: number; top: number; width: number; height: number };
  readonly offsetWidth: number;
  readonly offsetHeight: number;
  readonly clientLeft: number;
  readonly clientTop: number;
}

/** The renderer's canvas CSS px → source px mapping; null before the first frame is drawn. */
export interface ImageMapper {
  toImage(x: number, y: number): [number, number] | null;
}

/**
 * Maps a viewport point (for example `PointerEvent.clientX/Y`) to source image pixels. Returns null before the first
 * frame is drawn, for non-finite input and while the canvas has no size. The bounding box includes CSS transforms
 * that scale the canvas, while the renderer works in the canvas's own CSS pixels, so the offset is scaled back; the
 * border is skipped because the drawing starts inside it.
 */
export function clientToImage(canvas: CanvasBox, renderer: ImageMapper, clientX: number, clientY: number): [number, number] | null {
  if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return null;
  const box = canvas.getBoundingClientRect();
  const scaleX = box.width > 0 && canvas.offsetWidth > 0 ? canvas.offsetWidth / box.width : 1;
  const scaleY = box.height > 0 && canvas.offsetHeight > 0 ? canvas.offsetHeight / box.height : 1;
  const point = renderer.toImage((clientX - box.left) * scaleX - canvas.clientLeft, (clientY - box.top) * scaleY - canvas.clientTop);
  return point && Number.isFinite(point[0]) && Number.isFinite(point[1]) ? [point[0], point[1]] : null;
}
