// View rotation and gravity (ER6): the rotated drawing fits its rotated bounding box, toImagePoint and toClientPoint
// undo exactly what the vertex shader does, lighting stays off, the fields validate as finite numbers, and with the
// image and rig turned by 90 degrees and gravity turned with them, hair, buns and garments move exactly as turned.
import { afterEach, describe, expect, test, vi } from 'vitest';
import fixture from '../../../samples/miko-qipao/rig.json';
import { parseRig, validateRig, type Cloth, type Rig } from 'mesh-avatar';
import { createRig } from '../src/engine/rig.js';
import { createRenderer } from '../src/engine/renderer.js';
import { createPhysics } from '../src/engine/physics.js';
import { createCloth } from '../src/engine/cloth.js';
import { clientToImage, imageToClient, type CanvasBox } from '../src/interaction/coordinates';
import { deviceToSource, placeView, rotateVertexShader, rotatedSize, sourceToDevice, viewRotation } from '../src/view/rotation';
import { DEFAULT_GRAVITY, gravityFrame, splitByGravity } from '../src/view/gravity';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

const rigWith = (view: Partial<Rig['view']> = {}, extra: Partial<Rig> = {}): Rig => {
  const rig = parseRig(fixture);
  return { ...rig, ...extra, view: { ...rig.view, ...view } };
};

describe('view rotation math', () => {
  test('absent or non-finite rotations are 0 and keep the original code path', () => {
    expect(viewRotation({})).toBe(0);
    expect(viewRotation({ view: {} })).toBe(0);
    expect(viewRotation({ view: { rotation: Number.NaN } })).toBe(0);
    expect(viewRotation({ view: { rotation: -Math.PI / 2 } })).toBe(-Math.PI / 2);
  });

  test('the rotated bounding box, and source to device and back', () => {
    const [w, h] = rotatedSize(300, 100, Math.PI / 2);
    expect(w).toBeCloseTo(100, 9); expect(h).toBeCloseTo(300, 9);
    const [dw, dh] = rotatedSize(300, 100, Math.PI / 6);
    expect(dw).toBeCloseTo(300 * Math.cos(Math.PI / 6) + 100 * 0.5, 9);
    expect(dh).toBeCloseTo(300 * 0.5 + 100 * Math.cos(Math.PI / 6), 9);
    for (const rotation of [0.3, -Math.PI / 2, Math.PI, 2.2]) {
      const view = placeView([640, 480], { w: 300, h: 500 }, { top: 0.1, side: 0.05 }, 'contain', rotation);
      for (const p of [[0, 0], [300, 500], [12.5, 431.25], [-40, 600]] as [number, number][]) {
        const back = deviceToSource(view, ...sourceToDevice(view, ...p));
        expect(back[0]).toBeCloseTo(p[0], 9); expect(back[1]).toBeCloseTo(p[1], 9);
      }
    }
  });

  test('contain keeps the whole rotated image in view and stands it on the bottom edge', () => {
    // a 300 x 500 image turned by -90 degrees is 500 x 300: on a 1000 x 1000 canvas it is scaled by 2 and fills the width
    const view = placeView([1000, 1000], { w: 300, h: 500 }, { top: 0, side: 0 }, 'contain', -Math.PI / 2);
    expect(view.scale).toBeCloseTo(2, 9);
    const corners = [[0, 0], [300, 0], [0, 500], [300, 500]].map(([x, y]) => sourceToDevice(view, x, y));
    expect(Math.min(...corners.map(c => c[0]))).toBeCloseTo(0, 9);
    expect(Math.max(...corners.map(c => c[0]))).toBeCloseTo(1000, 9);
    expect(Math.max(...corners.map(c => c[1]))).toBeCloseTo(1000, 9); // the floor
    expect(Math.min(...corners.map(c => c[1]))).toBeCloseTo(400, 9);
    // -90 degrees turns the image's top edge to the left: its top-left corner goes to the bottom-left
    expect(corners[0][0]).toBeCloseTo(0, 9); expect(corners[0][1]).toBeCloseTo(1000, 9);
  });

  test('the rotated vertex shader turns aPos about the centre before scaling', () => {
    const shader = rotateVertexShader('uniform vec2 uScale; uniform vec2 uOffset;\nvoid main() { gl_Position = vec4(aPos * uScale + uOffset, 0.0, 1.0); }');
    expect(shader).toContain('uniform vec4 uView;');
    expect(shader).toContain('gl_Position = vec4(q * uScale + uOffset, 0.0, 1.0);');
    expect(() => rotateVertexShader('void main() {}')).toThrow();
  });
});

describe('the renderer with view.rotation', () => {
  interface Layout { css: [number, number]; dpr: number; border: number; zoom: number }
  const LEFT = 21.5, TOP = 8.75;
  const layouts: Layout[] = [
    { css: [720, 960], dpr: 1, border: 0, zoom: 1 },
    { css: [333, 251], dpr: 1.5, border: 0, zoom: 1 },
    { css: [1254, 600], dpr: 2, border: 3, zoom: 1 },
    { css: [480, 640], dpr: 3, border: 2, zoom: 0.5 },
  ];

  function place(rig: Rig, fit: 'contain' | 'cover', pad: [number, number], layout: Layout) {
    vi.stubGlobal('window', { devicePixelRatio: layout.dpr });
    const { Renderer } = createRenderer(createRig(rig), rig);
    const canvas = {
      clientWidth: layout.css[0], clientHeight: layout.css[1], width: 300, height: 150,
      clientLeft: layout.border, clientTop: layout.border,
      offsetWidth: layout.css[0] + 2 * layout.border, offsetHeight: layout.css[1] + 2 * layout.border,
      getBoundingClientRect() { return { left: LEFT, top: TOP, width: this.offsetWidth * layout.zoom, height: this.offsetHeight * layout.zoom }; },
    };
    const renderer = Object.create(Renderer.prototype);
    Object.assign(renderer, { canvas, pad: { top: pad[0], side: pad[1] }, fit });
    renderer.resize();
    const rotation = viewRotation(rig), c = [rig.image.width / 2, rig.image.height / 2];
    /** What the rotated vertex shader does with a source pixel, through the viewport and the CSS box. */
    const toClient = ([x, y]: [number, number]): [number, number] => {
      const dx = x - c[0], dy = y - c[1];
      const qx = c[0] + dx * Math.cos(rotation) - dy * Math.sin(rotation), qy = c[1] + dx * Math.sin(rotation) + dy * Math.cos(rotation);
      const ndcX = qx * renderer.scale[0] + renderer.offset[0], ndcY = qy * renderer.scale[1] + renderer.offset[1];
      const bufferX = (ndcX + 1) / 2 * canvas.width, bufferY = (1 - ndcY) / 2 * canvas.height;
      const cssX = bufferX * canvas.clientWidth / canvas.width, cssY = bufferY * canvas.clientHeight / canvas.height;
      return [LEFT + (layout.border + cssX) * layout.zoom, TOP + (layout.border + cssY) * layout.zoom];
    };
    return { canvas: canvas as CanvasBox & typeof canvas, renderer, toClient };
  }

  test('an explicit rotation of 0 fits exactly like no rotation', () => {
    for (const fit of ['contain', 'cover'] as const) {
      const plain = place(rigWith(), fit, [0.1, 0.05], layouts[1]).renderer;
      const zero = place(rigWith({ rotation: 0 }), fit, [0.1, 0.05], layouts[1]).renderer;
      expect([zero.scale, zero.offset, zero.pxScale, zero.pxOff]).toEqual([plain.scale, plain.offset, plain.pxScale, plain.pxOff]);
      expect(zero.view).toBeUndefined();
    }
  });

  const rotations = [-Math.PI / 2, Math.PI / 2, Math.PI, 0.35, -2.2];
  const fixtureRig = parseRig(fixture);
  const points: [number, number][] = [];
  for (let y = 0; y <= fixtureRig.image.height; y += fixtureRig.image.height / 4) for (let x = 0; x <= fixtureRig.image.width; x += fixtureRig.image.width / 4) points.push([x, y]);
  points.push([fixtureRig.head.cx, fixtureRig.head.cy], [-50, 1300]);

  test.each(rotations.flatMap(rotation => (['contain', 'cover'] as const).map(fit => [rotation, fit] as const)))(
    'toImagePoint and toClientPoint undo the drawing (rotation %f, %s)', (rotation, fit) => {
      for (const pad of [[0, 0], [0.1, 0.05], [-0.07, 0.1]] as [number, number][]) for (const layout of layouts) {
        const { canvas, renderer, toClient } = place(rigWith({ rotation }), fit, pad, layout);
        for (const p of points) {
          const client = toClient(p);
          const back = clientToImage(canvas, renderer, ...client)!;
          expect(back[0]).toBeCloseTo(p[0], 6); expect(back[1]).toBeCloseTo(p[1], 6);
          const forward = imageToClient(canvas, renderer, ...p)!;
          expect(forward[0]).toBeCloseTo(client[0], 6); expect(forward[1]).toBeCloseTo(client[1], 6);
        }
      }
    });

  test('turned by -90 degrees on a canvas of swapped size, every source pixel lands where the upright drawing turned shows it', () => {
    const rig = rigWith({ padTop: 0, padSide: 0 });
    const n = rig.image.width;
    expect(rig.image.height).toBe(n); // the sample is square: both canvases are n x n at scale 1
    const layout: Layout = { css: [n, n], dpr: 1, border: 0, zoom: 1 };
    const upright = place(rig, 'contain', [0, 0], layout), turned = place({ ...rig, view: { ...rig.view, rotation: -Math.PI / 2 } }, 'contain', [0, 0], layout);
    for (const p of points) {
      const a = upright.toClient(p), b = turned.toClient(p);
      // turning the upright picture by -90 degrees moves (x, y) to (y, n - x)
      expect(b[0] - LEFT).toBeCloseTo(a[1] - TOP, 6);
      expect(b[1] - TOP).toBeCloseTo(n - (a[0] - LEFT), 6);
    }
  });

  test('lighting stays off on a rotated view', () => {
    const { renderer } = place(rigWith({ rotation: Math.PI / 2 }), 'contain', [0, 0], layouts[0]);
    renderer.lighting = { enabled: false };
    renderer.setLighting({ enabled: true });
    expect(renderer.lighting.enabled).toBe(false);
  });

  test('toClientPoint is null before the first frame and for non-finite input', () => {
    const { canvas, renderer } = place(rigWith({ rotation: 1 }), 'contain', [0, 0], layouts[0]);
    const { Renderer } = createRenderer(createRig(rigWith()), rigWith());
    const unsized = Object.assign(Object.create(Renderer.prototype), { canvas, pad: { top: 0, side: 0 }, fit: 'contain' });
    expect(imageToClient(canvas, unsized, 10, 10)).toBeNull();
    expect(imageToClient(canvas, renderer, Number.NaN, 10)).toBeNull();
  });
});

describe('rig fields', () => {
  test('view.rotation and physics.gravity are optional finite numbers', () => {
    expect(validateRig(rigWith({ rotation: -Math.PI / 2 }, { physics: { gravity: Math.PI } }))).toEqual([]);
    expect(validateRig(rigWith({}, { physics: {} }))).toEqual([]);
    expect(validateRig(rigWith({ rotation: Number.POSITIVE_INFINITY })).join()).toContain('rig.view.rotation: expected a finite number');
    expect(validateRig(rigWith({}, { physics: { gravity: Number.NaN } })).join()).toContain('rig.physics.gravity: expected a finite number');
    expect(validateRig({ ...rigWith(), physics: 3 }).join()).toContain('rig.physics: expected an object');
    const kept = parseRig(rigWith({ rotation: 0.5 }, { physics: { gravity: 1 } }));
    expect([kept.view.rotation, kept.physics?.gravity]).toEqual([0.5, 1]);
  });

  test('gravity defaults to straight down and only other directions get a frame', () => {
    expect(gravityFrame({})).toBeNull();
    expect(gravityFrame({ physics: { gravity: DEFAULT_GRAVITY } })).toBeNull();
    const frame = gravityFrame({ physics: { gravity: Math.PI } })!;
    expect(frame.down[0]).toBeCloseTo(-1, 12); expect(frame.down[1]).toBeCloseTo(0, 12);
    expect(frame.across[0]).toBeCloseTo(0, 12); expect(frame.across[1]).toBeCloseTo(1, 12);
    const [across, along] = splitByGravity(frame, 3, 4);
    expect(across).toBeCloseTo(4, 12); expect(along).toBeCloseTo(-3, 12);
  });
});

// ---- gravity: the same scene turned by 90 degrees --------------------------------------------------------------

const TURN = Math.PI / 2, CENTRE: [number, number] = [400, 300];
const turnPoint = ([x, y]: number[]): [number, number] => {
  const dx = x - CENTRE[0], dy = y - CENTRE[1];
  return [CENTRE[0] + dx * Math.cos(TURN) - dy * Math.sin(TURN), CENTRE[1] + dx * Math.sin(TURN) + dy * Math.cos(TURN)];
};
const turnVector = ([x, y]: number[]): [number, number] => [x * Math.cos(TURN) - y * Math.sin(TURN), x * Math.sin(TURN) + y * Math.cos(TURN)];

/** Waist motion of a few seconds: sideways and up-down shifts, as a parameter sequence. */
const drive = (frame: number) => ({ angleZ: 12 * Math.sin(frame / 23), bodyAngleZ: 4 * Math.sin(frame / 41), shift: 30 * Math.sin(frame / 17), lift: 9 * Math.cos(frame / 13) });

function hairScene(turned: boolean) {
  const base = parseRig(fixture);
  const strands = base.strands!.map(s => ({ ...s, nodes: turned ? s.nodes.map(turnPoint) : s.nodes }));
  // the buns' breeze takes its phase from where the bun is, so they sit on the centre of the turn (it stays put)
  const buns = Object.fromEntries(Object.entries(base.buns ?? {}).map(([k, e]) => [k, { ...e!, cx: CENTRE[0], cy: CENTRE[1] }]));
  const rig = { ...base, strands, buns, ...(turned ? { physics: { gravity: Math.PI / 2 + TURN } } : {}) } as Rig;
  const engine = {
    TASSELS: [], tasselAnchor: () => [0, 0], stepBody: () => undefined,
    applyHead(p: number[], w: number, P: Record<string, number>) {
      const [dx, dy] = turned ? turnVector([P.shift, P.lift]) : [P.shift, P.lift];
      p[0] += dx * w; p[1] += dy * w;
    },
    applyBody() { /* the waist motion is all in applyHead */ },
  } as unknown as Parameters<typeof createPhysics>[0];
  const { Physics } = createPhysics(engine, rig);
  return new Physics();
}

function clothScene(turned: boolean, garment: Cloth) {
  const rig = parseRig(fixture);
  const c = turned ? { ...garment, outline: garment.outline.map(turnPoint), waist: garment.waist.map(turnPoint), hem: garment.hem.map(turnPoint),
    ...(garment.seams ? { seams: garment.seams.map(s => s.map(turnPoint)) } : {}), ...(garment.tiers ? { tiers: garment.tiers.map(t => t.map(turnPoint)) } : {}) } : garment;
  const scene = { ...rig, cloth: [c], ...(turned ? { physics: { gravity: Math.PI / 2 + TURN } } : {}) } as Rig;
  const engine = {
    baseWeights: () => ({}),
    deformBase(x: number, y: number, _w: unknown, P: Record<string, number>, _phys: unknown, out: number[]) {
      const [dx, dy] = turned ? turnVector([P.shift, P.lift]) : [P.shift, P.lift];
      out[0] = x + dx; out[1] = y + dy; return out;
    },
  } as unknown as Parameters<typeof createCloth>[0];
  const { Cloth: Simulator, garments } = createCloth(engine, scene);
  return { cloth: new Simulator(), garments };
}

describe('gravity follows the turned scene', () => {
  test('hair strands and buns: offsets of the turned scene are the offsets of the upright one, turned', () => {
    const upright = hairScene(false), turned = hairScene(true);
    let largest = 0, moved = 0;
    for (let frame = 0; frame < 240; frame++) {
      type Offsets = { strands: number[][][]; bunL: number[]; bunR: number[] };
      const a = upright.step(drive(frame), 1 / 60) as unknown as Offsets, b = turned.step(drive(frame), 1 / 60) as unknown as Offsets;
      a.strands.forEach((nodes, i) => nodes.forEach((o, j) => {
        const expected = turnVector(o), got = b.strands[i][j];
        largest = Math.max(largest, Math.hypot(got[0] - expected[0], got[1] - expected[1]));
        moved = Math.max(moved, Math.hypot(...o));
      }));
      for (const key of ['bunL', 'bunR'] as const) {
        const expected = turnVector(a[key]);
        largest = Math.max(largest, Math.hypot(b[key][0] - expected[0], b[key][1] - expected[1]));
      }
    }
    expect(moved).toBeGreaterThan(1); // the scene really moves
    expect(largest).toBeLessThan(1e-9);
  });

  const skirt = (): Cloth => ({ name: 'skirt', outline: [[300, 200], [500, 200], [520, 420], [280, 420]], waist: [[300, 200], [500, 200]], hem: [[280, 420], [520, 420]], columns: 6, rows: 5 });
  const tiered = (): Cloth => ({ ...skirt(), seams: [[[300, 200], [285, 320], [280, 420]], [[400, 200], [400, 310], [400, 420]], [[500, 200], [515, 320], [520, 420]]], tiers: [[[270, 300], [530, 300]]] });

  test.each([['two-link columns', skirt], ['seams and tiers', tiered]] as const)('garments (%s): the turned grid is the upright grid, turned', (_name, garment) => {
    const upright = clothScene(false, garment()), turned = clothScene(true, garment());
    let largest = 0, moved = 0;
    for (let frame = 0; frame < 240; frame++) {
      upright.cloth.step(drive(frame), 1 / 60, {}); turned.cloth.step(drive(frame), 1 / 60, {});
      const a = upright.garments[0].out, b = turned.garments[0].out;
      for (let i = 0; i < a.length; i += 2) {
        const expected = turnPoint([a[i], a[i + 1]]);
        largest = Math.max(largest, Math.hypot(b[i] - expected[0], b[i + 1] - expected[1]));
        moved = Math.max(moved, Math.hypot(a[i] - upright.garments[0].rest[i] - drive(frame).shift, a[i + 1] - upright.garments[0].rest[i + 1] - drive(frame).lift));
      }
    }
    expect(moved).toBeGreaterThan(1); // the hem swings relative to the waist
    expect(largest).toBeLessThan(1e-6);
  });

  test('gravity given as exactly pi/2 is the original code path', () => {
    const plain = clothScene(false, skirt());
    const rig = { ...parseRig(fixture), cloth: [skirt()], physics: { gravity: DEFAULT_GRAVITY } } as Rig;
    const engine = { baseWeights: () => ({}), deformBase: (x: number, y: number, _w: unknown, P: Record<string, number>, _p: unknown, out: number[]) => { out[0] = x + P.shift; out[1] = y + P.lift; return out; } } as unknown as Parameters<typeof createCloth>[0];
    const { Cloth: Simulator, garments } = createCloth(engine, rig);
    const explicit = new Simulator();
    for (let frame = 0; frame < 120; frame++) { plain.cloth.step(drive(frame), 1 / 60, {}); explicit.step(drive(frame), 1 / 60, {}); }
    expect(Array.from(garments[0].out)).toEqual(Array.from(plain.garments[0].out));
  });
});
