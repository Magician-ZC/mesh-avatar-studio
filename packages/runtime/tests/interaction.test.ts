// Pointer interaction (ER1): toImagePoint agrees with where the renderer draws each source pixel, impulses move their
// part and settle within 3 s, every kind of impulse stays within its bound, and unknown targets change nothing.
import { afterEach, describe, expect, test, vi } from 'vitest';
import fixture from '../../../samples/miko-qipao/rig.json';
import { parseRig, type Cloth, type Rig } from 'mesh-avatar';
import { createRig } from '../src/engine/rig.js';
import { createRenderer } from '../src/engine/renderer.js';
import { createPhysics } from '../src/engine/physics.js';
import { createCloth } from '../src/engine/cloth.js';
import { Motion } from '../src/engine/motion.js';
import { clientToImage, type CanvasBox, type ImageMapper } from '../src/interaction/coordinates';
import { addCapped, addCapped2, createImpulse, type ImpulseTarget } from '../src/interaction/impulse';
import { CLOTH_FALLOFF, CLOTH_SWING_RATE, FRINGE_FALLOFF, IMPULSE_LIMITS } from '../src/interaction/limits';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

const FPS = 60;
const SETTLE_FRAMES = 3 * FPS;

describe('toImagePoint', () => {
  const rig = parseRig(fixture);
  const { Renderer } = createRenderer(createRig(rig), rig);
  interface Layout { css: [number, number]; dpr: number; border: number; zoom: number }
  const LEFT = 37.5, TOP = 12.25;

  /** A canvas laid out as `layout` and a renderer that has sized it, as on the first draw. */
  function place(fit: 'contain' | 'cover', pad: [number, number], layout: Layout) {
    vi.stubGlobal('window', { devicePixelRatio: layout.dpr });
    const canvas = {
      clientWidth: layout.css[0], clientHeight: layout.css[1], width: 300, height: 150,
      clientLeft: layout.border, clientTop: layout.border,
      offsetWidth: layout.css[0] + 2 * layout.border, offsetHeight: layout.css[1] + 2 * layout.border,
      // `transform: scale(zoom)` with its origin at the top-left corner
      getBoundingClientRect() { return { left: LEFT, top: TOP, width: this.offsetWidth * layout.zoom, height: this.offsetHeight * layout.zoom }; },
    };
    const renderer = Object.create(Renderer.prototype) as InstanceType<typeof Renderer> & ImageMapper & { scale: number[]; offset: number[] };
    Object.assign(renderer, { canvas, pad: { top: pad[0], side: pad[1] }, fit });
    renderer.resize();
    /** Where the vertex shader puts a source pixel (aPos * uScale + uOffset), through the viewport and the CSS box. */
    const toClient = ([x, y]: [number, number]): [number, number] => {
      const ndcX = x * renderer.scale[0] + renderer.offset[0], ndcY = y * renderer.scale[1] + renderer.offset[1];
      const bufferX = (ndcX + 1) / 2 * canvas.width, bufferY = (1 - ndcY) / 2 * canvas.height;
      const cssX = bufferX * canvas.clientWidth / canvas.width, cssY = bufferY * canvas.clientHeight / canvas.height;
      return [LEFT + (layout.border + cssX) * layout.zoom, TOP + (layout.border + cssY) * layout.zoom];
    };
    return { canvas: canvas as CanvasBox & typeof canvas, renderer, toClient };
  }

  const points: [number, number][] = [];
  for (let y = 0; y <= rig.image.height; y += rig.image.height / 4) for (let x = 0; x <= rig.image.width; x += rig.image.width / 4) points.push([x, y]);
  points.push([rig.head.cx, rig.head.cy], [-50, 1300]);
  const layouts: Layout[] = [
    { css: [720, 960], dpr: 1, border: 0, zoom: 1 },
    { css: [333, 251], dpr: 1.5, border: 0, zoom: 1 }, // the device-pixel buffer size is rounded
    { css: [1254, 600], dpr: 2, border: 3, zoom: 1 },
    { css: [480, 640], dpr: 3, border: 2, zoom: 0.5 }, // devicePixelRatio is capped at 2; scaled by a CSS transform
  ];

  test.each([
    ['contain', [0, 0]], ['contain', [0.1, 0.05]], ['contain', [rig.view.padTop, rig.view.padSide]],
    ['cover', [0, 0]], ['cover', [0.1, 0.05]], ['cover', [rig.view.padTop, rig.view.padSide]],
  ] as const)('agrees with the drawing for %s fit and padding %j', (fit, pad) => {
    for (const layout of layouts) {
      const { canvas, renderer, toClient } = place(fit, [...pad], layout);
      for (const p of points) {
        const [x, y] = toClient(p);
        const back = clientToImage(canvas, renderer, x, y);
        expect(back, JSON.stringify({ layout, p })).not.toBeNull();
        expect(back![0]).toBeCloseTo(p[0], 6);
        expect(back![1]).toBeCloseTo(p[1], 6);
      }
    }
  });

  test('follows a CSS resize that has not been drawn yet, like the stretched canvas on screen', () => {
    const { canvas, renderer, toClient } = place('contain', [0, 0], layouts[0]);
    canvas.clientWidth = 360; canvas.offsetWidth = 360; // CSS halves the width; the buffer keeps its pixels
    for (const p of points) {
      const [x, y] = toClient(p);
      const back = clientToImage(canvas, renderer, x, y)!;
      expect(back[0]).toBeCloseTo(p[0], 6);
      expect(back[1]).toBeCloseTo(p[1], 6);
    }
  });

  test('is null before the first frame, for non-finite input and for a canvas without size', () => {
    const { canvas, renderer } = place('contain', [0, 0], layouts[0]);
    const unsized = Object.assign(Object.create(Renderer.prototype), { canvas, pad: { top: 0, side: 0 }, fit: 'contain' });
    expect(clientToImage(canvas, unsized, 100, 100)).toBeNull();
    expect(clientToImage(canvas, renderer, Number.NaN, 100)).toBeNull();
    expect(clientToImage(canvas, renderer, 100, Infinity)).toBeNull();
    canvas.clientWidth = 0;
    expect(clientToImage(canvas, renderer, 100, 100)).toBeNull();
  });
});

/** The same pseudo-random sequence in every run, so that a run with an impulse can be compared with one without. */
function seeded(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

type Impulse = ReturnType<typeof createImpulse>;

function parts(rig: Rig) {
  const engine = createRig(rig);
  const { Physics } = createPhysics(engine, rig);
  const physics = new Physics();
  const motion = new Motion(rig.view.gazeCenter ?? [rig.head.cx, rig.head.cy]);
  motion.autoIdle = false; motion.autoMotion = false;
  const { Cloth: Simulator } = createCloth(engine, rig);
  const cloth = new Simulator();
  return { engine, physics, motion, cloth, impulse: createImpulse(rig, { motion, physics, cloth }) };
}

/** Idle motion, hair physics and the base deformation of a 25 px grid for `frames`, optionally with an impulse. */
function simulate(rig: Rig, frames: number, kick?: { frame: number; push: (impulse: Impulse) => void }) {
  const { engine, physics, motion, impulse } = parts(rig);
  const points: [number, number][] = [];
  for (let y = 0; y <= rig.image.height; y += 25) for (let x = 0; x <= rig.image.width; x += 25) points.push([x, y]);
  const weights = points.map(([x, y]) => engine.baseWeights(x, y, 1));
  const positions: Float64Array[] = [], parameters: Record<string, number>[] = [];
  vi.spyOn(Math, 'random').mockImplementation(seeded(11));
  for (let f = 0; f < frames; f++) {
    if (kick?.frame === f) kick.push(impulse);
    const P = motion.update(1 / FPS), phys = physics.step(P, 1 / FPS);
    const out = new Float64Array(points.length * 2), p = [0, 0];
    points.forEach(([x, y], i) => { engine.deformBase(x, y, weights[i], P, phys, p); out[i * 2] = p[0]; out[i * 2 + 1] = p[1]; });
    positions.push(out); parameters.push({ ...P });
  }
  vi.restoreAllMocks();
  return { positions, parameters };
}

const largest = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let max = 0;
  for (let i = 0; i < a.length; i += 2) max = Math.max(max, Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1]));
  return max;
};

describe('impulse on the body, head and hair', () => {
  const rig = parseRig(fixture);
  const KICK = 30, FRAMES = KICK + SETTLE_FRAMES + 1;
  const reference = simulate(rig, FRAMES);

  test.each([
    ['body', { kind: 'body' }, 160, 0, ['bodyAngleX', 'bodyAngleZ']],
    ['head', { kind: 'head' }, 150, 100, ['angleX', 'angleY', 'angleZ', 'browY']],
    ['a side strand', { kind: 'strand', name: 'sideL' }, 600, -200, []],
    ['a fringe strand', { kind: 'strand', name: 'bang3' }, 900, 0, []],
  ] as [string, ImpulseTarget, number, number, string[]][])('%s: moves, then returns to rest within 3 s (< 0.1 px)', (_name, target, vx, vy, moved) => {
    const run = simulate(rig, FRAMES, { frame: KICK, push: impulse => impulse(target, vx, vy) });
    for (let f = 0; f < KICK; f++) expect(run.positions[f]).toEqual(reference.positions[f]);
    const peak = Math.max(...run.positions.slice(KICK, KICK + FPS).map((p, i) => largest(p, reference.positions[KICK + i])));
    expect(peak).toBeGreaterThan(2);
    expect(largest(run.positions[KICK + SETTLE_FRAMES], reference.positions[KICK + SETTLE_FRAMES])).toBeLessThan(0.1);
    // only the target's own parameters change
    const changed = Object.keys(reference.parameters[0]).filter(key =>
      run.parameters.some((p, f) => Math.abs(p[key] - reference.parameters[f][key]) > 1e-9));
    expect(changed.sort()).toEqual([...moved].sort());
  });

  test('a fringe strand shares its impulse with its neighbours in the sheet', () => {
    const { physics, impulse } = parts(rig);
    const fringe = physics.fringe as number;
    expect(fringe).toBe(6);
    const index = rig.strands!.findIndex(s => s.name === 'bang3');
    impulse({ kind: 'strand', name: 'bang3' }, 300, 0);
    const tips = physics.strands.map((s: { v: number[][] }) => s.v[3][0]);
    for (let k = 0; k < rig.strands!.length; k++) {
      const share = k < fringe && Math.abs(k - index) < FRINGE_FALLOFF ? Math.cos(Math.PI / 2 * Math.abs(k - index) / FRINGE_FALLOFF) : 0;
      expect(tips[k]).toBeCloseTo(300 * share, 9);
    }
    // the side locks are not bound into the sheet: only the named one moves
    impulse({ kind: 'strand', name: 'sideR' }, 0, 300);
    expect(physics.strands.map((s: { v: number[][] }) => s.v[3][1])).toEqual(rig.strands!.map(s => s.name === 'sideR' ? 300 : 0));
  });

  test('the strand impulse grows from the pinned root to the tip', () => {
    const { physics, impulse } = parts(rig);
    impulse({ kind: 'strand', name: 'sideL' }, 90, -30);
    const chain = physics.strands[rig.strands!.findIndex(s => s.name === 'sideL')];
    expect(chain.v.map((v: number[]) => v.map(n => +n.toFixed(9)))).toEqual([[0, 0], [30, -10], [60, -20], [90, -30]]);
  });
});

describe('impulse bounds', () => {
  const rig = parseRig(fixture);

  test('a single huge impulse is truncated to the bound of its kind, and repeating it never exceeds the bound', () => {
    const { motion, physics, impulse } = parts(rig);
    for (let i = 0; i < 5; i++) {
      impulse({ kind: 'body' }, 1e9, 0);
      impulse({ kind: 'head' }, -1e9, 1e9);
      impulse({ kind: 'strand', name: 'sideL' }, 1e9, 1e9);
      for (const id of ['bodyAngleX', 'bodyAngleZ']) expect(Math.abs(motion.vel[id])).toBe(IMPULSE_LIMITS.body);
      for (const id of ['angleX', 'angleY', 'angleZ']) expect(Math.abs(motion.vel[id])).toBe(IMPULSE_LIMITS.head);
      const chain = physics.strands[rig.strands!.findIndex(s => s.name === 'sideL')];
      chain.v.forEach((v: number[], j: number) => expect(Math.hypot(v[0], v[1])).toBeCloseTo(IMPULSE_LIMITS.strand * j / 3, 6));
    }
    // a push the other way changes the velocity by at most the bound
    impulse({ kind: 'body' }, -1e9, 0);
    expect(motion.vel.bodyAngleX).toBe(0);
  });

  test('a velocity the part already has from its own motion is not raised further', () => {
    const { motion, impulse } = parts(rig);
    motion.vel.angleX = 500;
    impulse({ kind: 'head' }, 1e9, 0);
    expect(motion.vel.angleX).toBe(500);
    impulse({ kind: 'head' }, -1e9, 0);
    expect(motion.vel.angleX).toBe(500 - IMPULSE_LIMITS.head);
  });

  test('the largest head and body impulses peak at a few degrees and units', () => {
    const KICK = 30;
    const reference = simulate(rig, KICK + FPS);
    const run = simulate(rig, KICK + FPS, { frame: KICK, push: impulse => { impulse({ kind: 'head' }, 1e9, 1e9); impulse({ kind: 'body' }, 1e9, 0); } });
    const excursion = (id: string) => Math.max(...run.parameters.map((p, f) => Math.abs(p[id] - reference.parameters[f][id])));
    for (const id of ['angleX', 'angleY']) expect(excursion(id)).toBeLessThan(7);
    expect(excursion('angleZ')).toBeLessThan(7.5);
    for (const id of ['bodyAngleX', 'bodyAngleZ']) expect(excursion(id)).toBeLessThan(3);
  });

  test('capped additions', () => {
    expect(addCapped(0, 50, 10)).toBe(10);
    expect(addCapped(-4, 5, 10)).toBe(1);
    expect(addCapped(30, 5, 10)).toBe(30);
    expect(addCapped(30, -5, 10)).toBe(25);
    const v = [3, 4];
    addCapped2(v, 30, 40, 10);
    expect(v).toEqual([6, 8]);
    const fast = [30, 40];
    addCapped2(fast, 3, 4, 10);
    expect(Math.hypot(fast[0], fast[1])).toBeCloseTo(50, 9);
  });
});

describe('impulse on cloth', () => {
  const garment = (over: Partial<Cloth> = {}): Cloth => ({
    name: 'skirt', outline: [[0, 0], [100, 0], [100, 100], [0, 100]], waist: [[0, 0], [100, 0]], hem: [[0, 100], [100, 100]],
    columns: 8, rows: 5, ...over,
  });
  function simulation(garments: Cloth[]) {
    const rig = parseRig(fixture); rig.cloth = garments;
    const engine = {
      baseWeights: () => ({}),
      deformBase(x: number, y: number, _w: unknown, P: { shift?: number }, _phys: unknown, out: number[]) { out[0] = x + (P.shift ?? 0); out[1] = y; return out; },
    } as unknown as Parameters<typeof createCloth>[0];
    const { Cloth: Simulator, garments: states } = createCloth(engine, rig);
    const cloth = new Simulator();
    return { cloth, states, step: (frames = 1) => { for (let i = 0; i < frames; i++) cloth.step({ shift: 0 }, 1 / FPS, {}); } };
  }

  test('the columns nearest `at` turn, falling off as a cosine; without `at` the whole hem turns', () => {
    const spacing = 1 / 7;
    const share = (k: number, at: number) => {
      const columns = Math.abs(k * spacing - at) / spacing;
      return columns < CLOTH_FALLOFF ? Math.cos(Math.PI / 2 * columns / CLOTH_FALLOFF) : 0;
    };
    for (const at of [0, 0.5, 1, 0.3]) {
      const s = simulation([garment()]);
      expect(s.cloth.impulse('skirt', at, 50, 0)).toBe(true);
      const g = s.states[0];
      for (let k = 0; k < g.cols; k++) {
        // a vertical 100 px column: the hem moves at 50 px/s when the column turns at 0.5 rad/s
        expect(g.angularVelocity[k]).toBeCloseTo(0.5 * share(k, at), 9);
        expect(g.lowerAngularVelocity[k]).toBeCloseTo(0.5 * share(k, at), 9);
      }
    }
    const whole = simulation([garment()]);
    whole.cloth.impulse('skirt', undefined, 0, 50); // straight down a vertical column: no turn
    expect(Array.from(whole.states[0].angularVelocity)).toEqual(new Array(8).fill(0));
    whole.cloth.impulse('skirt', undefined, -50, 0);
    expect(Array.from(whole.states[0].angularVelocity).every(v => Math.abs(v + 0.5) < 1e-12)).toBe(true);
  });

  test('a pushed hem swings the way it was pushed and settles within 3 s (< 0.1 px)', () => {
    const KICK = 30;
    const reference = simulation([garment()]), pushed = simulation([garment()]);
    reference.step(KICK); pushed.step(KICK);
    expect(pushed.cloth.impulse('skirt', 0.5, 1e9, 0)).toBe(true);
    const hem = (s: typeof pushed) => s.states[0].out[((s.states[0].rows - 1) * s.states[0].cols + 4) * 2];
    let peak = 0;
    for (let f = 0; f < SETTLE_FRAMES; f++) {
      reference.step(); pushed.step();
      if (f < FPS) peak = Math.max(peak, hem(pushed) - hem(reference));
    }
    expect(peak).toBeGreaterThan(2);
    expect(largest(pushed.states[0].out, reference.states[0].out)).toBeLessThan(0.1);
  });

  test('angular velocity stays within the bound of each column', () => {
    const s = simulation([garment({ max: 20 })]);
    for (let i = 0; i < 4; i++) s.cloth.impulse('skirt', undefined, 1e9, 1e9);
    const g = s.states[0];
    const length = 100, cap = Math.min(IMPULSE_LIMITS.cloth / length, CLOTH_SWING_RATE * 2 * Math.asin(20 / (2 * length)));
    for (let k = 0; k < g.cols; k++) {
      expect(Math.abs(g.angularVelocity[k])).toBeCloseTo(cap, 9);
      expect(Math.abs(g.lowerAngularVelocity[k])).toBeCloseTo(cap, 9);
    }
  });

  test('custom grids are pushed once stepped; collapsed columns and unknown garments are left alone', () => {
    const seams = { seams: [[[0, 0], [-5, 40], [0, 100]], [[100, 0], [110, 40], [100, 120]]] } as Partial<Cloth>;
    const s = simulation([garment(seams)]);
    expect(s.cloth.impulse('skirt', undefined, 60, 0)).toBe(false); // no pendulums before the first step
    s.step();
    expect(s.cloth.impulse('skirt', undefined, 60, 0)).toBe(true);
    for (const p of s.states[0].pendulums) expect(p.velocities.every((v: number) => v > 0)).toBe(true);

    const collapsed = garment(); collapsed.hem = structuredClone(collapsed.waist);
    const c = simulation([collapsed]);
    expect(c.cloth.impulse('skirt', 0.5, 60, 0)).toBe(false);
    expect(Array.from(c.states[0].angularVelocity)).toEqual(new Array(8).fill(0));
    expect(c.cloth.impulse('missing', 0.5, 60, 0)).toBe(false);
    expect(c.cloth.impulse('toString', 0.5, 60, 0)).toBe(false);
  });
});

describe('targets that do not exist', () => {
  test('are ignored without throwing, as are limb targets and non-finite vectors', () => {
    const rig = parseRig(fixture);
    rig.cloth = [{ name: 'skirt', outline: [[0, 0], [100, 0], [100, 100], [0, 100]], waist: [[0, 0], [100, 0]], hem: [[0, 100], [100, 100]] } as Cloth];
    const { motion, physics, impulse } = parts(rig);
    const state = () => JSON.stringify({ vel: motion.vel, strands: physics.strands });
    const before = state();
    const calls: [unknown, number, number][] = [
      [null, 10, 10], [undefined, 10, 10], ['head', 10, 10], [{}, 10, 10], [{ kind: 'wing' }, 10, 10],
      [{ kind: 'limb', name: 'armL' }, 10, 10], [{ kind: 'strand', name: 'missing' }, 10, 10], [{ kind: 'strand' }, 10, 10],
      [{ kind: 'cloth', name: 'missing', at: 0.5 }, 10, 10], [{ kind: 'cloth' }, 10, 10],
      [{ kind: 'head' }, Number.NaN, 10], [{ kind: 'body' }, Infinity, 0], [{ kind: 'strand', name: 'sideL' }, 10, Number.NaN],
    ];
    for (const [target, vx, vy] of calls) expect(() => impulse(target as ImpulseTarget, vx, vy)).not.toThrow();
    expect(state()).toBe(before);
  });
});
