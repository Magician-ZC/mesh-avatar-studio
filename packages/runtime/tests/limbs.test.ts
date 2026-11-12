// Generic limb chains (ER4): validation, the skinning weights (by bone direction, not image height), a limb that is
// exactly the drawing at rest, a root that moves with the base so the cut never opens, swing and bend at their limits
// without flipped triangles, impulses that settle within 3 s, idle sway, and the files a limb avatar needs.
import { describe, expect, test } from 'vitest';
import fixture from '../../../samples/miko-qipao/rig.json';
import { parseRig, validateAvatarFiles, validateRig, PARAMS, type Limb, type Rig } from 'mesh-avatar';
import { createRig } from '../src/engine/rig.js';
import { createRenderer } from '../src/engine/renderer.js';
import { createPhysics } from '../src/engine/physics.js';
import { createImpulse } from '../src/interaction/impulse';
import { IMPULSE_LIMITS } from '../src/interaction/limits';
import { bendParameter, limbChain, limbWeight, swingParameter } from '../src/limbs/chain';
import { createLimbs } from '../src/limbs/runtime';
import { triangleMetrics } from '../../../tools/quality-metrics.mjs';

const defaults: Record<string, number> = Object.fromEntries(PARAMS.map(p => [p.id, p.def]));

/** A raised forearm-and-hand chain beside Miko's body: root on the shoulder, bent at the elbow. */
const ARM: Limb = {
  name: 'arm_r', kind: 'arm', joints: [[980, 1000], [1110, 1150], [1180, 1000]],
  outline: [[930, 960], [1040, 960], [1150, 1080], [1170, 960], [1230, 960], [1230, 1060], [1150, 1220], [1060, 1200], [930, 1060]],
};

function rigWith(limbs: Limb[] = [ARM], extra: Partial<Rig> = {}): Rig {
  return { ...parseRig(fixture), limbs, ...extra };
}

describe('validation', () => {
  test('a well-formed limb passes and is kept by parseRig', () => {
    expect(validateRig(rigWith())).toEqual([]);
    expect(parseRig(rigWith()).limbs).toEqual([ARM]);
    expect(validateRig(rigWith([{ ...ARM, behind: true, swing: 10, bend: 12 }]))).toEqual([]);
  });

  test.each([
    ['a name with other characters', { name: 'arm-1' }, 'name'],
    ['joints outside the image', { joints: [[980, 1000], [1110, 1150], [1300, 1000]] }, 'joints: must stay inside'],
    ['joints on top of each other', { joints: [[980, 1000], [980, 1000], [1180, 1000]] }, 'at least 1 px apart'],
    ['bones folded back', { joints: [[980, 1000], [1110, 1000], [980, 1000.5]] }, 'fold back'],
    ['an outline without area', { outline: [[10, 10], [20, 20], [30, 30]] }, 'positive area'],
    ['an outline leaving the image', { outline: [[930, 960], [1300, 960], [1100, 1200]] }, 'outline: points must stay inside'],
    ['a swing beyond 10 degrees', { swing: 10.5 }, 'swing'],
    ['a bend beyond 12 degrees', { bend: 13 }, 'bend'],
    ['an unknown kind', { kind: 'tail' }, 'kind'],
  ] as [string, Partial<Limb>, string][])('rejects %s', (_case, change, message) => {
    expect(validateRig(rigWith([{ ...ARM, ...change } as Limb])).join('\n')).toContain(message);
  });

  test('names are unique, and a limb may not also be bound as hanging arms', () => {
    expect(validateRig(rigWith([ARM, { ...ARM }])).join()).toContain('unique');
    const arms = { left: { shoulder: [300, 900], elbow: [290, 1050], wrist: [285, 1180], split: [[330, 950], [330, 1240]] },
      right: { shoulder: [1000, 980], elbow: [1100, 1100], wrist: [1150, 1180], split: [[1060, 1000], [1060, 1240]] } } as Rig['arms'];
    expect(validateRig(rigWith([ARM], { arms })).join()).toContain('must not also be bound in arms or legs');
  });
});

describe('the chain', () => {
  test('weights follow the bones: the root fades in along the first bone, the joint blend crosses the bisector', () => {
    const horizontal = limbChain({ ...ARM, joints: [[100, 100], [300, 100], [500, 100]] });
    // root: 0 at and behind the root, 1 past ROOT_BLEND of the first bone, whatever the height
    expect(limbWeight(horizontal, 100, 100).root).toBe(0);
    expect(limbWeight(horizontal, 60, 400).root).toBe(0);
    expect(limbWeight(horizontal, 200, 100).root).toBe(1);
    expect(limbWeight(horizontal, 200, -300).root).toBe(1);
    // joint: half at the middle joint, 0 and 1 on either side along x; the same at any height
    expect(limbWeight(horizontal, 300, 100).joint).toBeCloseTo(0.5, 12);
    expect(limbWeight(horizontal, 300, 160).joint).toBeCloseTo(0.5, 12);
    expect(limbWeight(horizontal, 250, 100).joint).toBe(0);
    expect(limbWeight(horizontal, 350, 140).joint).toBe(1);
    // a bent chain splits along the bisector of the bones
    const bent = limbChain({ ...ARM, joints: [[100, 100], [300, 100], [300, 300]] });
    expect(limbWeight(bent, 250, 100).joint).toBeLessThan(0.05);
    expect(limbWeight(bent, 300, 150).joint).toBeGreaterThan(0.95);
    expect(bent.bendSign).toBe(1); // drawn turning clockwise: positive bend turns further clockwise
    expect(limbChain({ ...ARM, joints: [[100, 100], [300, 100], [300, -100]] }).bendSign).toBe(-1);
  });
});

/** The limb mesh of ARM (a grid over its outline, as the renderer builds it) and the deformation of a frame. */
function scene(rig: Rig = rigWith()) {
  const engine = createRig(rig);
  const { buildGrid } = createRenderer(engine, rig);
  const { Physics } = createPhysics(engine, rig);
  const physics = new Physics();
  const limbs = createLimbs(rig, engine)!;
  const outline = rig.limbs![0].outline;
  const xs = outline.map(p => p[0]), ys = outline.map(p => p[1]);
  const rect = [Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)), Math.ceil(Math.max(...xs) - Math.min(...xs)), Math.ceil(Math.max(...ys) - Math.min(...ys))];
  const mesh = buildGrid(rect, 8, new Uint8Array([255]), 1);
  limbs.attach(rig.limbs![0].name, mesh);
  const frame = (P: Record<string, number>, dt = 0) => {
    const values = { ...defaults, ...P };
    const phys = physics.step(values, dt);
    limbs.step(dt);
    limbs.deform(values, phys);
    return { values, phys };
  };
  return { engine, limbs, mesh, frame };
}

describe('deformation', () => {
  test('with every parameter at 0 the limb is exactly where it is drawn', () => {
    const { mesh, frame } = scene();
    frame({ [swingParameter('arm_r')]: 0, [bendParameter('arm_r')]: 0 });
    let largest = 0;
    for (let i = 0; i < mesh.rest.length; i++) largest = Math.max(largest, Math.abs(mesh.pos[i] - mesh.rest[i]));
    expect(largest).toBe(0);
  });

  test.each([
    ['swing +1', { 'limb:arm_r:swing': 1 }], ['swing -1', { 'limb:arm_r:swing': -1 }],
    ['bend +1', { 'limb:arm_r:bend': 1 }], ['bend -1', { 'limb:arm_r:bend': -1 }],
    ['both at the limit with the body leaning and breathing', { 'limb:arm_r:swing': 1, 'limb:arm_r:bend': 1, bodyAngleZ: -8, bodyAngleX: 6, breath: 1 }],
    ['head turned and tilted', { 'limb:arm_r:swing': -1, 'limb:arm_r:bend': -1, angleX: 30, angleZ: -15 }],
  ] as [string, Record<string, number>][])('%s: the root cut moves with the base and no triangle flips', (_name, P) => {
    const { engine, mesh, frame } = scene();
    const { values, phys } = frame(P);
    const chain = limbChain(ARM);
    const out = [0, 0];
    let gap = 0, rootVertices = 0;
    for (let v = 0; v < mesh.rest.length / 2; v++) {
      const x = mesh.rest[v * 2], y = mesh.rest[v * 2 + 1];
      if (limbWeight(chain, x, y).root > 0) continue;
      engine.deformBase(x, y, engine.baseWeights(x, y, 0), values, phys, out);
      gap = Math.max(gap, Math.hypot(mesh.pos[v * 2] - out[0], mesh.pos[v * 2 + 1] - out[1]));
      rootVertices++;
    }
    expect(rootVertices).toBeGreaterThan(10);
    expect(gap).toBeLessThan(2e-4); // the cut where the limb meets the body never opens (float32 vertex buffers)
    expect(triangleMetrics(mesh).flipped).toBe(0);
  });

  test('swing turns the chain about the moved root by the swing limit, bend the second bone about the middle joint', () => {
    const { limbs } = scene();
    const chain = limbChain(ARM);
    expect(chain.swingLimit).toBeCloseTo(4 * Math.PI / 180, 12);
    expect(chain.bendLimit).toBeCloseTo(6 * Math.PI / 180, 12);
    expect(limbs.values('arm_r', { [swingParameter('arm_r')]: 3, [bendParameter('arm_r')]: -2 })).toEqual({ swing: 1, bend: -1 });
    const { mesh, frame } = scene();
    frame({ [swingParameter('arm_r')]: 1 });
    // a vertex on the end of the first bone, far from the root and the joint: turned by 4 degrees about the root
    const v = nearest(mesh, [1060, 1090]);
    const [x, y] = [mesh.rest[v * 2], mesh.rest[v * 2 + 1]];
    const turned = [980 + (x - 980) * Math.cos(chain.swingLimit) - (y - 1000) * Math.sin(chain.swingLimit),
      1000 + (x - 980) * Math.sin(chain.swingLimit) + (y - 1000) * Math.cos(chain.swingLimit)];
    expect(limbWeight(chain, x, y)).toEqual({ root: 1, joint: 0 });
    expect(mesh.pos[v * 2]).toBeCloseTo(turned[0], 3);
    expect(mesh.pos[v * 2 + 1]).toBeCloseTo(turned[1], 3);
  });
});

function nearest(mesh: { rest: ArrayLike<number> }, [px, py]: [number, number]) {
  let best = 0, distance = Infinity;
  for (let v = 0; v < mesh.rest.length / 2; v++) {
    const d = Math.hypot(mesh.rest[v * 2] - px, mesh.rest[v * 2 + 1] - py);
    if (d < distance) { distance = d; best = v; }
  }
  return best;
}

describe('impulses and idle sway', () => {
  test('a push swings the limb and it settles within 3 s (< 0.1 px); pushes are bounded; unknown limbs are ignored', () => {
    const { limbs, mesh, frame } = scene();
    const impulse = createImpulse(rigWith(), { motion: { vel: {} }, physics: { strands: [] }, cloth: { impulse: () => false }, limbs });
    frame({}, 1 / 60);
    const rest = Float32Array.from(mesh.pos);
    impulse({ kind: 'limb', name: 'arm_r' }, 900, -400);
    let peak = 0;
    for (let f = 0; f < 180; f++) {
      frame({}, 1 / 60);
      let largest = 0;
      for (let i = 0; i < mesh.pos.length; i++) largest = Math.max(largest, Math.abs(mesh.pos[i] - rest[i]));
      peak = Math.max(peak, largest);
      if (f === 179) expect(largest).toBeLessThan(0.1);
    }
    expect(peak).toBeGreaterThan(5);
    // a huge push is truncated: the swing stays within its limit
    impulse({ kind: 'limb', name: 'arm_r' }, 1e9, 1e9);
    for (let f = 0; f < 30; f++) {
      frame({}, 1 / 60);
      expect(Math.abs(limbs.values('arm_r', {})!.swing)).toBeLessThanOrEqual(1);
    }
    expect(IMPULSE_LIMITS.limb).toBeGreaterThan(0);
    expect(limbs.impulse('missing', 100, 100)).toBe(false);
    expect(limbs.impulse('arm_r', Number.NaN, 0)).toBe(false);
  });

  test('idle sway is small and slow, only for limbs without parameters, and off at gain 0', () => {
    const { limbs } = scene();
    const values: number[] = [];
    for (let t = 0; t < 20; t += 0.05) {
      const P: Record<string, number> = {};
      limbs.idle(P, t, 1);
      values.push(P[swingParameter('arm_r')]);
      expect(Math.abs(P[swingParameter('arm_r')])).toBeLessThanOrEqual(0.3);
      expect(Math.abs(P[bendParameter('arm_r')])).toBeLessThanOrEqual(0.2);
    }
    const steps = values.slice(1).map((v, i) => Math.abs(v - values[i]));
    expect(Math.max(...steps)).toBeLessThan(0.03); // under 0.6 per second
    const set: Record<string, number> = { [swingParameter('arm_r')]: 0.5 };
    limbs.idle(set, 3, 1);
    expect(set[swingParameter('arm_r')]).toBe(0.5);
    const still: Record<string, number> = {};
    limbs.idle(still, 3, 0);
    expect(still[swingParameter('arm_r')]).toBe(0);
  });

  test('a rig without limbs has no limb runtime', () => {
    expect(createLimbs(parseRig(fixture), createRig(parseRig(fixture)))).toBeNull();
  });
});

describe('files of a limb avatar', () => {
  const json = (value: unknown) => new TextEncoder().encode(JSON.stringify(value));
  const rig = rigWith();
  const layers = (extra: Record<string, unknown> = {}) => ({
    version: 1, size: [rig.image.width, rig.image.height], build: 'b1',
    layers: { ...Object.fromEntries(['ball', 'low', 'crease', 'lash'].flatMap(part => [0, 1].map(i => [`eye${i}_${part}`, [0, 0, 1, 1]]))),
      hand: [0, 0, 1, 1], tassel_l: [0, 0, 1, 1], tassel_r: [0, 0, 1, 1], limb_arm_r: [930, 960, 300, 260] },
    eyes: rig.eyes.map(e => ({ x0: e.x0, x1: e.x1, top: e.top, bot: e.bot })),
    ...extra,
  });
  const files = (meta: Record<string, unknown>, images: string[]) => ({
    'rig.json': json(rig), 'built/layers.json': json(meta),
    ...Object.fromEntries(['base', 'hairmask', ...Object.keys(meta.layers as object), ...images].map(name => [`built/${name}.png`, { size: 1 }])),
  });

  test('need the limb layers and the bare base named by layers.json', () => {
    expect(validateAvatarFiles(files(layers({ bare: 'base_bare' }), ['base_bare']))).toEqual([]);
    expect(validateAvatarFiles(files(layers(), []))).toContain('built/layers.json.bare: required for rig.limbs (the base without the limbs)');
    expect(validateAvatarFiles(files(layers({ bare: 'base_bare' }), []))).toContain('built/base_bare.png: missing file');
    expect(validateAvatarFiles(files(layers({ bare: '../x' }), []))).toContain('built/layers.json.bare: expected a safe layer name');
    const without = layers({ bare: 'base_bare' });
    delete (without.layers as Record<string, unknown>).limb_arm_r;
    expect(validateAvatarFiles(files(without, ['base_bare']))).toContain('built/layers.json.layers.limb_arm_r: missing required layer');
  });
});
