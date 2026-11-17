// Sustained grabs (T-ER12, interaction/grab.ts): handles, refusals, and each part held at the pointer within its own
// limits, then handed back to its own dynamics.
import { describe, expect, test } from 'vitest';
import fixture from '../../../samples/miko-qipao/rig.json';
import { parseRig, PARAMS } from 'mesh-avatar';
import { createRig } from '../src/engine/rig.js';
import { createPhysics } from '../src/engine/physics.js';
import { createGrabs, type HeldGarment } from '../src/interaction/grab.ts';
import type { ImpulseTarget } from '../src/interaction/impulse.ts';
import type { Point } from '../src/rig/types';

const rig = parseRig(fixture);
const P: Record<string, number> = Object.fromEntries(PARAMS.map(p => [p.id, p.def]));

// a garment of 5 straight columns, 100 px long, hems at y = 300, its travel 12 px
function garment(): HeldGarment {
  const cols = 5, rows = 3, rest: number[] = [];
  for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) rest.push(100 + k * 20, 200 + r * 50);
  return {
    c: { hem: [[100, 300], [180, 300]] }, cols, rows, rest, pinned: Array(cols).fill(false), grid: { custom: false },
    tuning: { max: 12 }, angle: new Float64Array(cols), lowerAngle: new Float64Array(cols),
    angularVelocity: new Float64Array(cols).fill(1), lowerAngularVelocity: new Float64Array(cols).fill(1), pendulums: [],
  };
}

function setup() {
  const engine = createRig(rig);
  const { Physics } = createPhysics(engine, rig);
  const physics = new Physics();
  for (let i = 0; i < 30; i++) physics.step(P, 1 / 60);
  const impulses: [ImpulseTarget, number, number][] = [];
  const skirt = garment();
  const parameters: Record<string, number> = { ...P };
  const limbs = {
    chains: [{ name: 'arm', root: [0, 0] as Point, middle: [0, 50] as Point, end: [0, 100] as Point, swingLimit: 0.1, bendLimit: 0.2 }],
    values: (name: string, values: Record<string, number>) => name === 'arm'
      ? { swing: Math.max(-1, Math.min(1, values['limb:arm:swing'] ?? 0)), bend: Math.max(-1, Math.min(1, values['limb:arm:bend'] ?? 0)) } : null,
  };
  const grabs = createGrabs({
    rig, physics, limbs, cheeks: null, cheekFrame: () => null, place: p => [p[0], p[1]],
    garment: name => name === 'skirt' ? skirt : null, parameters: () => parameters,
    impulse: (target, vx, vy) => { impulses.push([target, vx, vy]); },
  });
  const run = (frames: number) => { for (let i = 0; i < frames; i++) { grabs.step(1 / 60); grabs.beforePhysics(P); physics.step(P, 1 / 60); } };
  return { physics, grabs, impulses, skirt, run, parameters };
}

const strandIndex = rig.strands!.findIndex(s => s.name === 'sideR');
const tip = (physics: ReturnType<typeof setup>['physics']) => (physics.out as unknown as { strands: number[][][] }).strands[strandIndex][3];

describe('grab', () => {
  test('refuses what this avatar cannot hold, and bad input', () => {
    const { grabs } = setup();
    for (const target of [{ kind: 'cheek', side: 'left' }, { kind: 'strand', name: 'nope' }, { kind: 'cloth', name: 'nope' },
      { kind: 'limb', name: 'nope' }, { kind: 'accessory', name: 'nope' }, { kind: 'head' }, { kind: 'cheek', side: 'up' }, null]) {
      expect(grabs.grab(target, 10, 10)).toBeNull();
      expect(grabs.can(target)).toBe(false);
    }
    expect(grabs.grab({ kind: 'strand', name: 'sideR' }, Number.NaN, 0)).toBeNull();
    expect(grabs.grab({ kind: 'cloth', name: 'skirt', at: Number.NaN }, 0, 0)).toBeNull();
    expect(grabs.active).toBe(false);
  });

  test('a strand is held at the pointer, keeps still while held and springs back when let go', () => {
    const { physics, grabs, impulses, run } = setup();
    const nodes = rig.strands![strandIndex].nodes, end = nodes[nodes.length - 1];
    const handle = grabs.grab({ kind: 'strand', name: 'sideR' }, end[0], end[1])!;
    expect(handle).toBeGreaterThan(0);
    expect(Number.isInteger(handle)).toBe(true);
    grabs.move(handle, 30, 0);
    run(60);
    const held = tip(physics)[0];
    expect(held).toBeGreaterThan(0.5 * rig.strands![strandIndex].max);
    expect(held).toBeLessThanOrEqual(rig.strands![strandIndex].max + 1e-9);
    run(30);
    expect(tip(physics)[0]).toBeCloseTo(held, 0);
    grabs.release(handle, 200, 0);
    expect(impulses).toEqual([[{ kind: 'strand', name: 'sideR' }, 200, 0]]);
    run(240);
    expect(Math.abs(tip(physics)[0])).toBeLessThan(2);
    // stale and unknown handles are ignored
    grabs.release(handle, 1, 1); grabs.move(handle, 50, 0); grabs.release(999, 1, 1);
    expect(impulses.length).toBe(1);
  });

  test('a new grab of the same part takes it over; handles keep counting up', () => {
    const { physics, grabs, run } = setup();
    const a = grabs.grab({ kind: 'strand', name: 'sideR' }, 0, 0)!, b = grabs.grab({ kind: 'strand', name: 'sideR' }, 0, 0)!;
    expect(b).toBeGreaterThan(a);
    grabs.move(a, 40, 0); // stale: ignored
    run(30);
    expect(Math.abs(tip(physics)[0])).toBeLessThan(2);
  });

  test('a tassel points at the pointer within its swing, and a release gives it a bounded turn', () => {
    const { physics, grabs, run } = setup();
    type Tassel = { t: { name: string; tip: number[] }; phi: number[]; st: { w: number[] } };
    const chain = physics.chains.find(c => c.t.name === 'tassel_l') as unknown as Tassel;
    const handle = grabs.grab({ kind: 'accessory', name: 'tassel_l' }, chain.t.tip[0], chain.t.tip[1])!;
    grabs.move(handle, 300, 0);
    run(20);
    expect(Math.abs(chain.phi[0])).toBeGreaterThan(0.3);
    expect(Math.abs(chain.phi[0])).toBeLessThanOrEqual(0.55 + 1e-9);
    grabs.release(handle, 0, 1e6);
    expect(Math.abs(chain.st.w[0])).toBeLessThanOrEqual(6);
  });

  test('a hem follows the pointer sideways, never beyond its travel and never up', () => {
    const { grabs, skirt, impulses } = setup();
    const handle = grabs.grab({ kind: 'cloth', name: 'skirt' }, 140, 300)!; // the middle column's hem
    grabs.move(handle, 100, 0);
    grabs.beforeCloth();
    const limit = 0.95 * 2 * Math.asin(12 / 200);
    expect(skirt.angle[2]).toBeCloseTo(limit, 9);
    expect(skirt.angularVelocity[2]).toBe(0);
    expect(skirt.angle[0]).toBe(0); // two columns away: untouched
    grabs.move(handle, 0, -100);
    for (let i = 0; i < 5; i++) grabs.beforeCloth();
    expect(Math.abs(skirt.angle[2])).toBeLessThan(1e-3);
    grabs.release(handle, 50, 0);
    expect(impulses).toEqual([[{ kind: 'cloth', name: 'skirt', at: 0.5 }, 50, 0]]);
  });

  test('a limb swings towards the pointer within its limit and settles back after the release', () => {
    const { grabs } = setup();
    const handle = grabs.grab({ kind: 'limb', name: 'arm' }, 0, 100)!;
    grabs.move(handle, 10, 0);
    const held: Record<string, number> = {};
    grabs.parameters(held);
    expect(held['limb:arm:swing']).toBeCloseTo((Math.atan2(100, 10) - Math.PI / 2) / 0.1, 9);
    grabs.move(handle, 1000, 0);
    grabs.parameters(held);
    expect(held['limb:arm:swing']).toBe(-1);
    grabs.release(handle);
    let frames = 0, last: Record<string, number> = {};
    do { last = {}; grabs.step(1 / 60); grabs.parameters(last); } while (grabs.active && ++frames < 1200);
    expect(grabs.active).toBe(false);
    expect(Math.abs(last['limb:arm:swing'] ?? 0)).toBeLessThan(0.01);
  });

  test('releaseAll lets go of everything', () => {
    const { grabs, impulses } = setup();
    grabs.grab({ kind: 'strand', name: 'sideL' }, 0, 0);
    grabs.grab({ kind: 'cloth', name: 'skirt' }, 0, 0);
    grabs.releaseAll();
    expect(grabs.active).toBe(false);
    expect(impulses.map(([t]) => t.kind)).toEqual(['strand', 'cloth']);
  });
});
