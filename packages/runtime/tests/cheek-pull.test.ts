// Cheek pull (T-ER12, interaction/cheek.ts): the contour of a held cheek moves outwards in an arc and never inwards, the
// eyes, nose and chin side of the cheek (beyond its reach) stay, a release returns exactly to rest, and the data is
// validated before it is used.
import { describe, expect, test } from 'vitest';
import {
  CHEEK_LIMITS, CheekPull, cheekBump, cheekDisplace, cheekField, cheekPose, cheekProblems, limitPull, onEdge, parseCheeks,
  type CheekEdge,
} from '../src/interaction/cheek.ts';
import type { Point } from '../src/rig/types';

// a left contour running straight down at x = 100 from y = 100 to 300; the face is to its right
const edge: CheekEdge = { edge: [[100, 100], [100, 150], [100, 200], [100, 250], [100, 300]], reach: [60, 60, 60, 60, 60], travel: 20 };
const right: CheekEdge = { edge: [[300, 100], [300, 200], [300, 300]], reach: [60, 60, 60], travel: 20 };
// a grid of rest vertices around the contour, 10 px apart, from 20 px outside to 80 px inside
const rest: number[] = [];
for (let y = 90; y <= 310; y += 10) for (let x = 80; x <= 180; x += 10) rest.push(x, y);
const at = (x: number, y: number) => { for (let k = 0; k < rest.length / 2; k++) if (rest[k * 2] === x && rest[k * 2 + 1] === y) return k; throw new Error('no vertex'); };
const frame = { rest: [100, 200] as Point, drawn: [100, 200] as Point, angle: 0 };

function pulled(offset: Point, depth = 0, seconds = 1, pinch: Point = [110, 200]) {
  const cheeks = new CheekPull({ left: edge }, { left: rest });
  cheeks.hold('left', pinch[0], pinch[1], frame);
  cheeks.move('left', offset[0], offset[1], depth);
  for (let i = 0; i < seconds * 60; i++) cheeks.step(1 / 60, { left: 0 });
  return cheeks;
}

describe('cheek data', () => {
  test('accepts either side and reports what is wrong', () => {
    expect(cheekProblems({ left: edge, right })).toEqual([]);
    expect(cheekProblems({ left: { ...edge, reach: [1, 2] } })[0]).toMatch(/reach/);
    expect(cheekProblems({ left: { ...edge, travel: 0 } })[0]).toMatch(/travel/);
    expect(cheekProblems({ left: { ...edge, edge: [[0, 0], [0, 0], [1, 1]], reach: [1, 1, 1] } })[0]).toMatch(/repeated/);
    expect(cheekProblems({ middle: edge })[0]).toMatch(/unknown side/);
    expect(cheekProblems([])[0]).toMatch(/expected an object/);
  });

  test('keeps only valid sides', () => {
    expect(Object.keys(parseCheeks({ left: edge, right: { ...right, travel: -1 } }))).toEqual(['left']);
    expect(parseCheeks(undefined)).toEqual({});
    expect(parseCheeks({ left: { ...edge, edge: 'x' } })).toEqual({});
  });
});

describe('cheek field', () => {
  const field = cheekField(edge, 'left', rest);
  test('measures along the contour, inwards from it and outwards', () => {
    const hit = onEdge(edge, 'left', 130, 200);
    expect(hit.along).toBeCloseTo(0.5, 9);
    expect(hit.depth).toBeCloseTo(30, 9);
    expect(hit.normal[0]).toBeCloseTo(-1, 9);
    expect(onEdge(edge, 'left', 90, 200).depth).toBeCloseTo(-10, 9);
    expect(onEdge(right, 'right', 310, 200).normal[0]).toBeCloseTo(1, 9);
    expect(field.outward[0]).toBeCloseTo(-1, 9);
  });

  test('leaves out what a pull cannot move: beyond the reach and past both ends', () => {
    const moved = new Set(field.index);
    expect(moved.has(at(100, 200))).toBe(true);
    expect(moved.has(at(160, 200))).toBe(false); // 60 px inside: the reach
    expect(moved.has(at(110, 90))).toBe(false); // above the contour's top
  });
});

describe('cheek pull', () => {
  test('pulls the contour outwards in an arc, as far as the travel, and leaves the far skin', () => {
    const cheeks = pulled([-200, 0]);
    const d = cheeks.displacement('left')!;
    const middle = d[at(100, 200) * 2], upper = d[at(100, 150) * 2], top = d[at(100, 100) * 2];
    expect(middle).toBeLessThan(-0.95 * edge.travel);
    expect(middle).toBeGreaterThanOrEqual(-edge.travel - 1e-9);
    expect(upper).toBeLessThan(0); expect(upper).toBeGreaterThan(middle); // an arc: less towards the ends
    expect(top).toBeCloseTo(0, 9);
    expect(d[at(160, 200) * 2]).toBe(0);
    // the drawn line just outside the contour goes with it
    expect(d[at(90, 200) * 2]).toBeCloseTo(middle, 9);
  });

  test('a pull inwards or along the contour moves the skin inside it, never the contour', () => {
    for (const offset of [[30, 0], [0, 40], [0, -40], [25, 30]] as Point[]) {
      const d = pulled(offset).displacement('left')!;
      for (const y of [150, 200, 250]) {
        expect(Math.abs(d[at(100, y) * 2])).toBeLessThan(1e-9);
        expect(Math.abs(d[at(100, y) * 2 + 1])).toBeLessThan(1e-9);
      }
      const inner = at(130, 200);
      expect(Math.hypot(d[inner * 2], d[inner * 2 + 1])).toBeGreaterThan(1);
    }
  });

  test('a diagonal pull keeps its vertical part and moves the bulge with it', () => {
    const d = pulled([-15, 30]).displacement('left')!;
    expect(d[at(100, 200) * 2]).toBeLessThan(-5);
    expect(d[at(120, 200) * 2 + 1]).toBeGreaterThan(1);
    const down = cheekPose(cheekField(edge, 'left', rest), 20, [-10, 10], 0.5, 0, [110, 200], 50).centre;
    expect(down).toBeGreaterThan(0.5);
  });

  test('towards the camera the skin swells around the pinch and the contour hardly moves', () => {
    const d = pulled([0, 0], 1).displacement('left')!;
    const below = at(120, 230), above = at(120, 170);
    expect(d[below * 2 + 1]).toBeGreaterThan(0.3);
    expect(d[above * 2 + 1]).toBeLessThan(-0.3);
    expect(Math.abs(d[at(100, 200) * 2])).toBeLessThan(0.25 * edge.travel);
    expect(d[at(100, 200) * 2]).toBeLessThanOrEqual(0);
    expect(pulled([0, 0], 1).shading('left')[3]).toBeGreaterThan(0.9);
  });

  test('holding still keeps the shape; a release returns exactly to rest', () => {
    const cheeks = pulled([-200, 0]);
    const before = Array.from(cheeks.displacement('left')!);
    for (let i = 0; i < 30; i++) cheeks.step(1 / 60, { left: 0 });
    Array.from(cheeks.displacement('left')!).forEach((v, i) => expect(v).toBeCloseTo(before[i], 6));
    cheeks.release('left', -5000, 0, 0);
    let frames = 0;
    while (cheeks.displacement('left') && frames++ < 600) cheeks.step(1 / 60, { left: 0 });
    expect(cheeks.displacement('left')).toBeNull();
    expect(cheeks.shading('left')).toEqual([0, 0, 0, 0]);
  });

  test('where hair lies over the contour the pull eases out (free), elsewhere it is whole', () => {
    expect(cheekProblems({ left: { ...edge, free: [1, 1, 0, 1] } })[0]).toMatch(/free/);
    expect(cheekProblems({ left: { ...edge, free: [1, 1, 2, 1, 1] } })[0]).toMatch(/free/);
    const covered = { ...edge, free: [1, 0, 0, 1, 1] };
    expect(parseCheeks({ left: covered }).left?.free).toEqual([1, 0, 0, 1, 1]);
    const cheeks = new CheekPull({ left: covered }, { left: rest });
    cheeks.hold('left', 110, 250, frame);
    cheeks.move('left', -200, 0, 0);
    for (let i = 0; i < 60; i++) cheeks.step(1 / 60, { left: 0 });
    const d = cheeks.displacement('left')!;
    expect(d[at(100, 160) * 2]).toBe(0); // under the hair: stays
    expect(d[at(100, 250) * 2]).toBeLessThan(-5);
  });

  test('a release speed is bounded', () => {
    const cheeks = pulled([0, 0]);
    cheeks.release('left', 1e6, 0, 0);
    let widest = 0;
    for (let i = 0; i < 60; i++) { cheeks.step(1 / 60, { left: 0 }); widest = Math.max(widest, Math.abs(cheeks.displacement('left')?.[at(100, 200) * 2] ?? 0)); }
    expect(widest).toBeLessThanOrEqual(edge.travel + 1e-9);
    expect(widest).toBeLessThan(CHEEK_LIMITS.release / 16);
  });

  test('a turned head turns the pull into rest space', () => {
    const cheeks = new CheekPull({ left: edge }, { left: rest });
    cheeks.hold('left', 110, 200, frame);
    cheeks.move('left', 0, 200, 0); // straight down on screen ...
    for (let i = 0; i < 60; i++) cheeks.step(1 / 60, { left: -Math.PI / 2 }); // ... is outwards for a head turned a quarter
    expect(cheeks.displacement('left')![at(100, 200) * 2]).toBeLessThan(-0.9 * edge.travel);
  });

  test('nothing to do without data or a mesh; limits stay within the travel', () => {
    expect(new CheekPull({}, { left: rest }).has('left')).toBe(false);
    expect(new CheekPull({ left: edge }, {}).has('left')).toBe(false);
    const field = cheekField(edge, 'left', rest);
    const [x, y] = limitPull(field, 20, [-1000, 1000]);
    expect(-x).toBeLessThanOrEqual(20);
    expect(y).toBeLessThanOrEqual(CHEEK_LIMITS.along * 20);
    expect(cheekBump(0)).toBe(0); expect(cheekBump(1)).toBe(0); expect(cheekBump(0.5)).toBe(1);
    const out = new Float64Array(rest.length);
    cheekDisplace(field, cheekPose(field, 20, [0, 0], 0.5, 0, [0, 0], 0), rest, out);
    expect(out.every(v => v === 0)).toBe(true);
  });
});
