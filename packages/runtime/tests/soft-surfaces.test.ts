import { expect, test } from 'vitest';
import { SoftSurfaces, type SurfaceTarget } from '../src/interaction/soft-surfaces.ts';
import type { Cloth, Point, Rig } from '../src/rig/types';
import { sampleFiles } from './avatar-fixture.ts';

// A 10 px square grid exposes the geometry and light sent to the renderer each frame.
function surface(material?: Cloth['material'], approved = true) {
  const rest: number[] = [], triangles: number[] = [];
  for (let i = 0; i < 121; i++) {
    rest.push((i % 11) * 10, Math.floor(i / 11) * 10);
    if (i < 110 && i % 11 < 10) triangles.push(i, i + 1, i + 11, i + 1, i + 12, i + 11);
  }
  const mesh = { rest: Float32Array.from(rest), pos: Float32Array.from(rest), tris: Uint32Array.from(triangles) };
  const layer = { mesh, surfaceLight: new Float32Array(12) };
  const rig = JSON.parse(new TextDecoder().decode(sampleFiles()['rig.json'])) as Rig;
  rig.body.softChest = { left: { cx: 30, cy: 50, rx: 25, ry: 30 }, right: { cx: 70, cy: 50, rx: 25, ry: 30 } };
  const cloth: Cloth = { name: 'sample', material, outline: [[0, 0], [100, 0], [100, 100], [0, 100]], waist: [[0, 0], [100, 0]], hem: [[0, 100], [100, 100]], reach: 30, max: 5 };
  if (material === 'knit' || material === 'woven') cloth.neckline = [[0, 0], [50, 10], [100, 0]];
  rig.cloth = material ? [cloth] : [];
  const model = new SoftSurfaces(rig, new Set(approved ? ['sample'] : []));
  model.attach(layer, material ? cloth : undefined);
  const target: SurfaceTarget = material ? { kind: 'fabric', name: 'sample' } : { kind: 'chest', side: 'left' };
  function frames(count: number) {
    for (let i = 0; i < count; i++) { mesh.pos.set(mesh.rest); model.step(1 / 60, (point: Point) => point); }
  }
  function offset(x: number, y: number) {
    const i = (x / 10 + 11 * y / 10) * 2;
    return [mesh.pos[i] - x, mesh.pos[i + 1] - y];
  }
  return { model, target, frames, offset, light: layer.surfaceLight };
}

test('chest follows a circular hold, its depth expands or compresses independently, and the free side responds weakly', () => {
  const { model, target, frames, offset, light } = surface();
  expect(model.hold(target, 30, 50)).toBe(true);
  for (const [x, y] of [[1, 0], [0, 1], [-1, 0], [0, -1]]) {
    model.move(target, x * 30, y * 30, 0); frames(45);
    const [dx, dy] = offset(30, 50);
    expect(x * dx + y * dy).toBeGreaterThan(7);
    expect(Math.hypot(...offset(70, 50))).toBeLessThan(1);
    expect(offset(50, 0)).toEqual([0, 0]);
  }
  model.move(target, 0, 0, 1); frames(45);
  const forward = light[3]; expect(forward).toBeGreaterThan(0.5);
  model.move(target, 0, 0, -1); frames(45);
  expect(light[3]).toBeLessThan(-0.1); expect(-light[3]).toBeLessThan(forward * 0.5);
  model.release(target, 0, 0); frames(150);
  expect(offset(30, 50)).toEqual([0, 0]); expect(light[3]).toBe(0);
});

test.each(['knit', 'woven'] as const)('a %s bodice carries a chest-level pull to the neck and stays attached at the shoulders and hem', material => {
  const { model, target, frames, offset } = surface(material);
  expect(model.hold(target, 50, 50)).toBe(true);
  for (const [dx, dy, depth] of [[90, 0, 0], [0, 90, 0], [0, 0, 1]]) {
    model.move(target, dx, dy, depth); frames(50);
    expect(offset(50, 10)[1]).toBeGreaterThan(material === 'knit' ? 8 : 6);
    expect(offset(0, 0)).toEqual([0, 0]); expect(offset(100, 0)).toEqual([0, 0]);
    expect(offset(50, 100)).toEqual([0, 0]);
  }
  model.release(target, 0, 0); frames(150);
  expect(offset(50, 10)).toEqual([0, 0]);
  const uncut = surface(material, false);
  uncut.model.hold(uncut.target, 50, 50); uncut.model.move(uncut.target, 25, 35, 1); uncut.frames(50);
  expect(uncut.offset(50, 10)).toEqual([0, 0]);
});

test('a stocking pulls across the leg, a skirt lifts from its hem, and its waistband stays attached', () => {
  for (const material of ['stocking', 'pleat'] as const) {
    const { model, target, frames, offset } = surface(material);
    expect(model.hold(target, 50, 90)).toBe(true);
    model.move(target, 0, -150, 0); frames(60);
    const lift = -offset(50, 90)[1];
    model.move(target, 150, 0, 0); frames(60);
    const stretch = offset(50, 90)[0];
    if (material === 'pleat') expect(lift).toBeGreaterThan(1.7 * stretch);
    else expect(stretch).toBeGreaterThan(2 * lift);
    expect(offset(50, 0)).toEqual([0, 0]);
    model.release(target, 0, 0); frames(150);
    expect(offset(50, 90)).toEqual([0, 0]);
  }
  const limited = surface('pleat', false);
  limited.model.hold(limited.target, 50, 90);
  limited.model.move(limited.target, 0, -150, 0); limited.frames(60);
  expect(-limited.offset(50, 90)[1]).toBeCloseTo(5);
});
