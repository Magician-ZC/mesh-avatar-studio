import { afterEach, expect, test, vi } from 'vitest';
import { openAvatar, packAvatar, validateAvatarFiles } from 'mesh-avatar';
import { OPEN_EYE_FADE, openEyeAlpha, openEyeLayerName } from '../src/engine/open-eyes.js';
import { sampleFiles } from './avatar-fixture';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

test('an open-eye layer is fully shown only while the eye is as drawn', () => {
  expect(openEyeAlpha(1, 0)).toBe(1);
  for (const open of [1 - OPEN_EYE_FADE, 0.5, 0, 1 + OPEN_EYE_FADE, 1.25]) expect(openEyeAlpha(open, 0)).toBe(0);
  expect(openEyeAlpha(1, OPEN_EYE_FADE)).toBe(0);
  expect(openEyeAlpha(1 - OPEN_EYE_FADE / 2, 0)).toBeCloseTo(0.5);
  // it fades smoothly and monotonically as the lid starts to move
  const steps = Array.from({ length: 21 }, (_, i) => openEyeAlpha(1 - (i / 20) * OPEN_EYE_FADE, 0));
  for (let i = 1; i < steps.length; i++) expect(steps[i]).toBeLessThanOrEqual(steps[i - 1]);
  expect(Math.max(...steps.slice(1).map((v, i) => steps[i] - v))).toBeLessThan(0.1);
});

test('with drawn eye sprites the open layer stays until a sprite covers the eye', () => {
  // the layered lids of an eye with sprites never move, so the openness does not matter
  expect(openEyeAlpha(0.5, 0, true, false)).toBe(1);
  expect(openEyeAlpha(1, 0, true, true)).toBe(0);
  expect(openEyeAlpha(1, OPEN_EYE_FADE, true, false)).toBe(0);
  // an expression sprite covering an eye without blink sprites hides it as well
  expect(openEyeAlpha(1, 0, false, true)).toBe(0);
});

test('open-eye layers are optional data that validate and travel with the package', async () => {
  const files = sampleFiles();
  const meta = JSON.parse(new TextDecoder().decode(files['built/layers.json']));
  expect(validateAvatarFiles(files)).toEqual([]);
  for (const eye of [0, 1]) {
    const name = openEyeLayerName(eye);
    expect(meta.layers[name]).toBeUndefined();
    meta.layers[name] = meta.layers[`eye${eye}_ball`];
    files[`built/${name}.png`] = files[`built/eye${eye}_ball.png`];
  }
  files['built/layers.json'] = new TextEncoder().encode(JSON.stringify(meta));
  expect(validateAvatarFiles(files)).toEqual([]);
  vi.stubGlobal('URL', Object.assign(class extends URL {}, { createObjectURL: () => 'blob:open-eye', revokeObjectURL: () => {} }));
  const avatar = await openAvatar(packAvatar(files));
  expect(Object.keys(avatar.assets)).toEqual(expect.arrayContaining(['eye0_open.png', 'eye1_open.png']));
});
