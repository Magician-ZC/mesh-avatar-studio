// Iris layers (tools/eye/iris.py): an eye with `eye{i}_white` and `eye{i}_iris` is drawn from them, the iris as a
// sprite that moves by the gaze; an eye without both is drawn from its eye ball as before.
import { afterEach, describe, expect, test, vi } from 'vitest';
import fixture from '../../../samples/miko-qipao/rig.json';
import { openAvatar, packAvatar, validateAvatarFiles, validateRig } from 'mesh-avatar';
import { eyePartNames, gazeShift, hasIrisLayers, IRIS_RIM, irisLayerNames, irisRim, isEyeFill } from '../src/engine/iris-layers.js';
import { sampleFiles } from './avatar-fixture';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('iris layers', () => {
  test('an eye uses them only when the build has both', () => {
    expect(irisLayerNames(1)).toEqual({ white: 'eye1_white', iris: 'eye1_iris' });
    const rect = [0, 0, 4, 4];
    expect(hasIrisLayers({ eye0_ball: rect, eye0_white: rect, eye0_iris: rect }, 0)).toBe(true);
    expect(hasIrisLayers({ eye0_ball: rect, eye0_white: rect, eye0_iris: rect }, 1)).toBe(false);
    expect(hasIrisLayers({ eye0_ball: rect, eye0_white: rect }, 0)).toBe(false);
    expect(hasIrisLayers({ eye0_ball: rect, eye0_iris: rect }, 0)).toBe(false);
    for (const nothing of [undefined, null, 'eye0_white eye0_iris', 3]) expect(hasIrisLayers(nothing, 0)).toBe(false);
    // a name inherited from a prototype is not a layer
    expect(hasIrisLayers(Object.create({ eye0_white: rect, eye0_iris: rect }), 0)).toBe(false);
  });

  test('the white and the iris take the place of the eye ball, under the line layers', () => {
    expect(eyePartNames(false)).toEqual(['ball', 'low', 'crease', 'lash']);
    expect(eyePartNames(true)).toEqual(['white', 'iris', 'low', 'crease', 'lash']);
    expect(['ball', 'white', 'iris', 'low', 'crease', 'lash'].map(isEyeFill)).toEqual([true, true, true, false, false, false]);
  });

  test('the iris travels as far as the eye ball warp does at the middle of the opening', () => {
    // the numbers the renderer has always been given: gazeX * travel, -gazeY * travel * 0.6; the second eye 0.85 of it sideways
    for (const [gx, gy, travel] of [[1, 0, 7], [-1, 1, 7], [0.37, -0.81, 15.22], [0, 0, 12]]) {
      const [first, second] = gazeShift(gx, gy, travel);
      expect(first[0]).toBe(gx * travel);
      expect(first[1]).toBe(-gy * travel * 0.6);
      expect(second[0]).toBe(gx * travel * 0.85);
      expect(second[1]).toBe(-gy * travel * 0.6);
    }
    // looking to the right moves the iris to the right; looking up moves it up (y grows downwards)
    expect(gazeShift(1, 1, 10)[0][0]).toBeGreaterThan(0);
    expect(gazeShift(1, 1, 10)[0][1]).toBeLessThan(0);
  });

  test('the eye is drawn a pixel under its line layers, and three under an open-eye layer', () => {
    expect(IRIS_RIM).toEqual([1, 3]);
    expect(irisRim(0)).toBe(1);
    expect(irisRim(1)).toBe(3);
    expect(irisRim(0.5)).toBe(2);
    // out-of-range or missing opacities do not push the clip further
    for (const [value, rim] of [[undefined, 1], [Number.NaN, 1], [-2, 1], [7, 3]] as const) expect(irisRim(value as number)).toBe(rim);
  });

  test('they are optional data that validate and travel with the package', async () => {
    const files = sampleFiles();
    const meta = JSON.parse(new TextDecoder().decode(files['built/layers.json']));
    expect(validateAvatarFiles(files)).toEqual([]);
    expect(hasIrisLayers(meta.layers, 0)).toBe(false);
    for (const eye of [0, 1]) for (const name of Object.values(irisLayerNames(eye))) {
      meta.layers[name] = meta.layers[`eye${eye}_ball`];
      files[`built/${name}.png`] = files[`built/eye${eye}_ball.png`];
    }
    files['built/layers.json'] = new TextEncoder().encode(JSON.stringify(meta));
    expect(validateAvatarFiles(files)).toEqual([]);
    expect(hasIrisLayers(meta.layers, 0) && hasIrisLayers(meta.layers, 1)).toBe(true);
    vi.stubGlobal('URL', Object.assign(class extends URL {}, { createObjectURL: () => 'blob:iris', revokeObjectURL: () => {} }));
    const avatar = await openAvatar(packAvatar(files));
    expect(Object.keys(avatar.assets)).toEqual(expect.arrayContaining(['eye0_white.png', 'eye0_iris.png', 'eye1_white.png', 'eye1_iris.png', 'eye0_ball.png']));
    // a listed layer without its picture is a broken package, as for every layer
    delete files['built/eye1_iris.png'];
    expect(validateAvatarFiles(files).join('\n')).toContain('eye1_iris');
  });
});

describe('rig.eyes[i].iris', () => {
  const rigWith = (iris: unknown, eye = 0) => {
    const rig = structuredClone(fixture) as unknown as { eyes: Record<string, unknown>[] };
    rig.eyes[eye].iris = iris;
    return rig;
  };

  test('is optional, in a finished rig and in a draft', () => {
    expect(validateRig(fixture)).toEqual([]);
    expect(validateRig(rigWith({ cx: 600, cy: 420, rx: 20, ry: 24 }))).toEqual([]);
    expect(validateRig(rigWith({ cx: 600, cy: 420, rx: 20, ry: 24 }, 1), { draft: true })).toEqual([]);
  });

  test.each([
    [{ cx: 600, cy: 420, rx: 0, ry: 24 }, 'rig.eyes[0].iris.rx'],
    [{ cx: 600, cy: 420, rx: 20 }, 'rig.eyes[0].iris.ry'],
    [{ cx: '600', cy: 420, rx: 20, ry: 24 }, 'rig.eyes[0].iris.cx'],
    [[600, 420, 20, 24], 'rig.eyes[0].iris'],
  ])('must be an ellipse with positive radii: %j', (iris, path) => {
    expect(validateRig(rigWith(iris)).join('\n')).toContain(path);
  });
});
