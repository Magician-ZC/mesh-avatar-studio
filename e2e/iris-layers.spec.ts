import { expect, test } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

// Iris layers end to end: build-layers.py cuts the eye white and the iris of an eye whose iris is taller than its
// opening, and in the browser the resting avatar is the source, while for a gaze the iris moves as it is (every row
// by the same distance), the white it leaves is white, and no skin shows inside the opening. Drawn at another size
// than the picture's (textures filtered), the resting eye is the same as without iris layers: no seam at the lids.

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const W = 360, H = 300;
const CENTRES: [number, number][] = [[120, 150], [240, 150]];
const HALF_W = 34, HALF_H = 13, RADIUS = 17, TRAVEL = 8;
const SKIN = [236, 200, 180], WHITE = [246, 244, 240], RING = [44, 28, 22], IRIS = [128, 86, 62], PUPIL = [20, 12, 10], LASH = [40, 26, 22];

function colourAt(x: number, y: number): number[] {
  for (const [cx, cy] of CENTRES) {
    const reach = HALF_H * Math.max(0, 1 - ((x - cx) / HALF_W) ** 2);
    if (Math.abs(x - cx) < HALF_W && Math.abs(y - cy) < reach) {
      const r = Math.hypot(x - cx, y - cy);
      if (Math.hypot(x - cx - 5, y - cy + 5) < 2.5) return [255, 255, 255];
      if (r < 6) return PUPIL;
      if (r < RADIUS - 3) return IRIS;
      if (r < RADIUS) return RING;
      return y < cy - reach + 2.5 ? [214, 204, 200] : WHITE; // the lid's shadow on the eye
    }
    if (Math.abs(x - cx) < HALF_W + 2 && y <= cy - reach && y > cy - reach - 3) return LASH;
  }
  if (Math.abs(x - 180) < 22 && Math.abs(y - 232) < 1.5) return [170, 80, 80];
  return SKIN;
}

function drawSource(): PNG {
  const png = new PNG({ width: W, height: H });
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const sum = [0, 0, 0];
    // four samples a side, so edges are soft as in a real picture
    for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) colourAt(x + (i + 0.5) / 4, y + (j + 0.5) / 4).forEach((v, c) => { sum[c] += v; });
    png.data.set([...sum.map(v => Math.round(v / 16)), 255], (y * W + x) * 4);
  }
  return png;
}

function rig(withIris = true) {
  const base = JSON.parse(readFileSync(join(ROOT, 'samples/miko-qipao/rig.json'), 'utf8'));
  for (const key of ['hand', 'buns', 'accessories', 'strands']) delete base[key];
  const eyes = CENTRES.map(([cx, cy]) => {
    const xs = Array.from({ length: 11 }, (_, k) => cx - HALF_W + (2 * HALF_W * k) / 10);
    const arc = xs.map(x => HALF_H * (1 - ((x - cx) / HALF_W) ** 2));
    return {
      opening: [...xs.map((x, k) => [x, cy - arc[k]]), ...xs.slice(1, -1).reverse().map((x, k) => [x, cy + arc.slice(1, -1).reverse()[k]])],
      roi: [[cx - HALF_W - 10, cy - HALF_H - 12], [cx + HALF_W + 10, cy - HALF_H - 12], [cx + HALF_W + 10, cy + HALF_H + 8], [cx - HALF_W - 10, cy + HALF_H + 8]],
      ...(withIris ? { iris: { cx, cy, rx: RADIUS, ry: RADIUS } } : {}),
    };
  });
  return {
    ...base,
    image: { width: W, height: H },
    head: { ...base.head, cx: 180, cy: 150, rx: 150, ry: 140, pivotX: 180, pivotY: 280, weightBand: [270, 295], turnBand: [272, 296] },
    body: { ...base.body, pivotX: 180, pivotY: 600, breathBand: [285, 299], rollBand: [280, 600], chest: { cx: 180, cy: 290, rx: 60, ry: 8 },
      shoulders: [{ cx: 60, cy: 292, rx: 20, ry: 6 }, { cx: 300, cy: 292, rx: 20, ry: 6 }] },
    face: {
      nose: { cx: 180, cy: 195, rx: 16, ry: 16 }, mouth: { cx: 180, cy: 232, rx: 30, ry: 14 },
      eyeA: { cx: 120, cy: 150, rx: 44, ry: 26 }, eyeB: { cx: 240, cy: 150, rx: 44, ry: 26 },
      earR: { cx: 330, cy: 160, rx: 12, ry: 22 }, earL: { cx: 30, cy: 160, rx: 12, ry: 22 },
      brow: { cx: 180, cy: 110, rx: 90, ry: 8, band: [118, 126] }, jaw: { cx: 180, cy: 262, rx: 50, ry: 20, band: [246, 256] },
    },
    eyes,
    mouth: { cx: 180, cy: 232, angle: 0, halfLen: 20, bow: 0, area: { cx: 180, cy: 234, rx: 30, ry: 14, angle: 0 } },
    cheeks: [[110, 200], [250, 200]],
    mesh: { ...base.mesh, baseCell: 8, fine: { x0: 40, x1: 320, y0: 60, y1: 270, cell: 5 } },
    view: { padTop: 0, padSide: 0, irisTravel: TRAVEL },
  };
}

let project = '', plain = '';
function build(withIris: boolean): string {
  const folder = mkdtempSync(join(tmpdir(), 'iris-e2e-'));
  writeFileSync(join(folder, 'source.png'), PNG.sync.write(drawSource()));
  writeFileSync(join(folder, 'rig.draft.json'), JSON.stringify(rig(withIris)));
  const built = spawnSync('uv', ['run', '--with', 'numpy', '--with', 'pillow', '--with', 'opencv-python-headless', 'tools/build-layers.py', folder, '--rig', 'rig.draft.json', '--open-eyes'],
    { cwd: ROOT, encoding: 'utf8' });
  if (built.status !== 0) throw new Error(`build-layers failed: ${built.stderr}`);
  if (built.stderr.includes('iris layers left out')) throw new Error(built.stderr);
  return folder;
}
function assetsOf(folder: string) {
  const layers = JSON.parse(readFileSync(join(folder, 'built/layers.json'), 'utf8'));
  const assets: Record<string, string> = {};
  for (const name of ['layers.json', 'base.png', 'hairmask.png', ...Object.keys(layers.layers).map(n => `${n}.png`)]) {
    assets[name] = `data:${name.endsWith('.png') ? 'image/png' : 'application/json'};base64,${readFileSync(join(folder, 'built', name)).toString('base64')}`;
  }
  return { layers, assets, rig: JSON.parse(readFileSync(join(folder, 'rig.json'), 'utf8')) };
}
test.beforeAll(() => { project = build(true); plain = build(false); });
test.afterAll(() => { for (const folder of [project, plain]) if (folder) rmSync(folder, { recursive: true, force: true }); });

test('an eye with iris layers rests on the source and its iris moves as it is, leaving white', async ({ page }) => {
  const { layers, assets, rig: built } = assetsOf(project);
  expect(Object.keys(layers.layers)).toEqual(expect.arrayContaining(['eye0_white', 'eye0_iris', 'eye1_white', 'eye1_iris', 'eye0_ball', 'eye0_open']));
  const source = `data:image/png;base64,${readFileSync(join(project, 'source.png')).toString('base64')}`;
  await page.goto('/e2e/fixtures/embed.html');
  const result = await page.evaluate(async ({ rig, assets, source, eyes, skin }) => {
    const { createMeshAvatar, PARAMS } = await import('/packages/runtime/src/index.ts' as string) as typeof import('mesh-avatar');
    const w = rig.image.width, h = rig.image.height;
    const read = async (url: string) => {
      const image = new Image(); image.src = url; await image.decode();
      const board = document.createElement('canvas'); board.width = w; board.height = h;
      const context = board.getContext('2d', { willReadFrequently: true })!;
      context.drawImage(image, 0, 0, w, h);
      return context.getImageData(0, 0, w, h).data;
    };
    const original = await read(source);
    document.body.replaceChildren();
    const canvas = document.createElement('canvas');
    canvas.style.cssText = `display: block; width: ${w}px; height: ${h}px;`;
    document.body.append(canvas);
    const defaults = Object.fromEntries(PARAMS.map(p => [p.id, p.def]));
    const luma = (d: Uint8ClampedArray, i: number) => 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    const poses: [string, Record<string, number>][] = [['rest', {}], ['right', { gazeX: 1 }], ['left', { gazeX: -1 }], ['up', { gazeY: 1 }], ['down', { gazeY: -1 }], ['right-down', { gazeX: 1, gazeY: -1 }]];
    // the rows right under the lids, where an iris layer holds the iris's own colour and not the picture's blend with the lash
    const rim = (x: number, y: number) => eyes.some(({ cx, cy, halfW, halfH }) => {
      const reach = halfH * (1 - ((x + 0.5 - cx) / halfW) ** 2);
      return Math.abs(x + 0.5 - cx) < halfW + 1 && Math.abs(y + 0.5 - cy) < reach + 1 && Math.abs(y + 0.5 - cy) > reach - 3;
    });
    const out: Record<string, { restMax: number; rimOver: number; rimMean: number; eyes: { pupil: [number, number]; edges: number[]; skin: number; dark: number[] }[] }> = {};
    for (const [name, values] of poses) {
      const avatar = await createMeshAvatar(canvas, { rig, assets, manual: true, padTop: 0, padSide: 0 });
      avatar.setAutoIdle(false); avatar.setAutoMotion(false);
      avatar.setParameters({ ...defaults, ...values });
      avatar.advance(0);
      const frame = await read(canvas.toDataURL());
      avatar.destroy();
      let restMax = 0, rimOver = 0, rimSum = 0, rimCount = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const diff = Math.max(Math.abs(frame[i] - original[i]), Math.abs(frame[i + 1] - original[i + 1]), Math.abs(frame[i + 2] - original[i + 2]));
        if (rim(x, y)) { rimSum += diff; rimCount++; if (diff > 12) rimOver++; } else restMax = Math.max(restMax, diff);
      }
      const measured = eyes.map(({ cx, cy, halfW, halfH }) => {
        // inside the opening, three pixels away from the lids
        const inside = (x: number, y: number) => Math.abs(x + 0.5 - cx) < halfW - 3 && Math.abs(y + 0.5 - cy) < halfH * (1 - ((x + 0.5 - cx) / halfW) ** 2) - 3;
        let px = 0, py = 0, pn = 0, skinLike = 0;
        const dark: number[] = [];
        for (let y = cy - halfH; y < cy + halfH; y++) for (let x = cx - halfW; x < cx + halfW; x++) {
          if (!inside(x, y)) continue;
          const i = (y * w + x) * 4, l = luma(frame, i);
          if (l < 30) { px += x + 0.5; py += y + 0.5; pn++; }
          if (Math.max(Math.abs(frame[i] - skin[0]), Math.abs(frame[i + 1] - skin[1]), Math.abs(frame[i + 2] - skin[2])) < 14) skinLike++;
          if (l < 150) dark.push(x, y);
        }
        // where the white ends and the iris ring begins on three rows, coming from the left and from the right
        const edges = [-8, 0, 8].flatMap(dy => {
          const y = cy + dy, reach = Math.floor(halfW * Math.sqrt(1 - Math.abs(dy) / halfH)) - 2;
          return [1, -1].map(step => {
            for (let x = cx - step * reach; Math.abs(x - cx) <= reach; x += step) {
              const a = luma(frame, (y * w + x - step) * 4), b = luma(frame, (y * w + x) * 4);
              if (a >= 140 && b < 140) return x - step + step * (a - 140) / (a - b);
            }
            return NaN;
          });
        });
        return { pupil: [px / pn, py / pn] as [number, number], edges, skin: skinLike, dark };
      });
      out[name] = { restMax, rimOver, rimMean: rimSum / rimCount, eyes: measured };
    }
    return out;
  }, { rig: built, assets, source, eyes: CENTRES.map(([cx, cy]) => ({ cx, cy, halfW: HALF_W, halfH: HALF_H })), skin: SKIN });

  const rest = result.rest;
  console.log(`Iris layers: rest max ${rest.restMax} away from the lids, under the lids mean ${rest.rimMean.toFixed(2)} and ${rest.rimOver} px over 12; `
    + Object.entries(result).filter(([name]) => name !== 'rest').map(([name, r]) => `${name} pupil ${r.eyes.map((e, k) => `${(e.pupil[0] - rest.eyes[k].pupil[0]).toFixed(2)},${(e.pupil[1] - rest.eyes[k].pupil[1]).toFixed(2)}`).join(' / ')}`
      + ` edges ${r.eyes.map((e, k) => e.edges.map((x, j) => (x - rest.eyes[k].edges[j]).toFixed(2)).join(',')).join(' / ')} skin ${r.eyes.map(e => e.skin).join('/')}`).join('; '));
  // at rest the avatar is the picture, also along the lids (the lid's share of the pixels there is in the open-eye layer)
  expect(rest.restMax).toBeLessThanOrEqual(2);
  expect(rest.rimMean).toBeLessThan(0.5);
  expect(rest.rimOver).toBeLessThan(12);
  for (const eye of rest.eyes) expect(eye.skin).toBe(0);

  const shift: Record<string, [number, number]> = { right: [1, 0], left: [-1, 0], up: [0, -0.6], down: [0, 0.6], 'right-down': [1, 0.6] };
  for (const [name, [sx, sy]] of Object.entries(shift)) {
    result[name].eyes.forEach((eye, k) => {
      const still = rest.eyes[k], factor = k === 1 ? 0.85 : 1;
      const dx = sx * TRAVEL * factor, dy = sy * TRAVEL;
      // the pupil has moved by the iris travel
      expect(eye.pupil[0] - still.pupil[0], `${name} eye ${k} pupil x`).toBeCloseTo(dx, 0);
      expect(eye.pupil[1] - still.pupil[1], `${name} eye ${k} pupil y`).toBeCloseTo(dy, 0);
      // ... and so has the edge of the iris on every row: the iris moves as it is, it is not sheared
      // (read on the side the iris moves away from; on the other side a lid may cover the edge)
      if (sy === 0) for (const [j, edge] of eye.edges.entries()) {
        if ((j % 2 === 0) !== (sx > 0)) continue;
        expect(Math.abs(edge - still.edges[j] - dx), `${name} eye ${k} edge ${j}: moved ${(edge - still.edges[j]).toFixed(2)} of ${dx}`).toBeLessThan(1);
      }
      // no skin shows inside the opening
      expect(eye.skin, `${name} eye ${k} skin`).toBe(0);
      // nothing of the iris stays where it was: every dark pixel is on the iris where it is now
      const [cx, cy] = CENTRES[k];
      let left = 0;
      for (let i = 0; i < eye.dark.length; i += 2) if (Math.hypot(eye.dark[i] + 0.5 - cx - dx, eye.dark[i + 1] + 0.5 - cy - dy) > RADIUS + 1.5) left++;
      expect(left, `${name} eye ${k} left behind`).toBe(0);
    });
  }
});

// Sizes above 1: there the difference is the layers' alone. Below 1 every layer is read from its own mipmaps, and two
// builds with different layer rectangles differ by a few levels all over the iris.
test('drawn at another size than the picture, the resting eye is the one without iris layers: no seam at the lids', async ({ page }) => {
  const sets = [assetsOf(project), assetsOf(plain)];
  expect(Object.keys(sets[1].layers.layers)).not.toContain('eye0_iris');
  await page.goto('/e2e/fixtures/embed.html');
  const result = await page.evaluate(async ({ sets, eyes, sizes }) => {
    const { createMeshAvatar, PARAMS } = await import('/packages/runtime/src/index.ts' as string) as typeof import('mesh-avatar');
    document.body.replaceChildren();
    const canvas = document.createElement('canvas');
    document.body.append(canvas);
    const defaults = Object.fromEntries(PARAMS.map(p => [p.id, p.def]));
    const out: { size: number; max: number; over: number; mean: number }[] = [];
    for (const size of sizes) {
      const frames: Uint8ClampedArray[] = [];
      let w = 0, h = 0;
      for (const { rig, assets } of sets) {
        canvas.style.cssText = `display: block; width: ${rig.image.width * size}px; height: ${rig.image.height * size}px;`;
        const avatar = await createMeshAvatar(canvas, { rig, assets, manual: true, padTop: 0, padSide: 0 });
        avatar.setAutoIdle(false); avatar.setAutoMotion(false);
        avatar.setParameters(defaults);
        avatar.advance(0);
        const image = new Image(); image.src = canvas.toDataURL(); await image.decode();
        avatar.destroy();
        w = image.width; h = image.height;
        const board = document.createElement('canvas'); board.width = w; board.height = h;
        const context = board.getContext('2d', { willReadFrequently: true })!;
        context.drawImage(image, 0, 0);
        frames.push(context.getImageData(0, 0, w, h).data);
      }
      let max = 0, over = 0, sum = 0, count = 0;
      for (const { cx, cy, halfW, halfH } of eyes) {
        for (let y = Math.floor((cy - halfH - 8) * size); y < (cy + halfH + 8) * size; y++) for (let x = Math.floor((cx - halfW - 8) * size); x < (cx + halfW + 8) * size; x++) {
          const i = (y * w + x) * 4;
          const diff = Math.max(Math.abs(frames[0][i] - frames[1][i]), Math.abs(frames[0][i + 1] - frames[1][i + 1]), Math.abs(frames[0][i + 2] - frames[1][i + 2]));
          max = Math.max(max, diff); sum += diff; count++; if (diff > 24) over++;
        }
      }
      out.push({ size, max, over: over / (size * size), mean: sum / count });
    }
    return out;
  }, { sets: sets.map(({ rig, assets }) => ({ rig, assets })), eyes: CENTRES.map(([cx, cy]) => ({ cx, cy, halfW: HALF_W, halfH: HALF_H })), sizes: [1.25, 1.5, 2] });
  console.log(`Iris layers, filtered: ${result.map(r => `x${r.size} max ${r.max}, over 24: ${r.over.toFixed(1)} source px, mean ${r.mean.toFixed(3)}`).join('; ')}`);
  // The two differ only in a few pixels where the iris's edge meets a lid (the edge is rendered from three layers
  // there). A seam along a lid, as when the base shows through at the hole's edge, would be some hundred source px.
  for (const r of result) {
    expect(r.over, `size ${r.size}`).toBeLessThan(40);
    expect(r.mean, `size ${r.size}`).toBeLessThan(0.6);
  }
});
