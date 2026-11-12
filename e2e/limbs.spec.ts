import { expect, test } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

// Generic limb chains (ER4) end to end: build-layers.py cuts a dangling leg and the bare base, and in the browser the
// resting avatar is the source, while at the swing and bend limits the cut where the leg meets the body never opens.

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const W = 240, H = 320;
const BODY: [number, number, number, number] = [30, 20, 210, 200];
const LEG: [number, number, number, number] = [96, 180, 136, 310];
const JOINTS = [[116, 186], [116, 248], [116, 304]];
const OUTLINE = [[92, 176], [140, 176], [140, 314], [92, 314]];

function drawSource(): PNG {
  const png = new PNG({ width: W, height: H });
  const fill = ([x0, y0, x1, y1]: number[], rgba: number[]) => {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) png.data.set(rgba, (y * W + x) * 4);
  };
  fill(BODY, [190, 205, 228, 255]);
  for (let y = BODY[1]; y < BODY[3]; y += 6) fill([BODY[0], y, BODY[2], y + 2], [150, 170, 210, 255]); // texture
  fill([70, 80, 100, 94], [255, 255, 255, 255]); fill([140, 80, 170, 94], [255, 255, 255, 255]); // eyes
  fill([70, 77, 100, 80], [20, 25, 50, 255]); fill([140, 77, 170, 80], [20, 25, 50, 255]); // lashes
  fill([110, 130, 130, 133], [170, 70, 80, 255]); // mouth
  fill(LEG, [205, 120, 70, 255]);
  for (let y = LEG[1]; y < LEG[3]; y += 10) fill([LEG[0], y, LEG[2], y + 3], [150, 80, 40, 255]); // stripes
  return png;
}

function rig() {
  const base = JSON.parse(readFileSync(join(ROOT, 'samples/miko-qipao/rig.json'), 'utf8'));
  for (const key of ['hand', 'buns', 'accessories', 'strands']) delete base[key];
  return {
    ...base,
    image: { width: W, height: H },
    head: { ...base.head, cx: 120, cy: 90, rx: 80, ry: 70, pivotX: 120, pivotY: 160, weightBand: [140, 170], turnBand: [145, 175] },
    body: { ...base.body, pivotX: 120, pivotY: 400, breathBand: [150, 200], rollBand: [120, 300], chest: { cx: 120, cy: 170, rx: 60, ry: 30 },
      shoulders: [{ cx: 60, cy: 150, rx: 20, ry: 15 }, { cx: 180, cy: 150, rx: 20, ry: 15 }] },
    eyes: [{ opening: [[70, 80], [100, 80], [100, 94], [70, 94]], roi: [[64, 72], [106, 72], [106, 100], [64, 100]] },
      { opening: [[140, 80], [170, 80], [170, 94], [140, 94]], roi: [[134, 72], [176, 72], [176, 100], [134, 100]] }],
    mouth: { cx: 120, cy: 131, angle: 0, halfLen: 10, bow: 0, area: { cx: 120, cy: 131, rx: 13, ry: 6, angle: 0 } },
    cheeks: [[85, 110], [155, 110]],
    mesh: { ...base.mesh, baseCell: 6, fine: { x0: 40, x1: 200, y0: 30, y1: 160, cell: 4 } },
    view: { padTop: 0, padSide: 0 },
    limbs: [{ name: 'leg_l', kind: 'leg', joints: JOINTS, outline: OUTLINE }],
  };
}

let project = '';
test.beforeAll(() => {
  project = mkdtempSync(join(tmpdir(), 'limbs-e2e-'));
  writeFileSync(join(project, 'source.png'), PNG.sync.write(drawSource()));
  writeFileSync(join(project, 'rig.draft.json'), JSON.stringify(rig()));
  const built = spawnSync('uv', ['run', '--with', 'numpy', '--with', 'pillow', '--with', 'opencv-python-headless', 'tools/build-layers.py', project, '--rig', 'rig.draft.json', '--open-eyes'],
    { cwd: ROOT, encoding: 'utf8' });
  if (built.status !== 0) throw new Error(`build-layers failed: ${built.stderr}`);
});
test.afterAll(() => { if (project) rmSync(project, { recursive: true, force: true }); });

test('a limb avatar rests exactly on the source and its root never opens at the limits', async ({ page }) => {
  const layers = JSON.parse(readFileSync(join(project, 'built/layers.json'), 'utf8'));
  expect(layers.bare).toBe('base_bare');
  const assets: Record<string, string> = {};
  for (const name of ['layers.json', 'base.png', 'hairmask.png', 'base_bare.png', ...Object.keys(layers.layers).map(n => `${n}.png`)]) {
    assets[name] = `data:${name.endsWith('.png') ? 'image/png' : 'application/json'};base64,${readFileSync(join(project, 'built', name)).toString('base64')}`;
  }
  const built = JSON.parse(readFileSync(join(project, 'rig.json'), 'utf8'));
  const source = `data:image/png;base64,${readFileSync(join(project, 'source.png')).toString('base64')}`;
  await page.goto('/e2e/fixtures/embed.html');
  const result = await page.evaluate(async ({ rig, assets, source }) => {
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
    const defaults = { ...Object.fromEntries(PARAMS.map(p => [p.id, p.def])), 'limb:leg_l:swing': 0, 'limb:leg_l:bend': 0 };
    const poses: [string, Record<string, number>][] = [['rest', {}], ['swing+', { 'limb:leg_l:swing': 1 }], ['swing-', { 'limb:leg_l:swing': -1 }],
      ['bend+', { 'limb:leg_l:bend': 1 }], ['bend-', { 'limb:leg_l:bend': -1 }], ['both', { 'limb:leg_l:swing': 1, 'limb:leg_l:bend': 1 }], ['both-', { 'limb:leg_l:swing': -1, 'limb:leg_l:bend': -1 }]];
    const out: Record<string, { restMax: number; restOver: number; holes: number; moved: number }> = {};
    for (const [name, values] of poses) {
      const avatar = await createMeshAvatar(canvas, { rig, assets, manual: true, padTop: 0, padSide: 0 });
      avatar.setAutoIdle(false); avatar.setAutoMotion(false);
      avatar.setParameters({ ...defaults, ...values });
      avatar.advance(0);
      const frame = await read(canvas.toDataURL());
      avatar.destroy();
      let restMax = 0, restOver = 0, holes = 0, moved = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        // seen over grey, as the build pipeline's quality checks compare
        const seen = (d: Uint8ClampedArray, c: number) => d[i + c] * d[i + 3] / 255 + 128 * (1 - d[i + 3] / 255);
        const diff = Math.max(...[0, 1, 2].map(c => Math.abs(seen(frame, c) - seen(original, c))));
        if (name === 'rest') { restMax = Math.max(restMax, diff); if (diff > 8) restOver++; }
        else if (diff > 8) moved++;
        // a hole: opaque in the source, inside the body block (where the leg is attached), transparent now
        const inBody = x >= 30 && x < 210 && y >= 20 && y < 200;
        if (inBody && original[i + 3] === 255 && frame[i + 3] < 128) holes++;
      }
      out[name] = { restMax, restOver, holes, moved };
    }
    return out;
  }, { rig: built, assets, source });
  console.log(`Limbs: ${Object.entries(result).map(([name, r]) => `${name} holes ${r.holes}${name === 'rest' ? `, rest max ${r.restMax.toFixed(1)}, over 8 ${r.restOver}` : `, moved ${r.moved}`}`).join('; ')}`);
  expect(result.rest.restMax).toBeLessThanOrEqual(2);
  expect(result.rest.restOver).toBe(0);
  for (const [name, r] of Object.entries(result)) {
    expect(r.holes, name).toBe(0);
    if (name !== 'rest') expect(r.moved, name).toBeGreaterThan(50);
  }
});
