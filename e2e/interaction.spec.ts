import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';
import { samplePresent, sampleSkipReason } from './sample';

// Pointer interaction (ER1) in a real browser: toImagePoint must name the source pixel that is actually displayed under
// a viewport point, through the canvas position, border, CSS scaling, device pixel ratio, padding and both fits.

const BACKGROUND = [255, 0, 255];

interface Layout { fit: 'contain' | 'cover'; padTop: number; padSide: number; width: number; height: number; css: string }
const LAYOUTS: Layout[] = [
  { fit: 'contain', padTop: 0, padSide: 0, width: 600, height: 800, css: '' },
  { fit: 'cover', padTop: 0.1, padSide: 0.05, width: 520, height: 330, css: 'border: 5px solid #000;' },
  { fit: 'contain', padTop: -0.07, padSide: 0.1, width: 900, height: 640, css: 'transform: scale(0.55); transform-origin: 0 0;' },
  { fit: 'cover', padTop: 0, padSide: 0, width: 333, height: 577, css: 'border: 3px solid #000; transform: scale(1.25); transform-origin: 0 0;' },
];

async function blankPage(page: Page) {
  await page.goto('/e2e/fixtures/embed.html');
  await page.evaluate(color => {
    document.body.replaceChildren();
    document.documentElement.style.background = document.body.style.background = `rgb(${color.join(',')})`;
    document.body.style.margin = '0';
  }, BACKGROUND);
}

/** Draws the rest pose in `layout`; returns viewport sample points, their source pixels and whether the source is opaque there. */
function drawAndSample(page: Page, layout: Layout) {
  return page.evaluate(async layout => {
    const { createMeshAvatar } = await import('/packages/runtime/src/index.ts' as string) as typeof import('mesh-avatar');
    const rig = await (await fetch('/miko-qipao/rig.json')).json();
    const source = new Image();
    source.src = '/miko-qipao/source.png';
    await source.decode();
    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = source.width; sourceCanvas.height = source.height;
    const sourceContext = sourceCanvas.getContext('2d')!;
    sourceContext.drawImage(source, 0, 0);
    const alpha = sourceContext.getImageData(0, 0, source.width, source.height).data;
    const alphaAt = (x: number, y: number) => x < 0 || y < 0 || x >= source.width || y >= source.height ? 0
      : alpha[(Math.floor(y) * source.width + Math.floor(x)) * 4 + 3];
    // opaque or transparent across a 12 px neighbourhood, null near an edge: mipmapped textures drawn at half size and
    // a canvas resampled by CSS blur edges by several source pixels
    const offsets = [-12, -6, 0, 6, 12];
    const classify = (x: number, y: number) => {
      const values = offsets.flatMap(dx => offsets.map(dy => alphaAt(x + dx, y + dy)));
      return values.every(v => v >= 250) ? true : values.every(v => v <= 5) ? false : null;
    };

    document.body.replaceChildren();
    const canvas = document.createElement('canvas');
    canvas.style.cssText = `position: absolute; left: 31px; top: 17px; width: ${layout.width}px; height: ${layout.height}px; ${layout.css}`;
    document.body.append(canvas);
    const avatar = await createMeshAvatar(canvas, { rig, assetsBase: '/miko-qipao/built', manual: true, fit: layout.fit, padTop: layout.padTop, padSide: layout.padSide });
    const beforeFirstFrame = avatar.toImagePoint(100, 100);
    avatar.setAutoIdle(false); avatar.setAutoMotion(false);
    avatar.advance(0);
    const box = canvas.getBoundingClientRect();
    const inner = { left: box.left + canvas.clientLeft * box.width / canvas.offsetWidth, top: box.top + canvas.clientTop * box.height / canvas.offsetHeight,
      width: canvas.clientWidth * box.width / canvas.offsetWidth, height: canvas.clientHeight * box.height / canvas.offsetHeight };
    const samples: { x: number; y: number; opaque: boolean }[] = [];
    for (let i = 0; i < 48; i++) for (let j = 0; j < 48; j++) {
      const x = inner.left + (i + 0.5) / 48 * inner.width, y = inner.top + (j + 0.5) / 48 * inner.height;
      const p = avatar.toImagePoint(x, y);
      if (!p) throw new Error('no mapping after the first frame');
      const opaque = classify(p[0], p[1]);
      if (opaque !== null) samples.push({ x, y, opaque });
    }
    Reflect.set(window, 'interactionAvatar', avatar);
    return { beforeFirstFrame, samples };
  }, layout);
}

/** Destroys the avatar left by drawAndSample; returns toImagePoint afterwards. */
function destroyAvatar(page: Page) {
  return page.evaluate(() => {
    const avatar = Reflect.get(window, 'interactionAvatar') as import('mesh-avatar').MeshAvatar;
    avatar.destroy();
    return avatar.toImagePoint(100, 100);
  });
}

for (const deviceScaleFactor of [1, 2]) test.describe(`device pixel ratio ${deviceScaleFactor}`, () => {
  test.use({ deviceScaleFactor });
  test('toImagePoint names the displayed source pixel for both fits, padding, border and CSS scale', async ({ page }) => {
    test.skip(!samplePresent, sampleSkipReason);
    await blankPage(page);
    for (const layout of LAYOUTS) {
      const { beforeFirstFrame, samples } = await drawAndSample(page, layout);
      expect(beforeFirstFrame).toBeNull();
      expect(samples.filter(s => s.opaque).length).toBeGreaterThan(200);
      expect(samples.filter(s => !s.opaque).length).toBeGreaterThan(layout.fit === 'cover' ? 50 : 200);
      const shot = PNG.sync.read(await page.screenshot());
      const scale = shot.width / page.viewportSize()!.width;
      const wrong = samples.filter(({ x, y, opaque }) => {
        const i = (Math.floor(y * scale) * shot.width + Math.floor(x * scale)) * 4;
        const background = BACKGROUND.every((v, c) => Math.abs(shot.data[i + c] - v) <= 8);
        return background === opaque;
      });
      expect(wrong, JSON.stringify(layout)).toEqual([]);
      expect(await destroyAvatar(page)).toBeNull();
    }
  });
});

test('an impulse sways a hair strand, which comes back to the undisturbed frame', async ({ page }) => {
  test.skip(!samplePresent, sampleSkipReason);
  await blankPage(page);
  const result = await page.evaluate(async () => {
    const { createMeshAvatar } = await import('/packages/runtime/src/index.ts' as string) as typeof import('mesh-avatar');
    const rig = await (await fetch('/miko-qipao/rig.json')).json();
    const seeded = (seed: number) => () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const random = Math.random;
    const frames = async (push: boolean) => {
      Math.random = seeded(7);
      const canvas = document.createElement('canvas');
      canvas.style.cssText = 'width: 600px; height: 600px;';
      document.body.replaceChildren(canvas);
      const avatar = await createMeshAvatar(canvas, { rig, assetsBase: '/miko-qipao/built', manual: true, padTop: 0, padSide: 0 });
      avatar.setAutoIdle(false); avatar.setAutoMotion(false);
      const read = () => {
        const copy = document.createElement('canvas');
        copy.width = canvas.width; copy.height = canvas.height;
        const context = copy.getContext('2d')!;
        context.drawImage(canvas, 0, 0);
        return context.getImageData(0, 0, copy.width, copy.height).data;
      };
      avatar.advance(0.5);
      if (push) avatar.impulse({ kind: 'strand', name: 'sideL' }, 1600, 0);
      avatar.advance(0.15);
      const swaying = read();
      avatar.advance(3);
      const settled = read();
      avatar.destroy();
      return { swaying, settled };
    };
    try {
      const still = await frames(false), pushed = await frames(true);
      const differing = (a: Uint8ClampedArray, b: Uint8ClampedArray, tolerance: number) => {
        let count = 0;
        for (let i = 0; i < a.length; i += 4) if ([0, 1, 2, 3].some(c => Math.abs(a[i + c] - b[i + c]) > tolerance)) count++;
        return count;
      };
      return { swaying: differing(still.swaying, pushed.swaying, 8), settled: differing(still.settled, pushed.settled, 2) };
    } finally { Math.random = random; }
  });
  expect(result.swaying).toBeGreaterThan(500);
  expect(result.settled).toBe(0);
});
