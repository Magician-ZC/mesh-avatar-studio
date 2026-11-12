import { expect, test } from '@playwright/test';
import { samplePresent, sampleSkipReason } from './sample';

// View rotation (ER6) in a real browser: the sample turned by -90 degrees through rig.view.rotation shows exactly the
// upright drawing turned by -90 degrees, in every pose (head, body and the hair physics run unchanged in source pixels),
// and toImagePoint / toClientPoint follow the turned drawing.

const POSES: [string, Record<string, number>, number][] = [
  ['rest', {}, 0],
  ['turn-left', { angleX: -30 }, 1],
  ['tilt-right', { angleZ: 15 }, 1],
  ['body-left', { bodyAngleZ: -8, breath: 1 }, 1],
  ['mouth-open', { mouthOpen: 1, eyeLOpen: 0.4 }, 1],
];

test('a rotated view draws the upright avatar turned, pose for pose', async ({ page }) => {
  test.skip(!samplePresent, sampleSkipReason);
  await page.goto('/e2e/fixtures/embed.html');
  const result = await page.evaluate(async poses => {
    const { createMeshAvatar, PARAMS } = await import('/packages/runtime/src/index.ts' as string) as typeof import('mesh-avatar');
    const rig = await (await fetch('/miko-qipao/rig.json')).json();
    const n = rig.image.width;
    document.body.replaceChildren();
    document.body.style.margin = '0';
    const defaults = Object.fromEntries(PARAMS.map(p => [p.id, p.def]));
    async function frames(view: Record<string, number>) {
      const canvas = document.createElement('canvas');
      canvas.style.cssText = `display: block; width: ${n}px; height: ${n}px;`;
      document.body.replaceChildren(canvas);
      const out: Uint8ClampedArray[] = [];
      for (const [, parameters, seconds] of poses) {
        const avatar = await createMeshAvatar(canvas, { rig: { ...rig, view: { ...rig.view, ...view } }, assetsBase: '/miko-qipao/built', manual: true, padTop: 0, padSide: 0 });
        avatar.setAutoIdle(false); avatar.setAutoMotion(false);
        avatar.setParameters({ ...defaults, ...parameters });
        avatar.advance(seconds, 60);
        const board = document.createElement('canvas');
        board.width = n; board.height = n;
        const context = board.getContext('2d', { willReadFrequently: true })!;
        context.drawImage(canvas, 0, 0);
        out.push(context.getImageData(0, 0, n, n).data);
        avatar.destroy();
      }
      return out;
    }
    const upright = await frames({});
    const turned = await frames({ rotation: -Math.PI / 2 });
    // turning the upright frame by -90 degrees moves pixel (x, y) to (y, n - 1 - x)
    return poses.map(([name], k) => {
      const a = upright[k], b = turned[k];
      let over = 0, sum = 0, opaque = 0;
      for (let v = 0; v < n; v++) for (let u = 0; u < n; u++) {
        const i = ((n - 1 - v) * n + u) * 4, j = (v * n + u) * 4;
        // the frame of the turned avatar at (u, v) against the upright frame at (n - 1 - v, u)
        let d = 0;
        for (let c = 0; c < 4; c++) d = Math.max(d, Math.abs(a[(u * n + (n - 1 - v)) * 4 + c] - b[j + c]));
        if (d > 8) over++;
        sum += d;
        if (a[i + 3] > 0) opaque++;
      }
      return { name, over: over / (n * n), mean: sum / (n * n), opaque };
    });
  }, POSES);
  console.log(`View rotation: ${result.map(p => `${p.name} ${(p.over * 100).toFixed(3)}% over 8, mean ${p.mean.toFixed(3)}`).join('; ')}`);
  for (const pose of result) {
    expect(pose.opaque, pose.name).toBeGreaterThan(100_000);
    // only antialiased silhouette pixels may differ (the multisample pattern is not symmetric under a quarter turn)
    expect(pose.over, pose.name).toBeLessThan(0.002);
    expect(pose.mean, pose.name).toBeLessThan(0.2);
  }
});

test('toImagePoint and toClientPoint follow a rotated view in the page', async ({ page }) => {
  test.skip(!samplePresent, sampleSkipReason);
  await page.goto('/e2e/fixtures/embed.html');
  const checks = await page.evaluate(async () => {
    const { createMeshAvatar } = await import('/packages/runtime/src/index.ts' as string) as typeof import('mesh-avatar');
    const rig = await (await fetch('/miko-qipao/rig.json')).json();
    document.body.replaceChildren();
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'position: absolute; left: 40px; top: 25px; width: 700px; height: 420px; border: 4px solid #000; transform: scale(0.8); transform-origin: 0 0;';
    document.body.append(canvas);
    const avatar = await createMeshAvatar(canvas, { rig: { ...rig, view: { ...rig.view, rotation: Math.PI / 2 } }, assetsBase: '/miko-qipao/built', manual: true });
    const before = avatar.toClientPoint(10, 10);
    avatar.advance(0);
    const errors: number[] = [];
    for (const p of [[0, 0], [rig.head.cx, rig.head.cy], [rig.image.width, rig.image.height], [300.25, 900.5]] as [number, number][]) {
      const client = avatar.toClientPoint(...p)!;
      const back = avatar.toImagePoint(...client)!;
      errors.push(Math.hypot(back[0] - p[0], back[1] - p[1]));
    }
    // +90 degrees turns the image's top edge to the right: the head is right of the chest on the page
    const head = avatar.toClientPoint(rig.head.cx, rig.head.cy)!, chest = avatar.toClientPoint(rig.body.chest.cx, rig.body.chest.cy)!;
    avatar.destroy();
    return { before, errors, headRight: head[0] > chest[0] };
  });
  expect(checks.before).toBeNull();
  for (const error of checks.errors) expect(error).toBeLessThan(1e-6);
  expect(checks.headRight).toBe(true);
});
