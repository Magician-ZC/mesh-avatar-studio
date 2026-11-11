import { afterEach, expect, test, vi } from 'vitest';
import { createMeshAvatar, parseRig, type ImpulseTarget, type MeshAvatar } from 'mesh-avatar';
import fixture from '../../../samples/miko-qipao/rig.json';

// The public toImagePoint/impulse wiring (ER1) without drawing: only the browser's asset/canvas boundary is stubbed.
const drawn = vi.hoisted(() => ({ toImage: (_x: number, _y: number): [number, number] | null => null }));
vi.mock('../src/engine/renderer.js', () => ({ createRenderer: () => ({
  buildGrid: () => ({ rest: [], pos: [] }),
  Renderer: class {
    addLayer(_name: string, _image: unknown, mesh: unknown) { return { mesh }; }
    draw() {}
    destroy() {}
    toImage(x: number, y: number) { return drawn.toImage(x, y); }
  },
}) }));
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function stubBrowser() {
  vi.stubGlobal('Image', class {
    width = 1; height = 1; onload = () => {};
    set src(_value: string) { queueMicrotask(() => this.onload()); }
  });
  vi.stubGlobal('document', { createElement: () => ({ getContext: () => ({
    drawImage() {}, getImageData: () => ({ data: new Uint8Array(4) }),
  }) }) });
  vi.stubGlobal('fetch', vi.fn(async (url: string) => url.endsWith('layers.json')
    ? new Response(JSON.stringify({ layers: {} })) : new Response('', { status: 404 })));
  vi.stubGlobal('cancelAnimationFrame', vi.fn());
  vi.spyOn(console, 'warn').mockImplementation(() => {});
}

const canvas = {
  getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 100 }),
  offsetWidth: 200, offsetHeight: 100, clientLeft: 0, clientTop: 0,
} as unknown as HTMLCanvasElement;

async function avatar(): Promise<MeshAvatar> {
  const created = await createMeshAvatar(canvas, { rig: parseRig(fixture), assetsBase: 'https://example.test/interaction/', manual: true });
  created.setAutoIdle(false); created.setAutoMotion(false);
  return created;
}

test('toImagePoint offsets by the canvas box, and is null before the first frame and after destroy', async () => {
  stubBrowser();
  const a = await avatar();
  drawn.toImage = () => null;
  expect(a.toImagePoint(15, 25)).toBeNull();
  drawn.toImage = (x, y) => [x * 2, y * 2];
  expect(a.toImagePoint(15, 25)).toEqual([10, 10]);
  a.destroy();
  expect(a.toImagePoint(15, 25)).toBeNull();
});

test('impulse moves the head and body springs, which settle; unknown targets and a destroyed avatar are ignored', async () => {
  stubBrowser();
  vi.spyOn(Math, 'random').mockReturnValue(0.5);
  const pushed = await avatar(), still = await avatar();
  try {
    pushed.impulse({ kind: 'head' }, 150, 0);
    pushed.impulse({ kind: 'body' }, -160, 0);
    for (const target of [{ kind: 'limb', name: 'armL' }, { kind: 'strand', name: 'missing' }, { kind: 'cloth', name: 'skirt' }, { kind: 'tail' }] as unknown as ImpulseTarget[])
      expect(() => pushed.impulse(target, 100, 100)).not.toThrow();
    const difference = (id: string) => Math.abs(pushed.getParameters()[id] - still.getParameters()[id]);
    let turned = 0, leaned = 0;
    for (let f = 0; f < 30; f++) {
      pushed.advanceParameters(1 / 60); still.advanceParameters(1 / 60);
      turned = Math.max(turned, difference('angleX')); leaned = Math.max(leaned, difference('bodyAngleZ'));
    }
    expect(turned).toBeGreaterThan(1);
    expect(leaned).toBeGreaterThan(0.5);
    for (let f = 0; f < 150; f++) { pushed.advanceParameters(1 / 60); still.advanceParameters(1 / 60); }
    for (const id of ['angleX', 'angleZ', 'bodyAngleX', 'bodyAngleZ']) expect(difference(id)).toBeLessThan(1e-3);
  } finally { pushed.destroy(); still.destroy(); }
  expect(() => pushed.impulse({ kind: 'head' }, 100, 0)).not.toThrow();
});
