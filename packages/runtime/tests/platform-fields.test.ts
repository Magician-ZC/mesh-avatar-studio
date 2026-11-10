import { expect, test } from 'vitest';
import { openAvatar, packAvatar, parseManifest, parseRig, unpackAvatar, validateAvatarFiles, validateRig } from 'mesh-avatar';
import { packProjectAvatar } from '../src/format/project-archive';
import { sampleFiles } from './avatar-fixture';

const manifest = { format: 'mesh-avatar', version: 1, name: 'My avatar' };
const platform = { avatarId: 'av_1', ownerId: 'user_1', region: 'cn', issuedAt: '2026-11-10T12:00:00.000Z', style: 'anime', aiGenerated: true };
const finalCapabilities = {
  style: 'photoreal',
  face: { blink: false, mouth: true, gaze: true, brows: true },
  head: { turn: true, roll: false },
  body: { breath: true, lean: true, sway: false },
  legs: false, arms: false,
  limbs: [{ name: 'tail', kind: 'prop', free: true }],
  strands: ['front_l'], cloth: [], zones: ['head', 'ribbon'],
  motions: [{ id: 'nod', idle: false }],
};
const decode = (bytes: Uint8Array) => JSON.parse(new TextDecoder().decode(bytes));

test('manifests with and without platform fields round trip; QA-disabled capabilities survive', () => {
  expect(parseManifest(manifest)).toEqual(manifest);
  const parsed = parseManifest({ ...manifest, platform, finalCapabilities });
  expect(parsed).toEqual({ ...manifest, platform, finalCapabilities });
  expect(parseManifest(JSON.parse(JSON.stringify(parsed)))).toEqual(parsed);
  expect(parsed.finalCapabilities?.face?.blink).toBe(false);
  expect(parsed.platform).not.toBe(platform);
});

test('unknown capability and platform keys are ignored; partial capability lists stay partial', () => {
  const parsed = parseManifest({
    ...manifest,
    platform: { ...platform, extra: 1 },
    finalCapabilities: { face: { blink: false, wink: true }, future: { x: 1 } },
  });
  expect(parsed.platform).toEqual(platform);
  expect(parsed.finalCapabilities).toEqual({ face: { blink: false } });
});

test.each([
  [{ platform: null }, 'avatar.json.platform: expected an object'],
  [{ platform: { ...platform, ownerId: '' } }, 'avatar.json.platform.ownerId: expected a non-empty string'],
  [{ platform: { ...platform, aiGenerated: 'yes' } }, 'avatar.json.platform.aiGenerated: expected a boolean'],
  [{ platform: { ...platform, issuedAt: 'yesterday' } }, 'avatar.json.platform.issuedAt: expected an ISO 8601 date-time'],
  [{ finalCapabilities: [] }, 'avatar.json.finalCapabilities: expected an object'],
  [{ finalCapabilities: { face: { blink: 0 } } }, 'avatar.json.finalCapabilities.face.blink: expected a boolean'],
  [{ finalCapabilities: { limbs: [{ name: 'tail', kind: 'prop' }] } }, 'avatar.json.finalCapabilities.limbs[0].free: expected a boolean'],
  [{ finalCapabilities: { zones: ['head', 3] } }, 'avatar.json.finalCapabilities.zones[1]: expected a non-empty string'],
  [{ finalCapabilities: { motions: 'nod' } }, 'avatar.json.finalCapabilities.motions: expected an array'],
])('names the invalid path in %j', (change, message) => {
  expect(() => parseManifest({ ...manifest, ...change })).toThrow(message);
});

test('packing with a manifest override (no source) keeps platform fields; archives open with them', async () => {
  const files = sampleFiles();
  const stamped = new TextEncoder().encode(JSON.stringify({ ...decode(files['avatar.json']), platform, finalCapabilities }));
  const project: Record<string, Uint8Array> = { ...files, 'avatar.json': stamped };
  const bytes = await packProjectAvatar(async paths => Object.fromEntries([...paths].filter(p => p in project).map(p => [p, project[p]])),
    { name: 'sample', includeSource: false });
  const unpacked = unpackAvatar(bytes);
  expect(unpacked['source.png']).toBeUndefined();
  const written = decode(unpacked['avatar.json']);
  expect(written.platform).toEqual(platform);
  expect(written.finalCapabilities).toEqual(finalCapabilities);
  const opened = await openAvatar(bytes);
  expect(opened.manifest.platform).toEqual(platform);
  expect(opened.manifest.finalCapabilities?.face?.blink).toBe(false);
  opened.release();
  // An explicit override re-serialises the manifest and must not drop the blocks either.
  const repacked = unpackAvatar(packAvatar(unpacked, { manifest: { name: 'Renamed' } }));
  expect(decode(repacked['avatar.json'])).toMatchObject({ name: 'Renamed', platform, finalCapabilities });
});

const sampleRig = () => decode(sampleFiles()['rig.json']);
const square = [[100, 100], [300, 100], [300, 300], [100, 300]];

test('rigs keep optional hit zones through validation, parsing and packing', async () => {
  const rig = sampleRig();
  expect(rig.hitZones).toBeUndefined();
  expect(validateRig(rig)).toEqual([]);
  const hitZones = [
    { name: 'ribbon', outline: square, target: { kind: 'strand', name: 'front_l' } },
    { name: 'body', outline: [[0, 0], [rig.image.width, 0], [rig.image.width, rig.image.height]], side: 'left' },
  ];
  const zoned = { ...rig, hitZones };
  expect(validateRig(zoned)).toEqual([]);
  expect(parseRig(zoned).hitZones).toEqual(hitZones);
  const files = { ...sampleFiles(), 'rig.json': new TextEncoder().encode(JSON.stringify(zoned)) };
  expect(validateAvatarFiles(files)).toEqual([]);
  const opened = await openAvatar(packAvatar(files));
  expect(opened.rig.hitZones).toEqual(hitZones);
  opened.release();
});

test.each([
  [[{ name: 'none', outline: square }], 'rig.hitZones[0].name: "none" is reserved for a miss'],
  [[{ name: 'a', outline: square }, { name: 'a', outline: square }], 'rig.hitZones[1].name: must be unique'],
  [[{ name: '', outline: square }], 'rig.hitZones[0].name: expected a non-empty string'],
  [[{ name: 'a', outline: [[0, 0], [1, 1]] }], 'rig.hitZones[0].outline: expected 3..Infinity items'],
  [[{ name: 'a', outline: [[0, 0], [10, 10], [20, 20]] }], 'rig.hitZones[0].outline: needs at least three non-collinear points'],
  [[{ name: 'a', outline: [[-1, 0], [10, 0], [10, 10]] }], 'rig.hitZones[0].outline: points must stay inside the image'],
  [[{ name: 'a', outline: square, side: 'up' }], 'rig.hitZones[0].side: expected "left" or "right"'],
  [[{ name: 'a', outline: square, target: { kind: 'tail', name: 'x' } }], 'rig.hitZones[0].target.kind: expected "strand", "cloth" or "limb"'],
  [{ name: 'a' }, 'rig.hitZones: expected an array'],
])('rejects invalid hit zones %j', (hitZones, message) => {
  expect(validateRig({ ...sampleRig(), hitZones })).toContain(message);
});
