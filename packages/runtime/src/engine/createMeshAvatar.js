// Mesh avatar engine: turns a pre-split single illustration into a
// 2D mesh avatar animated on a WebGL2 canvas. No UI and no framework: the app drives it
// through the small API returned by createMeshAvatar().
import { createRig } from './rig.js';
import { createRenderer } from './renderer.js';
import { createPhysics } from './physics.js';
import { createCloth } from './cloth.js';
import { createStockingFold } from './stocking-fold.js';
import { createSpriteModule, eyeMood } from './sprites.js';
import { openEyeAlpha, openEyeLayerName } from './open-eyes.js';
import { eyePartNames, gazeShift, hasIrisLayers, irisLayerNames, irisRim, isEyeFill } from './iris-layers.js';
import { createOverlayLayers } from './overlays.js';
import { Motion } from './motion.js';
import { ExpressionOverlay } from './expression-overlay.js';
import { parseExpressionSettings } from './expression-settings.ts';
import { MOTIONS, IDLE_MOTIONS } from './motions.js';
import { clientToImage, imageToClient } from '../interaction/coordinates.ts';
import { createImpulse } from '../interaction/impulse.ts';
import { CHEEK_SIDES, CheekPull, parseCheeks } from '../interaction/cheek.ts';
import { SoftSurfaces } from '../interaction/soft-surfaces.ts';
import { createGrabs } from '../interaction/grab.ts';
import { createLimbs } from '../limbs/runtime.ts';

const imageCache = new Map();
const jsonCache = new Map();
async function loadJson(src) {
  if (!jsonCache.has(src)) jsonCache.set(src, fetch(src).then(r => { if (!r.ok) throw new Error(`failed to load ${src}`); return r.json(); }));
  return jsonCache.get(src);
}
function loadImage(src) {
  if (imageCache.has(src)) return imageCache.get(src);
  const result = new Promise((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = () => rej(new Error(`failed to load ${src}`));
    i.src = src;
  });
  imageCache.set(src, result);
  return result;
}

function channelOf(img, ch) {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const g = c.getContext('2d');
  g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  const a = new Uint8Array(c.width * c.height);
  for (let i = 0; i < a.length; i++) a[i] = d[i * 4 + ch];
  return a;
}
const alphaOf = img => channelOf(img, 3);
// Overlay layers only: a canvas holding the [x, y, w, h] part of an image (used as a texture).
function cropImage(img, [x, y, w, h]) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(img, x, y, w, h, 0, 0, w, h);
  return c;
}

// Two-bone skinning along the tassel: top follows segment 1, lower part segment 2.
function deformTassel(ch, mesh) {
  const [px, py] = ch.t.pivot, d = ch.dir, P = ch.pts;
  const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const c1 = Math.cos(ch.phi[0]), s1 = Math.sin(ch.phi[0]);
  const c2 = Math.cos(ch.phi[1]), s2 = Math.sin(ch.phi[1]);
  const r = mesh.rest, o = mesh.pos;
  for (let i = 0; i < r.length; i += 2) {
    const vx = r[i] - px, vy = r[i + 1] - py;
    const along = vx * d[0] + vy * d[1];
    const ax = P[0][0] + vx * c1 - vy * s1, ay = P[0][1] + vx * s1 + vy * c1;
    const ux = vx - d[0] * ch.L1, uy = vy - d[1] * ch.L1;
    const bx = P[1][0] + ux * c2 - uy * s2, by = P[1][1] + ux * s2 + uy * c2;
    const w = sstep(ch.L1 * 0.55, ch.L1 * 1.35, along);
    o[i] = ax + (bx - ax) * w; o[i + 1] = ay + (by - ay) * w;
  }
}

// Optional head layers (built by tools/head/layers.py, see docs/format.md).
const STILL_STRAND = [[0, 0], [0, 0], [0, 0], [0, 0]];
const FACE_FIELD = 'face_field', HAIR_FRONT = 'hair_front', FACE_SKIN = 'face_skin';

/** @param {HTMLCanvasElement} canvas
 * @param {import('./index').MeshAvatarOptions} options */
export async function createMeshAvatarImpl(canvas, options) {
  if (!options.assets && options.assetsBase === undefined) {
    throw new Error('createMeshAvatar requires assets or assetsBase.');
  }
  const rig = options.rig;
  const base = (options.assetsBase ?? '').replace(/\/?$/, '/');
  const asset = (name, build) => {
    if (options.assets) {
      if (!options.assets[name]) throw new Error(`Missing project asset: ${name}`);
      return options.assets[name];
    }
    return `${base}${name}${build === undefined ? '' : `?b=${build}`}`;
  };
  const meta = await loadJson(asset('layers.json'));
  // optional head layers (tools/head/layers.py): with face_field a head turn slides the face inside its contour instead
  // of bulging the whole head; hair_front and face_skin (always together) put the hair over the face on its own layer,
  // with skin under it. An avatar without them, or a runtime that cannot load them, turns the head the original way.
  let head = null;
  if (meta.layers?.[FACE_FIELD]) {
    try {
      // the hair layer comes with a skin layer under it, unless the base itself is the picture without that hair
      const names = [FACE_FIELD, ...(meta.layers[HAIR_FRONT] ? [HAIR_FRONT] : []), ...(meta.layers[HAIR_FRONT] && meta.layers[FACE_SKIN] ? [FACE_SKIN] : [])];
      const loaded = await Promise.all(names.map(n => loadImage(asset(`${n}.png`, meta.build))));
      const mask = loaded[0], hair = names.includes(HAIR_FRONT) ? loaded[1] : null, skin = names.includes(FACE_SKIN) ? loaded[2] : null;
      const [fx, fy] = meta.layers[FACE_FIELD], fw = mask.width, fh = mask.height, field = channelOf(mask, 0);
      head = {
        hair: hair ?? null, skin: skin ?? null,
        faceAt(x, y) {
          const u = Math.round(x) - fx, v = Math.round(y) - fy;
          return u < 0 || v < 0 || u >= fw || v >= fh ? 0 : field[v * fw + u] / 255;
        },
      };
    } catch (err) {
      console.warn('head layers not loaded:', err);
    }
  }
  const engine = createRig(rig, head);
  const { IMG, EYES, baseWeights, lockWeight, deformBase, eyePartY, eyePartAlpha,
    handWeights, handFrame, deformHand, TASSELS } = engine;
  const { Renderer, buildGrid } = createRenderer(engine, rig);
  const { Physics } = createPhysics(engine, rig);
  const { Cloth } = createCloth(engine, rig);
  const CLOTH = rig.cloth ?? [];
  const { createSprites } = createSpriteModule(engine, rig);

  // with garments the base drawing is the one without them (base.png keeps them for older runtimes)
  const garmentBase = meta.garmentBase;
  const stockingTextures = Object.assign({}, ...Object.values(meta.garmentUnderlayers ?? {}).map(entry => entry.stockings ?? {}));
  const skirtBodyLayers = Object.entries(meta.garmentUnderlayers ?? {}).flatMap(([owner, entry]) => [
    ...Object.entries(entry.legs ?? {}), ...(entry.shorts ? [['pelvis', entry.shorts]] : []),
    ...(entry.contact ? [['contact', entry.contact]] : []),
  ].map(([part, name]) => ({ part, name, owner, legs: entry.legs })));
  const baseName = CLOTH.length ? 'base_cloth' : 'base';
  // with limb chains (ER4) the base is the bare one, without the limbs (layers.json "bare"; base.png keeps them)
  const LIMBS = rig.limbs ?? [];
  const coveredSleeves = LIMBS.filter(l => meta.layers?.[`sleeve_${l.name}`] && meta.layers?.[`sleeve_cover_${l.name}`]);
  const accessoryPlates = TASSELS.flatMap((t, index) => {
    const name = `accessory_under_${t.name}`;
    const garment = CLOTH.find(c => c.name === meta.accessoryBacks?.[t.name]);
    return garment && meta.layers?.[name] ? [{ name, index, garment }] : [];
  });
  const bareName = LIMBS.length && typeof meta.bare === 'string' ? meta.bare : null;
  // iris layers (see iris-layers.js): an eye with both is drawn from them, not from its eye ball
  const irisEyes = await Promise.all([0, 1].map(async i => {
    if (!hasIrisLayers(meta.layers, i)) return false;
    try {
      await Promise.all(Object.values(irisLayerNames(i)).map(n => loadImage(asset(`${n}.png`, meta.build))));
      return true;
    } catch (err) {
      console.warn(`iris layers of eye ${i} not loaded:`, err);
      return false;
    }
  }));
  const names = [baseName, ...(garmentBase ? [garmentBase] : []), ...(bareName ? [bareName] : []), ...(rig.hand ? ['hand'] : []), ...TASSELS.map(t => t.name), ...CLOTH.map(c => `cloth_${c.name}`), ...LIMBS.map(l => `limb_${l.name}`), ...[0, 1].flatMap(i => eyePartNames(irisEyes[i]).map(p => `eye${i}_${p}`)), 'hairmask'];
  for (const { name } of coveredSleeves) names.push(`sleeve_${name}`, `sleeve_cover_${name}`);
  for (const { name } of accessoryPlates) names.push(name);
  for (const { name } of skirtBodyLayers) names.push(name);
  names.push(...Object.values(stockingTextures));
  const imgs = Object.fromEntries(await Promise.all(names.map(async n => [n, await loadImage(asset(`${n}.png`, meta.build))])));
  if (CLOTH.length) imgs.base = imgs.base_cloth;
  // strand layers (tools/hair/strands.py, layers.json "strands"): for a strand with a reach, its lock and the base filled
  // in behind it; a strand whose layers are missing or do not load is held by bending the base, as without them
  const strandLayers = (await Promise.all((rig.strands ?? []).map(async (s, index) => {
    const entry = s.reach > 0 ? meta.strands?.[s.name] : null;
    if (!entry || !meta.layers?.[entry.lock] || !meta.layers?.[entry.under]) return null;
    try {
      const [lock, under] = await Promise.all([entry.lock, entry.under].map(n => loadImage(asset(`${n}.png`, meta.build))));
      return { index, entry, lock, under };
    } catch (err) {
      console.warn(`strand layers of ${s.name} not loaded:`, err);
      return null;
    }
  }))).filter(Boolean);
  if (bareName) imgs.base = imgs[bareName];
  if (garmentBase) imgs.base = imgs[garmentBase];
  const hairMask = channelOf(imgs.hairmask, 0);
  const hairAt = (x, y) => hairMask[Math.min(IMG.h - 1, Math.max(0, Math.round(y))) * IMG.w + Math.min(IMG.w - 1, Math.max(0, Math.round(x)))] / 255;

  // The source image is cut flat at its top edge (crown and right bun). A negative top margin
  // keeps the top ~88px (7%) above the canvas, which the layout puts at the top of the panel:
  // the cut stays off screen even at the deepest head tilt.
  const R = new Renderer(canvas, { padTop: options.padTop ?? rig.view.padTop, padSide: options.padSide ?? rig.view.padSide, fit: options.fit });
  const rects = { base: [0, 0, IMG.w, IMG.h], ...meta.layers };

  // base layer, denser where hair strands bend and the face parts shift
  // garments behind the body (drawn first) and in front of it, each skinned to its cloth grid
  const cloth = new Cloth();
  const clothMeshes = new Map();
  const stockingFolds = new Map();
  const surfaces = new SoftSurfaces(rig, new Set(Object.keys(meta.garmentUnderlayers ?? {})));
  const addCloth = c => {
    const n = `cloth_${c.name}`;
    const image = imgs[stockingTextures[c.name] ?? n];
    const mesh = buildGrid(rects[n], rig.mesh.clothCell ?? 8, alphaOf(image), image.width);
    clothMeshes.set(c.name, mesh);
    cloth.attach(c.name, mesh);
    surfaces.attach(R.addLayer(n, image, mesh), c);
    if (c.material === 'stocking' && c.reach > 0) {
      stockingFolds.set(c.name, createStockingFold(R, c.name, image, rects[n]));
    }
  };
  CLOTH.filter(c => c.behind).forEach(addCloth);
  // limb chains (ER4): their own layers, skinned to their two bones; behind ones before the base, the rest after the
  // garments and overlays
  const limbs = createLimbs(rig, engine);
  const addLimb = (l, n = `limb_${l.name}`) => {
    const mesh = buildGrid(rects[n], rig.mesh.limbCell ?? rig.mesh.baseCell, alphaOf(imgs[n]), imgs[n].width);
    limbs.attach(l.name, mesh);
    R.addLayer(n, imgs[n], mesh);
  };
  LIMBS.filter(l => l.behind).forEach(l => addLimb(l));

  const baseMesh = buildGrid(rects.base, rig.mesh.baseCell, alphaOf(imgs.base), imgs.base.width, rig.mesh.fine);
  const baseW = [];
  for (let i = 0; i < baseMesh.rest.length / 2; i++) {
    const x = baseMesh.rest[i * 2], y = baseMesh.rest[i * 2 + 1];
    baseW.push(baseWeights(x, y, hairAt(x, y)));
  }
  surfaces.attach(R.addLayer('base', imgs.base, baseMesh, { face: true }));
  // The body behind a lifted skirt stays with its own leg, including where the legs cross.
  const bodyUnderCloth = skirtBodyLayers.map(({ part, name, owner, legs }) => {
    const image = imgs[name];
    const mesh = buildGrid(rects[name], rig.mesh.baseCell, alphaOf(image), image.width);
    const W = Array.from({ length: mesh.rest.length / 2 }, (_, i) => {
      const weights = baseWeights(mesh.rest[i * 2], mesh.rest[i * 2 + 1], 0);
      if (weights.legs && (part === 'left' || part === 'right')) weights.legs.right = part === 'right' ? 1 : 0;
      return weights;
    });
    const layer = part === 'contact' ? null : R.addLayer(name, image, mesh);
    return { name, mesh, W, layer, part, owner, legs };
  });

  // Strand layers, drawn only while their strand is held (or springs back after it): the plate takes the place of the
  // base round the lock, still where the base bends with the strand, and the lock goes over it as one piece across its
  // width. Both move with the head and body like the base; the plate also with the other strands.
  const heldStrands = strandLayers.map(({ index, entry, lock, under }) => {
    const plate = { mesh: buildGrid(rects[entry.under], rig.mesh.baseCell, new Uint8Array([255]), 1), W: [] };
    const own = { mesh: buildGrid(rects[entry.lock], rig.mesh.fine?.cell ?? rig.mesh.baseCell, alphaOf(lock), lock.width), W: [] };
    for (let k = 0; k < plate.mesh.rest.length / 2; k++) {
      const x = plate.mesh.rest[k * 2], y = plate.mesh.rest[k * 2 + 1], w = baseWeights(x, y, hairAt(x, y));
      plate.W.push({ ...w, strands: w.strands.filter(([i]) => i !== index) });
    }
    for (let k = 0; k < own.mesh.rest.length / 2; k++) {
      const x = own.mesh.rest[k * 2], y = own.mesh.rest[k * 2 + 1];
      own.W.push({ ...baseWeights(x, y, hairAt(x, y)), strands: [lockWeight(index, x, y)] });
    }
    plate.layer = R.addLayer(entry.under, under, plate.mesh, { replace: true });
    own.layer = R.addLayer(entry.lock, lock, own.mesh);
    plate.layer.visible = own.layer.visible = false;
    return { index, items: [plate, own] };
  });

  // head layers: the skin under the hair moves with the face and hides the base's own copy of that hair
  const headCell = rig.mesh.fine?.cell ?? rig.mesh.baseCell;
  const headLayer = (name, img, hairLayer) => {
    const mesh = buildGrid(rects[name], headCell, alphaOf(img), img.width);
    const W = [];
    for (let k = 0; k < mesh.rest.length / 2; k++) {
      const x = mesh.rest[k * 2], y = mesh.rest[k * 2 + 1];
      const w = baseWeights(x, y, hairAt(x, y));
      // the hair layer sways and rolls with the head but does not slide with the face
      W.push(hairLayer ? { ...w, turn: 0, nose: 0, mouth: 0, brow: 0, jaw: 0 } : w);
    }
    return { mesh, W, layer: null };
  };
  const faceSkin = head?.skin ? headLayer(FACE_SKIN, head.skin, false) : null;
  if (faceSkin) faceSkin.layer = R.addLayer(FACE_SKIN, head.skin, faceSkin.mesh);

  // optional cheek layers (tools/head/cheeks.py, ../interaction/cheek.ts): drawn only while a cheek is pulled, over the
  // base and the skin under the hair, under the eyes, the drawn mouths and the hair in front of the face
  const cheekData = parseCheeks(meta.cheeks);
  const cheekLayers = {};
  for (const [i, side] of CHEEK_SIDES.entries()) {
    const n = `cheek_${side}`;
    if (!cheekData[side] || !meta.layers?.[n]) continue;
    try {
      const img = await loadImage(asset(`${n}.png`, meta.build));
      const mesh = buildGrid(rects[n], headCell, alphaOf(img), img.width);
      const W = [];
      for (let k = 0; k < mesh.rest.length / 2; k++) {
        const x = mesh.rest[k * 2], y = mesh.rest[k * 2 + 1];
        W.push(baseWeights(x, y, hairAt(x, y)));
      }
      const layer = R.addLayer(n, img, mesh, { cheek: i });
      layer.visible = false;
      cheekLayers[side] = { mesh, W, layer };
    } catch (err) {
      console.warn(`cheek layer ${side} not loaded:`, err);
    }
  }
  const cheeks = Object.keys(cheekLayers).length
    ? new CheekPull(cheekData, Object.fromEntries(Object.entries(cheekLayers).map(([side, c]) => [side, c.mesh.rest])))
    : null;
  // two rest points on each pulled contour, whose drawn positions say where the cheek is and how it is turned
  const cheekProbes = Object.fromEntries(CHEEK_SIDES.flatMap(side => {
    const at = cheeks?.anchor(side);
    if (!at) return [];
    const below = [at[0], at[1] + 24];
    return [[side, { at, below, w: baseWeights(at[0], at[1], 0), wb: baseWeights(below[0], below[1], 0) }]];
  }));

  // eyes: white+iris clipped by the lids, then the line layers on top
  const eyeParts = [];
  EYES.forEach((e, i) => {
    for (const part of eyePartNames(irisEyes[i])) {
      const n = `eye${i}_${part}`;
      const mesh = buildGrid(rects[n], isEyeFill(part) ? rig.mesh.eyeBallCell : rig.mesh.eyeCell, alphaOf(imgs[n]), imgs[n].width);
      const W = [];
      for (let k = 0; k < mesh.rest.length / 2; k++) W.push(baseWeights(mesh.rest[k * 2], mesh.rest[k * 2 + 1], 0));
      const layer = R.addLayer(n, imgs[n], mesh, { eyeBall: part === 'ball' ? i : undefined, eyeWhite: part === 'white' ? i : undefined, eyeIris: part === 'iris' ? i : undefined });
      eyeParts.push({ e, eye: i, part, mesh, W, layer });
    }
  });

  // optional open-eye layers (see open-eyes.js): the drawn pixels over the cut-out lids while an eye is open
  const openEyes = [];
  for (let i = 0; i < EYES.length; i++) {
    // under a hair layer the open eye is the one of the face without that hair (the hair is drawn above it)
    const under = `${openEyeLayerName(i)}_under`;
    const n = head?.hair && meta.layers?.[under] ? under : openEyeLayerName(i);
    if (!meta.layers?.[n]) continue;
    const img = await loadImage(asset(`${n}.png`, meta.build));
    const mesh = buildGrid(rects[n], rig.mesh.eyeCell, alphaOf(img), img.width);
    const W = [];
    for (let k = 0; k < mesh.rest.length / 2; k++) {
      const x = mesh.rest[k * 2], y = mesh.rest[k * 2 + 1];
      W.push(baseWeights(x, y, hairAt(x, y)));
    }
    openEyes.push({ eye: i, mesh, W, layer: R.addLayer(n, img, mesh, { face: true }) });
  }

  // drawn eye / mouth variants (closed / half / smiling eyes, a/i/u/e/o mouths) over the face
  let sprites = null;
  try {
    {
      const sheet = await loadJson(asset('sprites/sprites.json'));
      const simgs = Object.fromEntries(await Promise.all(Object.keys(sheet.layers).map(async n =>
        [n, await loadImage(asset(`sprites/${n}.png`, sheet.build))])));
      sprites = createSprites(R, sheet, simgs, buildGrid, alphaOf);
    }
  } catch (err) {
    console.warn('eye / mouth sprites not loaded:', err);
  }

  // head layers: the hair over the face, above the eyes and the drawn eye / mouth variants
  const hairFront = head?.hair ? headLayer(HAIR_FRONT, head.hair, true) : null;
  if (hairFront) hairFront.layer = R.addLayer(HAIR_FRONT, head.hair, hairFront.mesh);

  const contactLayers = bodyUnderCloth.filter(item => item.part === 'contact');
  for (const garment of CLOTH.filter(c => !c.behind)) {
    const contact = contactLayers.find(item => item.owner === garment.name);
    if (contact) contact.layer = R.addLayer(contact.name, imgs[contact.name], contact.mesh);
    addCloth(garment);
  }
  // A contact shadow stays on the leg until the hem lifts away on that side.
  function contactProbe(mesh, cx, cy) {
    let nearest = 0, distance = Infinity;
    for (let i = 0; i < mesh.rest.length; i += 2) {
      const d = (mesh.rest[i] - cx) ** 2 + (mesh.rest[i + 1] - cy) ** 2;
      if (d < distance) { nearest = i; distance = d; }
    }
    const px = mesh.rest[nearest], py = mesh.rest[nearest + 1];
    return { mesh, i: nearest, px, py, weights: baseWeights(px, py, 0) };
  }
  const hemContacts = contactLayers.filter(item => item.layer).map(item => {
    const [, y, , h] = item.mesh.rect;
    const [left, right] = Object.values(item.legs).map(name => {
      const [x, , w] = rects[name];
      return x + w / 2;
    }).sort((a, b) => a - b);
    const centre = (left + right) * .5;
    const probes = [left, centre, right].map(cx => {
      const cy = y + h * .25;
      const near = [contactProbe(clothMeshes.get(item.owner), cx, cy)];
      for (const c of CLOTH.filter(c => c.material === 'stocking')) {
        const [ox, oy, ow, oh] = rects[`cloth_${c.name}`];
        if (cx >= ox && cx <= ox + ow && cy >= oy && cy <= oy + oh) near.push(contactProbe(clothMeshes.get(c.name), cx, cy));
      }
      return near;
    });
    const fade = new Float32Array([left, centre, right, 0, 0, 0, 0, 0]);
    item.layer.hemContact = fade;
    return { probes, fade };
  });
  const accessoryBacks = accessoryPlates.map(({ name, index, garment }) => {
    const mesh = buildGrid(rects[name], rig.mesh.baseCell, alphaOf(imgs[name]), imgs[name].width);
    const layer = R.addLayer(name, imgs[name], mesh);
    cloth.attach(garment.name, mesh);
    surfaces.attach(layer, garment);
    layer.visible = false;
    return { layer, index };
  });

  // optional overlay layers (glasses and other accessories) drawn above the eye / mouth sprites
  let overlays = null;
  try {
    const sheet = await loadJson(asset('overlays/overlays.json'));
    const images = Object.fromEntries(await Promise.all(sheet.overlays.map(async item =>
      [item.name, await loadImage(asset(`overlays/${item.file}`))])));
    overlays = createOverlayLayers(R, sheet, images, buildGrid, alphaOf, baseWeights, rig.mesh.spriteCell, IMG.w, IMG.h, cropImage);
  } catch (err) {
    // A folder avatar without overlays simply 404s; only warn when the archive lists them.
    if (options.assets?.['overlays/overlays.json']) console.warn('overlay layers not loaded:', err);
  }

  LIMBS.filter(l => !l.behind).forEach(l => addLimb(l));
  const sleeveCovers = coveredSleeves.map(l => {
    addLimb(l, `sleeve_${l.name}`);
    const name = `sleeve_cover_${l.name}`;
    const mesh = buildGrid(rects[name], rig.mesh.limbCell ?? rig.mesh.baseCell, alphaOf(imgs[name]), imgs[name].width);
    const W = Array.from({ length: mesh.rest.length / 2 }, (_, k) => baseWeights(mesh.rest[2 * k], mesh.rest[2 * k + 1], 0));
    R.addLayer(name, imgs[name], mesh);
    return { mesh, W };
  });
  const attachedLayers = [faceSkin, hairFront, ...sleeveCovers, ...bodyUnderCloth].filter(Boolean);

  const tassels = TASSELS.map(t => {
    const mesh = buildGrid(rects[t.name], rig.mesh.tasselCell, alphaOf(imgs[t.name]), imgs[t.name].width);
    return R.addLayer(t.name, imgs[t.name], mesh);
  });

  const handMesh = rig.hand ? buildGrid(rects.hand, rig.mesh.handCell, alphaOf(imgs.hand), imgs.hand.width) : { rest: [], pos: [] };
  const handW = [];
  for (let i = 0; i < handMesh.rest.length / 2; i++) handW.push(handWeights(handMesh.rest[i * 2], handMesh.rest[i * 2 + 1]));
  if (rig.hand) R.addLayer('hand', imgs.hand, handMesh);

  const physics = new Physics();
  const motion = new Motion(rig.view.gazeCenter ?? [rig.head.cx, rig.head.cy], !!rig.legs, rig.arms ? rig : null);
  if (options.lipSync !== undefined) motion.setLipSync(options.lipSync);
  const listeners = new Set();
  motion.onMotion = id => { for (const fn of listeners) fn(id); };
  // pointer interaction (ER1): only runs when impulse() is called
  const applyImpulse = createImpulse(rig, { motion, physics, cloth, limbs });
  // the latest frame's parameters and physics output, for grabs that start or move between frames
  let lastP = null, lastPhys = null;
  const place = (point, weights = baseWeights(point[0], point[1], 0)) => {
    if (!lastP) return [point[0], point[1]];
    const out = [0, 0];
    deformBase(point[0], point[1], weights, lastP, lastPhys, out);
    return out;
  };
  const cheekFrame = side => {
    const probe = cheekProbes[side];
    if (!probe) return null;
    const drawn = place(probe.at, probe.w), below = place(probe.below, probe.wb);
    const angle = Math.atan2(below[0] - drawn[0], below[1] - drawn[1]) * -1;
    return { rest: probe.at, drawn, angle: Number.isFinite(angle) ? angle : 0 };
  };
  // sustained grabs (../interaction/grab.ts): nothing runs until something is grabbed
  const grabs = createGrabs({
    rig, physics, cheeks, cheekFrame, place, surfaces,
    garment: name => cloth.garment(name),
    limbs: limbs ? { chains: limbs.chains, values: (name, P) => limbs.values(name, P) } : null,
    parameters: () => lastParameters,
    impulse: (target, vx, vy) => applyImpulse(target, vx, vy),
  });

  let parameters = {}, parameterWeight = 1, parameterWeights = {}, lastParameters = {};
  const expression = new ExpressionOverlay(options.expressions === undefined ? undefined : parseExpressionSettings(options.expressions));
  const tmp = [0, 0];
  function updateParameters(dt) {
    const P = { ...motion.update(dt) };
    // limbs without parameters sway a little while idling (not in manual mode; the sway gain scales it)
    limbs?.idle(P, motion.t, motion.mode === 'manual' ? 0 : physics.gain);
    for (const [key, value] of Object.entries(parameters)) {
      const weight = parameterWeight * (parameterWeights[key] ?? 1);
      P[key] = weight === 1 ? value : (P[key] ?? 0) + (value - (P[key] ?? 0)) * weight;
    }
    // Speech owns the mouth while active; explicit pose sliders still own all other parameters.
    if (motion.lipOpen !== null) {
      P.mouthOpen = motion.P.mouthOpen;
      if (!options.preserveMouthForm || parameters.mouthForm === undefined) P.mouthForm = motion.P.mouthForm;
    }
    // Live expressions sit above tracking and lip sync; neutral leaves the parameters untouched.
    const out = expression.apply(P, dt);
    // a held (or settling) limb follows the pointer, above everything else
    if (grabs.active) grabs.parameters(out);
    lastParameters = out;
    return out;
  }
  function tick(dt) {
    const P = updateParameters(dt);
    if (grabs.active) { grabs.step(dt); grabs.beforePhysics(P); }
    const phys = physics.step(P, dt);
    lastP = P; lastPhys = phys;

    const bp = baseMesh.pos, br = baseMesh.rest;
    // a held strand drawn on its own layers does not bend the base: its plate covers the base there
    const layered = heldStrands.length ? heldStrands.filter(s => physics.held[s.index] > 0).map(s => s.index) : [];
    const basePhys = layered.length ? { ...phys, strands: phys.strands.map((o, i) => layered.includes(i) ? STILL_STRAND : o) } : phys;
    for (let i = 0; i < baseW.length; i++) {
      deformBase(br[i * 2], br[i * 2 + 1], baseW[i], P, basePhys, tmp);
      bp[i * 2] = tmp[0]; bp[i * 2 + 1] = tmp[1];
    }
    // with drawn eye sprites the layered eye is only ever shown exactly as drawn: moving the
    // cut-out lash leaves seams at its edges
    const eyeOpenFor = (v, eye) => (sprites?.eyesDrawn[eye] ? 1 : v);
    // px of iris travel; rig.view.irisTravel scales it for large eyes
    const gaze = gazeShift(P.gazeX, P.gazeY, rig.view.irisTravel ?? 7);
    for (const ep of eyeParts) {
      const open = eyeOpenFor(ep.eye === 0 ? P.eyeROpen : P.eyeLOpen, ep.eye);
      const smile = ep.eye === 0 ? P.eyeSmile : (P.eyeSmileL ?? P.eyeSmile);
      ep.layer.alpha = eyePartAlpha(ep.part, open, smile);
      // an iris layer is a sprite: it moves by the gaze as it is, then with the head like every eye layer
      const sx = ep.part === 'iris' ? gaze[ep.eye][0] : 0, sy = ep.part === 'iris' ? gaze[ep.eye][1] : 0;
      const r = ep.mesh.rest, o = ep.mesh.pos;
      for (let k = 0; k < ep.W.length; k++) {
        const x = r[k * 2], y = r[k * 2 + 1];
        deformBase(x + sx, eyePartY(ep.e, ep.part, x, y, open, smile) + sy, ep.W[k], P, phys, tmp);
        o[k * 2] = tmp[0]; o[k * 2 + 1] = tmp[1];
      }
    }
    // the motion expression is not applied in manual mode, so only the live one counts there
    const cover = sprites?.update(P, phys, dt, eyeMood(expression.name, motion.mode === 'manual' ? null : motion.expr));
    if (cover) for (const ep of eyeParts) if (cover.eyes[ep.eye]) ep.layer.alpha = 0;
    const eyeRim = [irisRim(0), irisRim(0)], openShown = [0, 0];
    for (const oe of openEyes) {
      const alpha = openEyeAlpha(oe.eye === 0 ? P.eyeROpen : P.eyeLOpen, oe.eye === 0 ? P.eyeSmile : (P.eyeSmileL ?? P.eyeSmile),
        Boolean(sprites?.eyesDrawn[oe.eye]), Boolean(cover?.eyes[oe.eye]));
      oe.layer.alpha = alpha;
      eyeRim[oe.eye] = irisRim(alpha);
      openShown[oe.eye] = alpha;
      oe.layer.visible = alpha > 0;
      if (!oe.layer.visible) continue;
      const r = oe.mesh.rest, o = oe.mesh.pos;
      for (let k = 0; k < oe.W.length; k++) {
        deformBase(r[k * 2], r[k * 2 + 1], oe.W[k], P, phys, tmp);
        o[k * 2] = tmp[0]; o[k * 2 + 1] = tmp[1];
      }
    }
    // An iris eye's open-eye layer has its hole right at the lids. While it is fully shown the line layers under it
    // are not drawn: under its edge, however the textures are filtered, only the eye itself may lie.
    for (const ep of eyeParts) if (irisEyes[ep.eye] && !isEyeFill(ep.part) && openShown[ep.eye] >= 0.999) ep.layer.alpha = 0;
    for (const { index, items } of heldStrands) {
      const shown = physics.held[index] > 0;
      for (const item of items) {
        item.layer.visible = shown;
        if (!shown) continue;
        const r = item.mesh.rest, o = item.mesh.pos;
        for (let k = 0; k < item.W.length; k++) {
          deformBase(r[k * 2], r[k * 2 + 1], item.W[k], P, phys, tmp);
          o[k * 2] = tmp[0]; o[k * 2 + 1] = tmp[1];
        }
      }
    }
    for (const { layer, index } of accessoryBacks) {
      const ch = physics.chains[index];
      const swing = Math.abs(ch.L1 * Math.sin(ch.phi[0]) + ch.L2 * Math.sin(ch.phi[1]));
      layer.alpha = Math.min(1, swing / 24);
      layer.visible = swing > 0.05;
    }
    for (const item of attachedLayers) {
      const r = item.mesh.rest, o = item.mesh.pos;
      for (let k = 0; k < item.W.length; k++) {
        deformBase(r[k * 2], r[k * 2 + 1], item.W[k], P, phys, tmp);
        o[k * 2] = tmp[0]; o[k * 2 + 1] = tmp[1];
      }
    }
    // pulled cheeks: their layers take the pull in rest space, then move with the head like the base
    if (cheeks) {
      const angles = {};
      for (const side of Object.keys(cheekLayers)) angles[side] = cheekFrame(side)?.angle ?? 0;
      cheeks.step(dt, angles);
      for (const [side, c] of Object.entries(cheekLayers)) {
        const d = cheeks.displacement(side);
        c.layer.visible = !!d;
        if (!d) continue;
        const r = c.mesh.rest, o = c.mesh.pos;
        for (let k = 0; k < c.W.length; k++) {
          deformBase(r[k * 2] + d[k * 2], r[k * 2 + 1] + d[k * 2 + 1], c.W[k], P, phys, tmp);
          o[k * 2] = tmp[0]; o[k * 2 + 1] = tmp[1];
        }
      }
    }
    overlays?.update(P, phys, deformBase, tmp);
    const hp = handMesh.pos, hr = handMesh.rest, HF = handFrame(P);
    for (let i = 0; i < handW.length; i++) {
      deformHand(hr[i * 2], hr[i * 2 + 1], handW[i], P, HF, tmp);
      hp[i * 2] = tmp[0]; hp[i * 2 + 1] = tmp[1];
    }
    physics.chains.forEach((ch, k) => deformTassel(ch, tassels[k].mesh));
    if (CLOTH.length) { if (grabs.active) grabs.beforeCloth(); cloth.step(P, dt, phys); cloth.deform(); }
    if (limbs) { limbs.step(dt); limbs.deform(P, phys); }
    surfaces.step(dt, place);
    for (const [name, fold] of stockingFolds) fold.update(surfaces.stockingPull(name), place);
    for (const { probes, fade } of hemContacts) {
      probes.forEach((near, side) => {
        let travel = 0;
        for (const { mesh, i, px, py, weights } of near) {
          deformBase(px, py, weights, P, phys, tmp);
          travel = Math.max(travel, Math.hypot(mesh.pos[i] - tmp[0], mesh.pos[i + 1] - tmp[1]));
        }
        fade[4 + side] = -Math.max(0, (travel - IMG.h * .0006) / (IMG.h * .004));
      });
    }

    R.draw({
      angleX: P.angleX, angleY: P.angleY, angleZ: P.angleZ,
      eyes: [
        eyeOpenFor(P.eyeROpen, 0), P.eyeSmile, gaze[0][0], gaze[0][1],
        eyeOpenFor(P.eyeLOpen, 1), P.eyeSmileL ?? P.eyeSmile, gaze[1][0], gaze[1][1],
      ],
      eyeRim,
      // with sprites the drawn mouths replace the shader-painted one
      mouthOpen: cover?.mouthDrawn ? 0 : P.mouthOpen, mouthForm: P.mouthForm, cheek: P.blush,
      cheekPull: cheeks ? [...cheeks.shading('left'), ...cheeks.shading('right')] : null,
      showMesh: false, originalAlpha: 0, joints: [],
    });
  }

  // `manual: true` runs no animation loop: the caller advances time with advance()
  // (offline video rendering, one call per frame)
  let raf = 0, last = performance.now(), destroyed = false;
  const frame = now => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    tick(dt);
    raf = requestAnimationFrame(frame);
  };
  if (!options.manual) raf = requestAnimationFrame(frame);

  const motionList = [
    ...Object.entries(MOTIONS).map(([id, m]) => ({ id, label: m.label, idle: false })),
    ...Object.entries(IDLE_MOTIONS).map(([id, m]) => ({ id, label: m.label, idle: true })),
  ];

  return {
    motions: motionList,
    /** Whether this avatar can pull a cheek (cheek layers built with --cheek-pull and loaded); per side see canGrab(). */
    cheekPull: !!(cheeks?.has('left') || cheeks?.has('right')),
    setParameters(values, weight = 1, weights = {}) {
      parameters = { ...values }; parameterWeight = Math.min(1, Math.max(0, Number(weight) || 0));
      parameterWeights = Object.fromEntries(Object.entries(weights ?? {}).map(([key, value]) => [key, Math.min(1, Math.max(0, Number(value) || 0))]));
    },
    getParameters() { return { ...lastParameters }; },
    /** 0..1 loudness of the voice being played (e.g. normalised RMS). */
    setVoiceLevel(v) { motion.setVoiceLevel(v); },
    /** Partial mouth-movement settings; omitted fields retain their current values. */
    setLipSync(settings) { motion.setLipSync(settings); },
    /** true while TTS audio is playing: idle motions pause and the head nods along. */
    setSpeaking(on) { motion.setSpeaking(on); },
    /** AITuber OnAir emotion tag: happy / sad / angry / surprised / relaxed / neutral (or null). */
    setEmotion(tag, opts) { motion.setEmotion(tag, opts); },
    /** Live expression layered over tracking (see OVERLAY_EXPRESSIONS); 'neutral' clears it. */
    setExpression(name) { expression.set(name); },
    /** Show the named overlay layers (built/overlays/overlays.json) and hide the others. */
    setOverlays(names) { overlays?.set(names); },
    getExpression() { return { name: expression.name, active: expression.active }; },
    /** How much the head moves with the voice while speaking (1 = default, calmer below). */
    setTalkGain(g) { motion.talkGain = Math.max(0, Number(g) || 0); },
    /** Play a motion or idle motion by id (see `motions`). */
    play(id) { if (motion.hasMotion(id)) motion.playMotion(id); },
    /** Move the mouth through the vowels of kana text (no audio; for previews). */
    speakKana(text, options) { motion.speakKana(String(text ?? ''), options); },
    holdMouth(vowel) { motion.holdMouth(vowel); },
    stopLipSync() { motion.stopLipSync(); },
    getLipSyncState() { return motion.getLipSyncState(); },
    setAutoIdle(on) { motion.autoIdle = !!on; },
    setLighting(value) { R.setLighting(value); },
    getLightingStats() { return { ...R.lightingStats }; },
    setAutoMotion(on) { motion.autoMotion = !!on; },
    /** Hair / tassel sway multiplier (1 = default). */
    setSwayGain(g) { physics.gain = g; cloth.gain = g; },
    setClothTuning(name, tuning) { cloth.setTuning(name, tuning); },
    /** Called with the motion id when a motion starts and with null when it ends. */
    onMotion(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    /** Viewport px (e.g. PointerEvent.clientX/Y) -> source image px; null before the first frame is drawn. */
    toImagePoint(clientX, clientY) { return destroyed ? null : clientToImage(canvas, R, clientX, clientY); },
    /** Source image px -> viewport px (the inverse of toImagePoint, following view.rotation); null before the first frame. */
    toClientPoint(x, y) { return destroyed ? null : imageToClient(canvas, R, x, y); },
    /** Instantaneous velocity (source px/s) for the body, head, a strand, a garment or a limb; unknown targets are ignored. */
    impulse(target, vx, vy) { if (!destroyed) applyImpulse(target, vx, vy); },
    /**
     * Hold a part at the pointer: a cheek, a hair strand, a garment's hem, a limb or a hanging accessory, pressed at
     * (x, y) in source px. Returns a handle for moveGrab / releaseGrab, or null when this avatar cannot grab the target.
     */
    grab(target, x, y) { return destroyed ? null : grabs.grab(target, x, y); },
    /** Whether grab(target, ...) would hold something on this avatar. */
    canGrab(target) { return !destroyed && grabs.can(target); },
    /** The pointer's offset from the press (source px) and, for cheeks, depth -1..1 (towards the camera > 0). */
    moveGrab(handle, dx, dy, depth = 0) { if (!destroyed) grabs.move(handle, dx, dy, depth); },
    /** Let go with a velocity (source px/s, bounded); unknown or stale handles are ignored. */
    releaseGrab(handle, vx = 0, vy = 0) { if (!destroyed) grabs.release(handle, vx, vy); },
    /** Advance the simulation by `sec` and draw (for tests / hidden tabs). */
    advance(sec, fps = 60) {
      if (destroyed) return;
      if (sec === 0) tick(0);
      else for (let i = 0; i < Math.round(sec * fps); i++) tick(1 / fps);
    },
    /** Update motion and lip sync without drawing a hidden preview. */
    advanceParameters(sec) { if (!destroyed) updateParameters(sec); },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      grabs.releaseAll();
      cancelAnimationFrame(raf);
      listeners.clear();
      R.destroy();
      // the GL context is not lost on purpose: the canvas may already be reused by a new
      // instance (React StrictMode mounts twice) and a canvas only ever has one context
    },
  };
}
