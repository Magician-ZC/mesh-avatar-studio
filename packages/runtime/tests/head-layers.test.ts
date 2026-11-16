// Head layers (tools/head/layers.py): with a face field a head turn moves only the face, by the whole shift where the
// field is 1 and not at all where it is 0; the hair layer's weights do not turn; a rig without the field deforms
// exactly as before.
import { describe, expect, test } from 'vitest';
import fixture from '../../../samples/miko-qipao/rig.json';
import { parseRig, PARAMS } from 'mesh-avatar';
import { createRig } from '../src/engine/rig.js';
import { createPhysics } from '../src/engine/physics.js';

const rig = parseRig(fixture);
const defaults: Record<string, number> = Object.fromEntries(PARAMS.map(p => [p.id, p.def]));
const phys = { gain: 0, bunL: [0, 0], bunR: [0, 0], strands: [] };
const { head, face } = rig;
// a face field for the test: 1 within 120 px of the point between the eyes and the mouth, 0 from 200 px on
const centre: [number, number] = [(face.eyeA.cx + face.eyeB.cx) / 2, (face.eyeA.cy + face.eyeB.cy + 2 * rig.mouth.cy) / 4];
const faceAt = (x: number, y: number) => Math.min(1, Math.max(0, (200 - Math.hypot(x - centre[0], y - centre[1])) / 80));
type Engine = ReturnType<typeof createRig>;
type Weights = ReturnType<Engine['baseWeights']>;
const move = (engine: Engine, point: [number, number], pose: Record<string, number>, weights?: (w: Weights) => object): [number, number] => {
  const out = [0, 0];
  const w = engine.baseWeights(point[0], point[1], 0);
  engine.deformBase(point[0], point[1], (weights ? weights(w) : w) as never, { ...defaults, ...pose }, phys, out);
  return [out[0] - point[0], out[1] - point[1]];
};
// the plate alone: without the nose's and the mouth's own lead (their weights are Gaussians with long tails)
const plate = (w: Weights) => ({ ...w, nose: 0, mouth: 0 });
// rotations about a pivot leave rounding noise even at angle 0, so "does not move" is to 1e-9 px
const still = (moved: [number, number]) => { expect(moved[0]).toBeCloseTo(0, 9); expect(moved[1]).toBeCloseTo(0, 9); };

describe('head layers: the face field', () => {
  const layered = createRig(rig, { faceAt });
  const classic = createRig(rig);
  // on the forehead above the eyes: inside the field's flat top, away from the nose and mouth
  const flat: [number, number] = [centre[0], centre[1] - 60];
  const hairline: [number, number] = [centre[0] - 230, centre[1] - 40]; // outside the field, inside the head ellipse
  const pose = { angleX: 30, angleY: -30 };

  test('reports the mode', () => {
    expect(layered.layered).toBe(true);
    expect(classic.layered).toBe(false);
  });

  test('moves the face by the whole shift where the field is 1 and leaves the rest of the head where it is', () => {
    expect(faceAt(...flat)).toBe(1);
    expect(faceAt(...hairline)).toBe(0);
    const moved = move(layered, flat, pose, plate);
    expect(moved[0]).toBeCloseTo(head.shiftX, 9);
    expect(moved[1]).toBeCloseTo(head.shiftY, 9);
    still(move(layered, hairline, pose));
    // the original turn bulges the whole head: hair inside the head ellipse moves too
    expect(Math.abs(move(classic, hairline, pose)[0])).toBeGreaterThan(5);
  });

  test('scales with the field in between, and the turn band still ends it below the chin', () => {
    const half: [number, number] = [centre[0] + 160, centre[1] - 30];
    expect(faceAt(...half)).toBeCloseTo(0.5, 1);
    expect(move(layered, half, { angleX: 30 }, plate)[0]).toBeCloseTo(faceAt(...half) * head.shiftX, 9);
    expect(move(layered, half, { angleX: -15 }, plate)[0]).toBeCloseTo(-0.5 * faceAt(...half) * head.shiftX, 9);
    const below: [number, number] = [centre[0], head.turnBand[1] + 5];
    still(move(createRig(rig, { faceAt: () => 1 }), below, { angleX: 30 }, plate));
  });

  test('the nose and mouth lead by offsets in units of the rig\'s shift; ears and buns stay with the head', () => {
    const everywhere = { faceAt: () => 1 };
    const nose: [number, number] = [face.nose.cx, face.nose.cy];
    const lead = (engine: Engine) => move(engine, nose, { angleX: 30 })[0];
    const small = lead(createRig(rig, everywhere));
    const doubled = lead(createRig({ ...rig, head: { ...head, shiftX: head.shiftX * 2, shiftY: head.shiftY * 2 } }, everywhere));
    expect(small).toBeGreaterThan(head.shiftX + 5); // the plate's shift plus the nose's own
    expect(doubled).toBeCloseTo(2 * small, 6);
    // an ear: in the original turn it slides the other way; under a field of 0 it does not move at all
    const ear: [number, number] = [face.earR.cx, face.earR.cy];
    still(move(createRig(rig, { faceAt: () => 0 }), ear, { angleX: 30 }, plate));
    expect(Math.abs(move(createRig(rig), ear, { angleX: 30 })[0])).toBeGreaterThan(3);
  });

  test('the hair layer\'s weights (no turn, no face parts) sway and roll with the head but do not slide', () => {
    const hairWeights = (w: Weights) => ({ ...w, turn: 0, nose: 0, mouth: 0, brow: 0, jaw: 0 });
    still(move(layered, flat, pose, hairWeights));
    const rolled = move(layered, flat, { angleZ: 20 }, hairWeights);
    expect(rolled).toEqual(move(layered, flat, { angleZ: 20 }, plate));
    expect(Math.hypot(...rolled)).toBeGreaterThan(5);
  });

  test('hair strands hang from anchors that do not slide with the face', () => {
    const at: [number, number] = [flat[0], flat[1]];
    const anchor = (engine: Engine) => new (createPhysics(engine, rig).Physics)().hairAnchor(at, { ...defaults, angleX: 30 });
    expect(anchor(layered)).toEqual(at);
    expect(anchor(classic)[0]).toBeGreaterThan(at[0] + 5);
  });

  test('a rig without the field deforms exactly as before', () => {
    const untouched = createRig(rig, null);
    const poses: Record<string, number>[] = [{ angleX: 30 }, { angleX: -17, angleY: 22 }, { angleY: -30, angleZ: 12 }];
    for (const each of poses) {
      for (const point of [flat, hairline, [face.nose.cx, face.nose.cy], [face.earR.cx, face.earR.cy]] as [number, number][]) {
        expect(move(untouched, point, each)).toEqual(move(classic, point, each));
      }
    }
  });
});
