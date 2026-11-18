// Local three-dimensional displacement layered over the animated drawing. X and Y are source pixels;
// z is the thickness change relative to the resting chest or garment, positive towards the camera.
import type { Cloth, Ellipse, Point, Rig } from '../rig/types';

type Side = 'left' | 'right';
export type SurfaceTarget = { kind: 'chest'; side: Side } | { kind: 'fabric'; name: string };
interface Mesh { rest: Float32Array; pos: Float32Array; tris: Uint32Array }
interface Layer { mesh: Mesh; surfaceLight?: Float32Array }
interface Grip {
  mesh: Mesh;
  triangle: number[];
  weights: number[];
  rest: Point;
  pointer: Point;
  offset: Point;
  depth: number;
}
interface Patch { mesh: Mesh; vertices: number[]; weights: number[]; light: Float32Array; slot: number }

const clamp = (n: number, low: number, high: number) => Math.max(low, Math.min(n, high));
const sample = (coords: Float32Array, indices: number[], weights: number[]): Point => [
  coords[indices[0]] * weights[0] + coords[indices[1]] * weights[1] + coords[indices[2]] * weights[2],
  coords[indices[0] + 1] * weights[0] + coords[indices[1] + 1] * weights[1] + coords[indices[2] + 1] * weights[2],
];

function pick(mesh: Mesh, x: number, y: number): Grip | null {
  const p = mesh.pos;
  for (let j = 0; j < mesh.tris.length; j += 3) {
    const a = mesh.tris[j] * 2, b = mesh.tris[j + 1] * 2, c = mesh.tris[j + 2] * 2;
    const bx = p[b] - p[a], by = p[b + 1] - p[a + 1], cx = p[c] - p[a], cy = p[c + 1] - p[a + 1];
    const dx = x - p[a], dy = y - p[a + 1], det = bx * cy - by * cx;
    if (Math.abs(det) < 1e-7) continue;
    const u = (dx * cy - dy * cx) / det, v = (bx * dy - by * dx) / det;
    if (u < -1e-5 || v < -1e-5 || 1 - u - v < -1e-5) continue;
    const triangle = [a, b, c], weights = [1 - u - v, u, v];
    return { mesh, triangle, weights, rest: sample(mesh.rest, triangle, weights), pointer: [x, y], offset: [0, 0], depth: 0 };
  }
  return null;
}

function bounds(points: Point[]) {
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  return { width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) };
}

/** Distance to the sewn boundary, measured on the original cloth. */
function seamDistance(x: number, y: number, seam: Point[]): number {
  let distance = Infinity;
  for (let i = 1; i < seam.length; i++) {
    const a = seam[i - 1], b = seam[i], sx = b[0] - a[0], sy = b[1] - a[1];
    const t = clamp(((x - a[0]) * sx + (y - a[1]) * sy) / (sx * sx + sy * sy || 1), 0, 1);
    distance = Math.min(distance, Math.hypot(x - a[0] - t * sx, y - a[1] - t * sy));
  }
  return distance;
}

function lineHeight(line: Point[], x: number): number {
  let i = 1;
  while (i + 1 < line.length && x > line[i][0]) i++;
  const [a, b] = [line[i - 1], line[i]];
  const t = clamp((x - a[0]) / (b[0] - a[0] || 1), 0, 1);
  return a[1] + (b[1] - a[1]) * t;
}

// Older knit rigs use the upper attachment line for the collar. For woven panels the
// authored neckline distinguishes a free upper edge from a jacket lapel.
const hasNeckline = (cloth: Cloth) => cloth.material === 'knit' || (cloth.material === 'woven' && !!cloth.neckline);

class Surface {
  private patches: Patch[] = [];
  readonly displacement = [0, 0, 0];
  private velocity = [0, 0, 0];
  private grip: Grip | null = null;
  private previousAnchor: Point | null = null;
  private previousVelocity: Point = [0, 0];
  private angle = 0;
  private desired = [0, 0, 0];
  private readonly reach: number;
  private readonly radius: Point;
  private centre: Point;
  private meshes: { layer: Layer; slot: number }[] = [];

  private readonly cloth: Cloth | null;
  private readonly side: number;
  private readonly support: number;
  private readonly collar?: { neck: Point[]; hem: Point[]; left: number; right: number; opens: boolean };
  constructor(ellipse: Ellipse, cloth: Cloth | null, side: number, support: number, availableTravel?: number) {
    this.cloth = cloth; this.side = side; this.support = support;
    this.centre = [ellipse.cx, ellipse.cy];
    this.radius = [ellipse.rx, ellipse.ry];
    const travel = cloth?.reach ?? Math.min(...this.radius) * (cloth ? 0.65 : 0.45);
    this.reach = Math.min(travel, availableTravel ?? travel);
    if (cloth && hasNeckline(cloth)) {
      const xs = cloth.outline.map(point => point[0]);
      this.collar = { neck: cloth.neckline ?? cloth.waist, hem: cloth.hem, left: Math.min(...xs), right: Math.max(...xs), opens: availableTravel === undefined };
    }
  }

  addLayer(layer: Layer, slot: number) {
    this.meshes.push({ layer, slot });
    layer.surfaceLight ??= new Float32Array(12);
    if (!this.cloth) this.bindPatch(layer, slot);
  }

  private weight(x: number, y: number) {
    const u = (x - this.centre[0]) / this.radius[0], v = (y - this.centre[1]) / this.radius[1];
    const edge = Math.max(0, 1 - u * u - v * v);
    if (this.collar) {
      const { neck, hem, opens } = this.collar;
      const left = neck[0][0], width = neck[neck.length - 1][0] - left;
      const across = clamp((x - left) / width, 0, 1);
      // The neck band ends at the shoulder seams. A pull travels up the bodice to
      // the free lip, while the waist seam stays with the torso.
      const shoulders = 4 * across * (1 - across);
      const top = lineHeight(neck, x), bottom = lineHeight(hem, x);
      const along = clamp((y - top) / Math.max(1, bottom - top), 0, 1);
      const fromHem = clamp((1 - along) / .65, 0, 1);
      const fixedHem = fromHem * fromHem * (3 - 2 * fromHem);
      // Below the shoulder attachments, even a grip to one side can stretch the
      // chest panel; the outside armholes still stay with the jacket/sleeves.
      const fromSide = clamp(Math.min(x - this.collar.left, this.collar.right - x) / (.14 * (this.collar.right - this.collar.left)), 0, 1);
      const panel = shoulders + (fromSide - shoulders) * Math.min(1, along / .3);
      const columns = Math.exp(-(((x - this.centre[0]) / (.65 * width)) ** 2));
      const transmission = columns * Math.exp(-Math.max(0, y - this.centre[1]) / this.radius[1]);
      const collar = opens ? 1 : Math.min(1, along * 4);
      return fixedHem * collar * Math.max(edge * edge * panel, transmission * shoulders);
    }
    // A compact smooth surface fixes the surrounding rib cage / cloth. Pin cloth to its sewn edge.
    const attach = this.cloth && this.cloth.material !== 'stocking'
      ? Math.min(1, seamDistance(x, y, this.cloth.waist) / (this.radius[1] * 0.42)) : 1;
    return edge * edge * attach;
  }

  private bindPatch(layer: Layer, slot: number) {
    const mesh = layer.mesh, vertices: number[] = [], weights: number[] = [];
    for (let i = 0; i < mesh.rest.length; i += 2) {
      const w = this.weight(mesh.rest[i], mesh.rest[i + 1]);
      if (w > 0) { vertices.push(i); weights.push(w); }
    }
    this.patches.push({ mesh, vertices, weights, light: layer.surfaceLight!, slot });
  }

  hold(x: number, y: number) {
    let grip: Grip | null = null;
    for (let i = this.meshes.length - 1; i >= 0 && !grip; i--) grip = pick(this.meshes[i].layer.mesh, x, y);
    if (!grip) return false;
    if (this.cloth) {
      this.centre = grip.rest;
      this.patches.length = 0;
      for (const { layer, slot } of this.meshes) this.bindPatch(layer, slot);
    }
    this.grip = grip;
    return true;
  }

  move(dx: number, dy: number, depth: number) {
    if (this.grip) { this.grip.offset = [dx, dy]; this.grip.depth = depth; }
  }

  release(vx: number, vy: number) {
    this.grip = null;
    const c = Math.cos(this.angle), s = Math.sin(this.angle);
    this.velocity[0] += clamp(c * vx + s * vy, -4 * this.reach, 4 * this.reach) * 0.22;
    this.velocity[1] += clamp(-s * vx + c * vy, -4 * this.reach, 4 * this.reach) * 0.22;
  }

  prepare(dt: number, place: (p: Point) => Point, gravity: Point) {
    const anchor = place(this.centre), horizontal = place([this.centre[0] + this.radius[0], this.centre[1]]);
    this.angle = Math.atan2(horizontal[1] - anchor[1], horizontal[0] - anchor[0]);
    const c = Math.cos(this.angle), s = Math.sin(this.angle);
    this.desired.fill(0);
    if (!this.cloth && dt > 0) {
      if (this.previousAnchor) {
        const v: Point = [(anchor[0] - this.previousAnchor[0]) / dt, (anchor[1] - this.previousAnchor[1]) / dt];
        const ax = v[0] - this.previousVelocity[0], ay = v[1] - this.previousVelocity[1];
        // An accelerating chest wall leaves the attached mass behind. Support limits the relative momentum.
        const mass = 0.55 - this.support * 0.35;
        this.velocity[0] -= clamp(c * ax + s * ay, -8 * this.reach, 8 * this.reach) * mass;
        this.velocity[1] -= clamp(-s * ax + c * ay, -8 * this.reach, 8 * this.reach) * mass;
        this.previousVelocity = v;
      }
      this.previousAnchor = anchor;
      // The painted shape already includes gravity at rest. Apply only the change as the torso tilts.
      const sag = this.reach * (0.45 - this.support * 0.30);
      this.desired[0] = (c * gravity[0] + s * gravity[1] - gravity[0]) * sag;
      this.desired[1] = (-s * gravity[0] + c * gravity[1] - gravity[1]) * sag;
    }
    const g = this.grip;
    if (g) {
      const base = sample(g.mesh.pos, g.triangle, g.weights);
      const dx = g.pointer[0] + g.offset[0] - base[0], dy = g.pointer[1] + g.offset[1] - base[1];
      const travel = this.reach * (this.cloth ? 1 : 1 - 0.45 * this.support);
      const x = c * dx + s * dy, y = -s * dx + c * dy;
      // The leg supports a stocking along its length. A pleat unfolds towards its waistband
      // and tightens quickly when pulled down or across neighbouring folds.
      const material = this.cloth?.material;
      const wovenBodice = this.collar && material === 'woven';
      const horizontal = material === 'pleat' ? 0.55 : material === 'knit' ? 0.66 : wovenBodice ? 0.52 : 1;
      const vertical = material === 'stocking' ? 0.36 : material === 'pleat' && y > 0 ? 0.32 : wovenBodice ? 0.82 : 1;
      const length = Math.hypot(x / horizontal, y / vertical) / travel;
      const gain = length ? Math.tanh(length) / length : 0;
      this.desired[0] += x * gain;
      this.desired[1] += y * gain;
      // Pressing into the chest wall / leg has a shorter, progressively firmer stroke.
      this.desired[2] = travel * g.depth * (g.depth >= 0 ? 1 : 0.33);
    }
  }

  advance(dt: number, other?: Surface) {
    const d = this.displacement;
    const h = Math.min(dt, 0.05), steps = Math.ceil(h * 240);
    const omega = this.grip ? 32 : (this.cloth?.material === 'stocking' ? 24 : 14 + 8 * this.support);
    const damp = this.grip ? 0.86 : (this.cloth?.material === 'stocking' ? 0.68 : 0.53 + 0.16 * this.support);
    for (let step = 0; step < steps; step++) for (let k = 0; k < 3; k++) {
      const w = omega * (k === 1 ? 0.91 : k === 2 ? 1.18 : 1);
      const target = this.desired[k] + (other ? other.displacement[k] * 0.065 : 0);
      const err = target - d[k], v = this.velocity[k];
      this.velocity[k] += (w * w * err * (1 + 0.4 * Math.min(1, Math.abs(err) / this.reach)) - 2 * damp * w * v) * h / steps;
      d[k] += this.velocity[k] * h / steps;
      const lower = k === 2 ? -0.35 * this.reach : -this.reach;
      if (d[k] < lower || d[k] > this.reach) { d[k] = clamp(d[k], lower, this.reach); this.velocity[k] = 0; }
    }
    const c = Math.cos(this.angle), s = Math.sin(this.angle), [dx, dy, z] = d;
    const active = Math.abs(dx) + Math.abs(dy) + Math.abs(z) > 0.003;
    // The lip of a top pulled towards the camera projects downwards in a front view.
    // A lateral pull also lowers the near side as the cloth comes under tension.
    const collarDrop = this.collar ? clamp(dy + Math.abs(dx) * .52 + Math.max(0, z) * .78, -.45 * this.reach, this.reach) : 0;
    for (const p of this.patches) {
      // Shader reads all lobes in rest space; the normal of a stretched sock also highlights its direction of tension.
      const strength = this.cloth ? (z + Math.hypot(dx, dy) * 0.38) / this.reach : z / this.reach;
      p.light.set([this.centre[0], this.centre[1], Math.min(...this.radius), active ? strength : 0], p.slot * 4);
      if (!active) continue;
      for (let v = 0; v < p.vertices.length; v++) {
        const i = p.vertices[v], weight = p.weights[v];
        const u = (p.mesh.rest[i] - this.centre[0]) / this.radius[0], h = (p.mesh.rest[i + 1] - this.centre[1]) / this.radius[1];
        // Compression spreads tissue in-plane; forward stretch narrows it. A raised pleat turns towards the viewer.
        const x = dx + z * (0.22 * this.side - (this.cloth ? 0.18 : 0.68) * u);
        const y = this.collar ? collarDrop : dy - z * (0.22 + (this.cloth ? 0.16 : 0.46) * h);
        p.mesh.pos[i] += weight * (c * x - s * y);
        p.mesh.pos[i + 1] += weight * (s * x + c * y);
      }
    }
  }
}

export class SoftSurfaces {
  private chest = new Map<Side, Surface>();
  private fabrics = new Map<string, Surface>();
  private gravity: Point;
  constructor(rig: Rig, underlayers: ReadonlySet<string> = new Set()) {
    const soft = rig.body.softChest;
    if (soft) for (const side of ['left', 'right'] as const) {
      this.chest.set(side, new Surface(soft[side], null, side === 'left' ? -1 : 1, soft.support ?? 0.35));
    }
    for (const cloth of rig.cloth ?? []) if (cloth.material && cloth.reach && cloth.reach > 0) {
      const { width, height } = bounds(cloth.outline), pleat = cloth.material === 'pleat';
      // A pleat transmits strain along its fold; elastic knits spread it across a smaller patch.
      const rx = Math.max(2, width * (pleat ? 0.27 : 0.42), (cloth.reach ?? 0) * (pleat ? 1.0 : 1.65));
      const ry = Math.max(2, Math.min(height * (pleat || hasNeckline(cloth) ? 0.76 : 0.42), rx * (pleat ? 3.2 : 1.75)));
      // Without an authored underlayer, the builder only fills the narrow band sized for passive travel.
      const available = underlayers.has(cloth.name) ? undefined : cloth.max ?? 20;
      this.fabrics.set(cloth.name, new Surface({ cx: 0, cy: 0, rx, ry }, cloth, 0, 0, available));
    }
    const a = rig.physics?.gravity ?? Math.PI / 2;
    this.gravity = [Math.cos(a), Math.sin(a)];
  }
  attach(layer: Layer, cloth?: Cloth) {
    if (!cloth || cloth.material === 'woven' || cloth.material === 'knit') for (const [side, surface] of this.chest) surface.addLayer(layer, side === 'left' ? 0 : 1);
    if (cloth) this.fabrics.get(cloth.name)?.addLayer(layer, 2);
  }
  private get(t: SurfaceTarget) { return t.kind === 'chest' ? this.chest.get(t.side) : this.fabrics.get(t.name); }
  can(t: SurfaceTarget) { return !!this.get(t); }
  hold(t: SurfaceTarget, x: number, y: number) { return this.get(t)?.hold(x, y) ?? false; }
  move(t: SurfaceTarget, x: number, y: number, z: number) { this.get(t)?.move(x, y, z); }
  release(t: SurfaceTarget, vx: number, vy: number) { this.get(t)?.release(vx, vy); }
  step(dt: number, place: (p: Point) => Point) {
    for (const s of this.chest.values()) s.prepare(dt, place, this.gravity);
    for (const s of this.fabrics.values()) s.prepare(dt, place, this.gravity);
    for (const [side, s] of this.chest) s.advance(dt, this.chest.get(side === 'left' ? 'right' : 'left'));
    for (const s of this.fabrics.values()) s.advance(dt);
  }
}
