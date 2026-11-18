// The painted thigh stays on the leg. Only the yarn lifted between the finger and its
// contact line leaves that surface, so the knee shading never travels into the fold.
const COLUMNS = 20, ROWS = 10;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function stockingSample(image, rect) {
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 480 / Math.max(image.width, image.height));
  const width = canvas.width = Math.max(1, Math.round(image.width * scale));
  const height = canvas.height = Math.max(1, Math.round(image.height * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const [rx, ry, rw, rh] = rect, sx = rw / width, sy = rh / height;
  function pixelAt(x, y) {
    const col = Math.floor((x - rx) / sx), row = Math.floor((y - ry) / sy);
    return col >= 0 && col < width && row >= 0 && row < height ? 4 * (row * width + col) : -4;
  }
  function covered(x, y) { return pixels[pixelAt(x, y) + 3] > 50; }
  function yarnAt(x, y, radius) {
    const tones = Array.from({ length: 32 }, () => [0, 0, 0, 0]);
    let count = 0;
    for (let j = -6; j <= 6; j++) for (let k = -6; k <= 6; k++) {
      const i = pixelAt(x + k * radius / 6, y + j * radius / 6);
      if (i < 0 || pixels[i + 3] <= 50) continue;
      const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
      const bin = tones[Math.floor((r + 2 * g + b) / 32)];
      bin[0] += r; bin[1] += g; bin[2] += b; bin[3]++; count++;
    }
    let consumed = 0;
    const tone = tones.find(bin => { consumed += bin[3]; return consumed > count * .24; });
    return tone.slice(0, 3).map(sum => sum / tone[3]);
  }
  function grain(x, y) {
    const i = pixelAt(x, y);
    if (i < 0 || pixels[i + 3] < 128) return 0;
    // Retain the fine variation of the yarn, leaving the broad anatomical shading behind.
    const value = p => (pixels[p] + pixels[p + 1] * 2 + pixels[p + 2]) * .25;
    let sum = 0, n = 0;
    for (const [u, v] of [[-2, 0], [2, 0], [0, -2], [0, 2]]) {
      const q = pixelAt(x + u * sx, y + v * sy);
      if (q >= 0 && pixels[q + 3] >= 128) { sum += value(q); n++; }
    }
    return n ? clamp(value(i) - sum / n, -16, 16) : 0;
  }
  function span(x, y) {
    if (!covered(x, y)) return 0;
    let left = x, right = x;
    while (covered(left - sx, y)) left -= sx;
    while (covered(right + sx, y)) right += sx;
    return right - left + sx;
  }
  // A root can meet the side of the leg, the skirt or the top of a boot. Stop at the
  // first clear texel along the ray, even if there is more fabric on the far side.
  function contact(x, y, dx, dy) {
    const steps = Math.max(1, Math.ceil(Math.max(Math.abs(dx / sx), Math.abs(dy / sy))));
    let i = 1;
    while (i <= steps && covered(x + dx * i / steps, y + dy * i / steps)) i++;
    return [x + dx * (i - 1) / steps, y + dy * (i - 1) / steps];
  }
  return { yarnAt, grain, span, contact };
}

// Texture coordinates wrap around the two sides of the lifted membrane. At a grazing
// angle more yarn overlaps the light path; the ridge facing the light has a soft sheen.
function paintFibres(canvas, colour, grainAt) {
  const size = canvas.width, ctx = canvas.getContext('2d'), image = ctx.createImageData(size, size);
  for (let x = 0; x < size; x++) {
    const angle = 2 * Math.PI * (x + .5) / size, cross = Math.sin(angle), face = Math.cos(angle);
    for (let y = 0; y < size; y++) {
      const along = (y + .5) / size, contactFade = Math.min(1, along / .32);
      const grain = grainAt(cross, along);
      const trough = Math.exp(-22 * (cross + .20 + .08 * along) ** 2);
      const ridge = Math.exp(-13 * (cross - .50 + .15 * along) ** 2);
      const grazing = Math.abs(cross) ** 3;
      const density = .12 + .22 * grazing + .125 * trough + .22 * along ** 3;
      const sheen = (.18 + .82 * Math.max(0, -face)) * ridge * (.19 + .10 * along);
      const i = (y * size + x) * 4;
      for (let c = 0; c < 3; c++) image.data[i + c] = colour[c] * (1 - .18 * trough) + (255 - colour[c]) * sheen + grain;
      image.data[i + 3] = 255 * contactFade * (density + .003 * grain);
    }
  }
  ctx.putImageData(image, 0, 0);
}

function foldMesh(rect) {
  const uv = new Float32Array(2 * (COLUMNS + 1) * (ROWS + 1));
  const rest = new Float32Array(uv.length), tris = [], lines = [];
  for (let row = 0; row <= ROWS; row++) for (let col = 0; col <= COLUMNS; col++) {
    const i = row * (COLUMNS + 1) + col, u = col / COLUMNS, v = row / ROWS;
    uv[2 * i] = u; uv[2 * i + 1] = v;
    rest[2 * i] = rect[0] + rect[2] * u; rest[2 * i + 1] = rect[1] + rect[3] * v;
    if (col === COLUMNS || row === ROWS) continue;
    const next = i + COLUMNS + 1;
    tris.push(i, i + 1, next, i + 1, next + 1, next);
    lines.push(i, i + 1, i, next);
  }
  return { rect, rest, uv, pos: new Float32Array(rest), tris: new Uint32Array(tris), lines: new Uint32Array(lines) };
}

/** Add a translucent stocking fold directly above its fitted sock and below the skirt. */
export function createStockingFold(renderer, name, image, rect, fittedLayer) {
  const sample = stockingSample(image, rect), mesh = foldMesh(rect);
  const texture = document.createElement('canvas');
  texture.width = texture.height = 96;
  // Lighting may build its normal map before the first grab supplies the local dye.
  paintFibres(texture, [96, 96, 96], () => 0);
  const layer = renderer.addLayer(`fold_${name}`, texture, mesh);
  const tension = fittedLayer.stockingTension = new Float32Array(8);
  layer.visible = false;
  let lastX, lastY, legWidth;
  return { update(grip, place) {
    layer.visible = false;
    tension[6] = 0;
    if (!grip) return;
    const [x, y] = grip.centre;
    if (x !== lastX || y !== lastY) {
      legWidth = sample.span(x, y);
      lastX = x; lastY = y;
      if (legWidth) {
        const radius = legWidth * .38;
        paintFibres(texture, sample.yarnAt(x, y, radius), (u, v) => sample.grain(x + u * radius, y + (2 * v - 1) * radius));
        const gl = renderer.gl;
        gl.bindTexture(gl.TEXTURE_2D, layer.tex);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, texture);
        gl.generateMipmap(gl.TEXTURE_2D);
      }
    }
    if (!legWidth) return;
    // A normal pull projects slightly downward in a front view. Stretch is limited by
    // the width of this section of the leg, independently of the sprite's bounding box.
    const dx = grip.offset[0], dy = grip.offset[1] + .28 * grip.depth;
    const length = Math.hypot(dx, dy), travel = Math.min(length, grip.reach, 1.02 * legWidth);
    if (travel < .004 * legWidth) return;
    const ax = dx / length, ay = dy / length, nx = -ay, ny = ax;
    const strength = clamp(travel / (.56 * legWidth), 0, 1);
    const width = legWidth * (.27 + .15 * strength), along = width * .66;
    tension.set([x, y, ax, ay, 1.28 * width, width, strength, 0]);
    const tipX = x + ax * travel, tipY = y + ay * travel;
    const centre = place([x, y]), acrossX = place([x + 1, y]), acrossY = place([x, y + 1]);
    const ex = [acrossX[0] - centre[0], acrossX[1] - centre[1]];
    const ey = [acrossY[0] - centre[0], acrossY[1] - centre[1]];
    for (let col = 0; col <= COLUMNS; col++) {
      const angle = 2 * Math.PI * col / COLUMNS, ca = Math.cos(angle), sa = Math.sin(angle);
      const [rootX, rootY] = sample.contact(x, y, ax * along * ca + nx * width * sa, ay * along * ca + ny * width * sa);
      const root = place([rootX, rootY]);
      const rx = rootX - tipX, ry = rootY - tipY;
      const axial = rx * ax + ry * ay, lateral = rx * nx + ry * ny;
      for (let row = 0; row <= ROWS; row++) {
        const t = row / ROWS, remaining = 1 - t, bow = Math.sin(Math.PI * t);
        const cap = Math.sqrt(1 - Math.max(0, (t - .80) / .20) ** 2);
        const spread = lateral * remaining * (1 + .18 * bow) + sa * .08 * width * t * cap;
        // The pinch has a rounded crown. Gravity bows the two walls between their
        // stationary contact ring and the fingers; both walls transmit the background.
        const forward = axial * remaining + .065 * width * ca * t * cap;
        const px = tipX + ax * forward + nx * spread, py = tipY + ay * forward + ny * spread + .07 * width * bow;
        const i = 2 * (row * (COLUMNS + 1) + col);
        // Off-leg yarn follows the local material frame even if it crosses the other leg.
        mesh.pos[i] = root[0] + (px - rootX) * ex[0] + (py - rootY) * ey[0];
        mesh.pos[i + 1] = root[1] + (px - rootX) * ex[1] + (py - rootY) * ey[1];
      }
    }
    layer.alpha = strength;
    layer.visible = true;
  } };
}
