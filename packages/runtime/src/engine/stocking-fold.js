// The painted thigh stays on the leg. Only the yarn lifted between the finger and its
// contact line leaves that surface, so the knee shading never travels into the fold.
const COLUMNS = 10, ROWS = 7;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function stockingSample(image, rect) {
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 480 / Math.max(image.width, image.height));
  const width = canvas.width = Math.max(1, Math.round(image.width * scale));
  const height = canvas.height = Math.max(1, Math.round(image.height * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const alpha = new Uint8Array(width * height), tones = Array.from({ length: 32 }, () => [0, 0, 0, 0]);
  let count = 0;
  for (let i = 0; i < alpha.length; i++) {
    const offset = i * 4;
    alpha[i] = pixels[offset + 3];
    if (alpha[i] < 180) continue;
    const r = pixels[offset], g = pixels[offset + 1], b = pixels[offset + 2];
    const bin = tones[Math.floor((r + 2 * g + b) / 32)];
    bin[0] += r; bin[1] += g; bin[2] += b; bin[3]++; count++;
  }
  // A darker midtone represents the dyed yarn without copying a painted body highlight.
  let consumed = 0;
  const tone = tones.find(bin => { consumed += bin[3]; return consumed > count * .24; });
  const colour = tone ? tone.slice(0, 3).map(sum => sum / tone[3]) : [0, 0, 0];
  const [rx, ry, rw, rh] = rect, sx = rw / width, sy = rh / height;
  function covered(x, y) {
    const col = Math.floor((x - rx) / sx), row = Math.floor((y - ry) / sy);
    return col >= 0 && col < width && row >= 0 && row < height && alpha[row * width + col] > 50;
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
  return { colour, span, contact };
}

function fibreTexture(colour) {
  const canvas = document.createElement('canvas');
  const size = canvas.width = canvas.height = 96;
  const ctx = canvas.getContext('2d'), image = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const across = 2 * (x + .5) / size - 1, along = (y + .5) / size;
    const ridge = Math.exp(-20 * (across - .12) ** 2);
    const edge = Math.min(1, (1 - Math.abs(across)) * 16);
    // The attachment blends into the stocking. Gathered threads near the fingertip
    // and in the narrow creases absorb more light than the stretched single ply.
    const contactFade = Math.min(1, along / .30);
    const crease = .5 + .5 * Math.cos(15 * across + .9 * Math.sin(3 * along));
    const density = .22 + .19 * ridge + .12 * crease ** 3 + .16 * along ** 3;
    const specular = 18 * Math.exp(-85 * (across + .29) ** 2) * along;
    const i = (y * size + x) * 4;
    for (let c = 0; c < 3; c++) image.data[i + c] = colour[c] + specular;
    image.data[i + 3] = 255 * edge * contactFade * density;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
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
export function createStockingFold(renderer, name, image, rect) {
  const sample = stockingSample(image, rect), mesh = foldMesh(rect);
  const layer = renderer.addLayer(`fold_${name}`, fibreTexture(sample.colour), mesh);
  layer.visible = false;
  let lastX, lastY, legWidth;
  return { update(grip, place) {
    layer.visible = false;
    if (!grip) return;
    const [x, y] = grip.centre;
    if (x !== lastX || y !== lastY) {
      legWidth = sample.span(x, y);
      lastX = x; lastY = y;
    }
    if (!legWidth) return;
    // A normal pull projects slightly downward in a front view. In-plane drags keep
    // their direction. The local leg width limits the yarn available to the pinch.
    const dx = grip.offset[0], dy = grip.offset[1] + .28 * grip.depth;
    const length = Math.hypot(dx, dy), travel = Math.min(length, grip.reach, .78 * legWidth);
    if (travel < .004 * legWidth) return;
    const ax = dx / length, ay = dy / length, nx = -ay, ny = ax;
    const width = Math.min(.35 * legWidth, .50 * grip.reach);
    const tipX = x + ax * travel, tipY = y + ay * travel;
    for (let col = 0; col <= COLUMNS; col++) {
      const u = 2 * col / COLUMNS - 1, back = (.28 + .25 * (1 - u * u)) * width;
      const [rootX, rootY] = sample.contact(x, y, nx * width * u - ax * back, ny * width * u - ay * back);
      const rx = rootX - tipX, ry = rootY - tipY;
      const axial = rx * ax + ry * ay, lateral = rx * nx + ry * ny;
      for (let row = 0; row <= ROWS; row++) {
        const remaining = 1 - row / ROWS, taper = remaining ** 1.32;
        const spread = lateral * taper + u * Math.min(travel, width) * .035 * (1 - remaining);
        const px = tipX + ax * axial * remaining + nx * spread;
        const py = tipY + ay * axial * remaining + ny * spread;
        const [posedX, posedY] = place([px, py]);
        const i = 2 * (row * (COLUMNS + 1) + col);
        mesh.pos[i] = posedX; mesh.pos[i + 1] = posedY;
      }
    }
    layer.alpha = clamp(travel / (.34 * legWidth), 0, 1);
    layer.visible = true;
  } };
}
