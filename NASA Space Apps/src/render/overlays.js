import * as THREE from 'three';

export function createSlopeTexture(terrain, maxSlope) {
  return canvasTexture(paint(terrain, (slope) => slopeColor(slope, maxSlope), (i) => terrain.slope[i]));
}

export function createRiskTexture(terrain) {
  return canvasTexture(
    paint(terrain, (_value, i) => riskColor(terrain.radiation[i], terrain.dust[i]), () => 0),
  );
}

function paint(terrain, colorAt, sample) {
  const { rows, cols } = terrain;
  const source = document.createElement('canvas');
  source.width = cols;
  source.height = rows;
  const ctx = source.getContext('2d', { willReadFrequently: true });
  const image = ctx.createImageData(cols, rows);
  const data = image.data;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i = r * cols + c;
      const [red, green, blue] = colorAt(sample(i), i);
      const p = i * 4;
      data[p] = red;
      data[p + 1] = green;
      data[p + 2] = blue;
      data[p + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  const out = document.createElement('canvas');
  out.width = 1024;
  out.height = 1024;
  const outCtx = out.getContext('2d');
  outCtx.imageSmoothingEnabled = true;
  outCtx.imageSmoothingQuality = 'high';
  outCtx.drawImage(source, 0, 0, 1024, 1024);
  return out;
}

function slopeColor(slope, maxSlope) {
  const limit = Math.max(1, maxSlope);
  const t = slope / limit;
  if (t >= 1) {
    const k = Math.min(1, (slope - limit) / 25);
    return mix([255, 36, 58], [92, 0, 18], k);
  }
  if (t < 0.45) return mix([14, 74, 78], [214, 176, 52], t / 0.45);
  return mix([214, 176, 52], [232, 64, 36], (t - 0.45) / 0.55);
}

function riskColor(radiation, dust) {
  const t = clamp((radiation - 0.5) / 0.95, 0, 1);
  const heat = t < 0.55 ? mix([12, 32, 58], [198, 78, 34], t / 0.55) : mix([198, 78, 34], [255, 236, 214], (t - 0.55) / 0.45);
  return mix(heat, [214, 176, 104], clamp(dust, 0, 1) * 0.62);
}

function canvasTexture(canvas) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.needsUpdate = true;
  return texture;
}

function mix(a, b, t) {
  const k = clamp(t, 0, 1);
  return [
    Math.round(a[0] + (b[0] - a[0]) * k),
    Math.round(a[1] + (b[1] - a[1]) * k),
    Math.round(a[2] + (b[2] - a[2]) * k),
  ];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
