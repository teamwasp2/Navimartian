/**
 * 8-connected A* on the Jezero hazard grid.
 *
 * Hard constraint: slope above maxSlope (percent) is impassable.
 * Step cost from A to B, with d = 1 orthogonal and sqrt(2) diagonal:
 *   d * (1 + wSlope * slopeB/25 + wRad * radB + wDust * dustB)
 * Heuristic is octile distance times the minimum positive step factor,
 * which stays admissible because every weight is >= 0.
 */

export const PRESETS = {
  safest: { wSlope: 3.5, wRad: 4.0, wDust: 2.0 },
  fastest: { wSlope: 0.5, wRad: 0.3, wDust: 0.2 },
};

/** Radial storm spike. Does not mutate the base terrain arrays. */
export function stormInfluence(storm, row, col) {
  if (!storm?.active || !(storm.radius > 0)) {
    return { intensity: 0, blocked: false, distance: Infinity };
  }
  const distance = Math.hypot(row - storm.row, col - storm.col);
  if (!(distance < storm.radius)) {
    return { intensity: 0, blocked: false, distance };
  }
  return {
    intensity: 1 - distance / storm.radius,
    blocked: distance < storm.radius * 0.45,
    distance,
  };
}

export function cellHazard(terrain, row, col, storm = null) {
  const r = clamp(Math.round(row), 0, terrain.rows - 1);
  const c = clamp(Math.round(col), 0, terrain.cols - 1);
  const index = r * terrain.cols + c;
  const influence = stormInfluence(storm, row, col);
  return {
    slope: terrain.slope[index],
    radiation: terrain.radiation[index] + 2.5 * influence.intensity,
    dust: Math.min(1, terrain.dust[index] + 0.9 * influence.intensity),
    blocked: influence.blocked,
    intensity: influence.intensity,
  };
}

const DIRS = [
  [1, 0, 1],
  [-1, 0, 1],
  [0, 1, 1],
  [0, -1, 1],
  [1, 1, Math.SQRT2],
  [1, -1, Math.SQRT2],
  [-1, 1, Math.SQRT2],
  [-1, -1, Math.SQRT2],
];

/** Pressurized Martian rover / SEV baseline used for traverse telemetry. */
export const ROVER_PHYSICS = {
  g: 3.72,
  massKg: 2500,
  crr0: 0.085,
  dustCrr: 0.14,
  etaDrive: 0.82,
  etaRegen: 0.35,
  tractionMaxW: 10000,
  eclssW: 1800,
  vNominal: 2.4,
  vFloor: 0.35,
};

/**
 * Newtonian step for a pressurized rover.
 * cellA / cellB provide elevation (m) and cellB.dust (0-1 slip/sinkage).
 * horizontalDistMeters is the planimetric step (cell size, or cell size * sqrt(2)).
 */
export function calculateSegmentPhysics(cellA, cellB, horizontalDistMeters) {
  const { g, massKg, crr0, dustCrr, etaDrive, etaRegen, tractionMaxW, eclssW, vNominal, vFloor } = ROVER_PHYSICS;
  const deltaD = Math.max(0, horizontalDistMeters);
  const deltaH = Number(cellB.elevation) - Number(cellA.elevation);
  const deltaS = Math.hypot(deltaD, deltaH);
  const theta = Math.atan2(deltaH, Math.max(deltaD, 1e-6));
  const dust = Math.min(1, Math.max(0, Number(cellB.dust) || 0));
  const crr = crr0 + dustCrr * dust;
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);
  const fRoll = massKg * g * crr * cosT;
  const fGrade = massKg * g * sinT;
  const fTotal = fRoll + fGrade;

  const vTerrain = vNominal * (1 - 0.35 * dust) * Math.max(0.3, 1 - 1.8 * Math.abs(sinT));
  const resisting = fRoll + fGrade;
  const vPowerCap = resisting > 0 ? (tractionMaxW * etaDrive) / resisting : Infinity;
  const vStep = Math.max(vFloor, Math.min(vTerrain, vPowerCap));
  const dtSec = deltaS > 0 ? deltaS / vStep : 0;

  const wMech = fTotal * deltaS;
  const eProp = wMech >= 0 ? wMech / etaDrive : wMech * etaRegen;
  const eEclss = eclssW * dtSec;
  const deltaKwh = Math.max(0.001, (eProp + eEclss) / 3_600_000);
  const o2KgPerHour = 0.18 * (1 + 0.35 * Math.max(0, sinT) + 0.15 * dust);
  const deltaO2 = o2KgPerHour * (dtSec / 3600);

  return {
    deltaD,
    deltaH,
    deltaS,
    theta,
    crr,
    fRoll,
    fGrade,
    fTotal,
    vStep,
    dtSec,
    dtHours: dtSec / 3600,
    deltaKwh,
    deltaO2,
  };
}

export function prepareTerrain(json) {
  const meta = json.metadata ?? {};
  const rows = meta.rows ?? json.elevation?.length;
  const cols = meta.cols ?? json.elevation?.[0]?.length;
  if (!rows || !cols) {
    throw new Error('Terrain metadata is missing rows and cols.');
  }
  const fields = ['elevation', 'slope', 'radiation', 'dust'];
  for (const name of fields) {
    if (!Array.isArray(json[name]) || json[name].length !== rows) {
      throw new Error(`Terrain field "${name}" does not match the row count.`);
    }
  }
  const n = rows * cols;
  const elevation = new Float64Array(n);
  const slope = new Float64Array(n);
  const radiation = new Float64Array(n);
  const dust = new Float64Array(n);
  let minElev = Infinity;
  let maxElev = -Infinity;
  let minRad = Infinity;
  let minDust = Infinity;
  for (let r = 0; r < rows; r += 1) {
    const eRow = json.elevation[r];
    const sRow = json.slope[r];
    const rRow = json.radiation[r];
    const dRow = json.dust[r];
    if (eRow.length !== cols || sRow.length !== cols || rRow.length !== cols || dRow.length !== cols) {
      throw new Error(`Row ${r} is not ${cols} cells wide.`);
    }
    for (let c = 0; c < cols; c += 1) {
      const i = r * cols + c;
      const e = Number(eRow[c]);
      const s = Number(sRow[c]);
      const rad = Number(rRow[c]);
      const d = Number(dRow[c]);
      if (![e, s, rad, d].every(Number.isFinite)) {
        throw new Error(`Non-finite terrain value at ${r}, ${c}.`);
      }
      elevation[i] = e;
      slope[i] = s;
      radiation[i] = rad;
      dust[i] = d;
      if (e < minElev) minElev = e;
      if (e > maxElev) maxElev = e;
      if (rad < minRad) minRad = rad;
      if (d < minDust) minDust = d;
    }
  }
  const bbox = normalizeBbox(meta.bbox);
  return {
    meta,
    rows,
    cols,
    n,
    elevation,
    slope,
    radiation,
    dust,
    minElev,
    maxElev,
    minRad,
    minDust,
    cellSizeMeters: Number(meta.cellSizeMeters) || 235,
    bbox,
  };
}

function normalizeBbox(bbox) {
  if (Array.isArray(bbox)) {
    return { west: bbox[0], south: bbox[1], east: bbox[2], north: bbox[3] };
  }
  return {
    west: bbox?.west ?? 77.3,
    south: bbox?.south ?? 18.15,
    east: bbox?.east ?? 77.9,
    north: bbox?.north ?? 18.75,
  };
}

export function snapToPassable(terrain, row, col, maxSlope, storm = null) {
  const { rows, cols, slope } = terrain;
  const open = (r, c) => slope[r * cols + c] <= maxSlope && !stormInfluence(storm, r, c).blocked;
  let r = clamp(Math.round(row), 0, rows - 1);
  let c = clamp(Math.round(col), 0, cols - 1);
  if (open(r, c)) return { row: r, col: c };

  const queue = [[r, c]];
  const seen = new Uint8Array(terrain.n);
  seen[r * cols + c] = 1;
  let head = 0;
  while (head < queue.length) {
    const [cr, cc] = queue[head];
    head += 1;
    for (let i = 0; i < 4; i += 1) {
      const nr = cr + DIRS[i][0];
      const nc = cc + DIRS[i][1];
      if (nr < 0 || nc < 0 || nr >= rows || nc >= cols) continue;
      const ni = nr * cols + nc;
      if (seen[ni]) continue;
      if (open(nr, nc)) return { row: nr, col: nc };
      seen[ni] = 1;
      queue.push([nr, nc]);
    }
  }
  return { row: r, col: c };
}

export function findPath(terrain, start, goal, weights, maxSlope, storm = null) {
  const { rows, cols, slope, minRad, minDust } = terrain;
  const wSlope = weights.wSlope;
  const wRad = weights.wRad;
  const wDust = weights.wDust;
  const blocked = (r, c) => slope[r * cols + c] > maxSlope || stormInfluence(storm, r, c).blocked;
  const startI = start.row * cols + start.col;
  const goalI = goal.row * cols + goal.col;
  if (blocked(start.row, start.col) || blocked(goal.row, goal.col)) {
    return { ok: false, path: null };
  }
  if (startI === goalI) {
    return { ok: true, path: [{ row: start.row, col: start.col }] };
  }

  const minFactor = 1 + wRad * minRad + wDust * minDust;
  const gScore = new Float64Array(terrain.n);
  gScore.fill(Infinity);
  gScore[startI] = 0;
  const cameFrom = new Int32Array(terrain.n);
  cameFrom.fill(-1);
  const closed = new Uint8Array(terrain.n);
  const heap = new MinHeap();
  heap.push({ f: octile(start.row, start.col, goal.row, goal.col) * minFactor, g: 0, i: startI });

  const limit = terrain.n * 12;
  let expansions = 0;
  while (heap.size > 0 && expansions < limit) {
    const node = heap.pop();
    expansions += 1;
    if (closed[node.i] || node.g !== gScore[node.i]) continue;
    if (node.i === goalI) {
      return { ok: true, path: reconstruct(cameFrom, goalI, startI, cols) };
    }
    closed[node.i] = 1;
    const row = Math.floor(node.i / cols);
    const col = node.i - row * cols;
    for (let d = 0; d < DIRS.length; d += 1) {
      const dr = DIRS[d][0];
      const dc = DIRS[d][1];
      const step = DIRS[d][2];
      const nr = row + dr;
      const nc = col + dc;
      if (nr < 0 || nc < 0 || nr >= rows || nc >= cols) continue;
      const ni = nr * cols + nc;
      if (closed[ni] || blocked(nr, nc)) continue;
      if (dr !== 0 && dc !== 0) {
        if (blocked(row, nc) || blocked(nr, col)) continue;
      }
      const hazard = cellHazard(terrain, nr, nc, storm);
      const factor = 1 + wSlope * (hazard.slope / 25) + wRad * hazard.radiation + wDust * hazard.dust;
      const nextG = node.g + step * factor;
      if (nextG < gScore[ni]) {
        gScore[ni] = nextG;
        cameFrom[ni] = node.i;
        const h = octile(nr, nc, goal.row, goal.col) * minFactor;
        heap.push({ f: nextG + h, g: nextG, i: ni });
      }
    }
  }
  return { ok: false, path: null };
}

export function evaluateRoute(terrain, path, storm = null) {
  const blank = {
    ok: false,
    path: path ?? null,
    distanceKm: 0,
    timeHours: 0,
    radiationMsv: 0,
    oxygenKg: 0,
    batteryKwh: 0,
    safetyScore: 0,
    avgSlope: 0,
    avgDust: 0,
    avgRadiation: 0,
    maxSlope: 0,
    cum: null,
  };
  if (!path || path.length < 2) {
    if (path && path.length === 1) {
      const hazard = cellHazard(terrain, path[0].row, path[0].col, storm);
      return {
        ...blank,
        ok: true,
        safetyScore: 100,
        avgSlope: hazard.slope,
        avgDust: hazard.dust,
        avgRadiation: hazard.radiation,
        maxSlope: hazard.slope,
        cum: {
          hours: Float64Array.of(0),
          distKm: Float64Array.of(0),
          rad: Float64Array.of(0),
          o2: Float64Array.of(0),
          kwh: Float64Array.of(0),
        },
      };
    }
    return blank;
  }

  const hours = new Float64Array(path.length);
  const distKm = new Float64Array(path.length);
  const rad = new Float64Array(path.length);
  const o2 = new Float64Array(path.length);
  const kwh = new Float64Array(path.length);
  const speed = new Float64Array(path.length);
  speed[0] = ROVER_PHYSICS.vNominal;
  let slopeW = 0;
  let dustW = 0;
  let radW = 0;
  let maxSlope = 0;

  for (let i = 1; i < path.length; i += 1) {
    const a = path[i - 1];
    const b = path[i];
    const diagonal = a.row !== b.row && a.col !== b.col;
    const horizontalM = (diagonal ? Math.SQRT2 : 1) * terrain.cellSizeMeters;
    const ai = a.row * terrain.cols + a.col;
    const hazard = cellHazard(terrain, b.row, b.col, storm);
    const physics = calculateSegmentPhysics(
      { elevation: terrain.elevation[ai], dust: 0 },
      { elevation: terrain.elevation[b.row * terrain.cols + b.col], dust: hazard.dust },
      horizontalM,
    );
    const segmentKm = horizontalM / 1000;
    const slope = hazard.slope;
    const radiation = hazard.radiation;
    const dust = hazard.dust;
    const segmentRad = (radiation * physics.dtHours) / 24;

    distKm[i] = distKm[i - 1] + segmentKm;
    hours[i] = hours[i - 1] + physics.dtHours;
    rad[i] = rad[i - 1] + segmentRad;
    o2[i] = o2[i - 1] + physics.deltaO2;
    kwh[i] = kwh[i - 1] + physics.deltaKwh;
    speed[i] = physics.vStep;
    slopeW += slope * segmentKm;
    dustW += dust * segmentKm;
    radW += radiation * segmentKm;
    if (slope > maxSlope) maxSlope = slope;
  }

  const distance = distKm[path.length - 1];
  const stats = {
    ok: true,
    path,
    distanceKm: distance,
    timeHours: hours[path.length - 1],
    radiationMsv: rad[path.length - 1],
    oxygenKg: o2[path.length - 1],
    batteryKwh: kwh[path.length - 1],
    avgSlope: distance > 0 ? slopeW / distance : 0,
    avgDust: distance > 0 ? dustW / distance : 0,
    avgRadiation: distance > 0 ? radW / distance : 0,
    maxSlope,
    cum: { hours, distKm, rad, o2, kwh, speed },
    safetyScore: 0,
  };
  stats.safetyScore = survivalScore(stats);
  return stats;
}

export function survivalScore(stats) {
  const radTerm = clamp((stats.avgRadiation - 0.55) / 0.85, 0, 1);
  const doseTerm = clamp(stats.radiationMsv / 0.22, 0, 1);
  const slopeTerm = clamp(stats.avgSlope / 16, 0, 1);
  const dustTerm = clamp((stats.avgDust - 0.12) / 0.75, 0, 1);
  const peakTerm = clamp((stats.maxSlope - 6) / 19, 0, 1);
  const timeTerm = clamp(stats.timeHours / 7, 0, 1);
  const score =
    100 -
    26 * radTerm -
    14 * doseTerm -
    18 * slopeTerm -
    16 * dustTerm -
    14 * peakTerm -
    8 * timeTerm;
  return Math.round(clamp(score, 0, 100));
}

function reconstruct(cameFrom, goalI, startI, cols) {
  const path = [];
  let cursor = goalI;
  let guard = 0;
  while (cursor !== -1 && guard < cameFrom.length + 2) {
    path.push({ row: Math.floor(cursor / cols), col: cursor % cols });
    if (cursor === startI) break;
    cursor = cameFrom[cursor];
    guard += 1;
  }
  path.reverse();
  return path;
}

function octile(r0, c0, r1, c1) {
  const dr = Math.abs(r1 - r0);
  const dc = Math.abs(c1 - c0);
  const smaller = dr < dc ? dr : dc;
  return dr + dc + (Math.SQRT2 - 2) * smaller;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

class MinHeap {
  constructor() {
    this.data = [];
  }

  get size() {
    return this.data.length;
  }

  push(node) {
    const data = this.data;
    data.push(node);
    let i = data.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (data[parent].f <= node.f) break;
      data[i] = data[parent];
      i = parent;
    }
    data[i] = node;
  }

  pop() {
    const data = this.data;
    if (data.length === 0) return undefined;
    const top = data[0];
    const last = data.pop();
    if (data.length === 0) return top;
    data[0] = last;
    let i = 0;
    const n = data.length;
    while (true) {
      const left = i * 2 + 1;
      const right = left + 1;
      let smallest = i;
      if (left < n && data[left].f < data[smallest].f) smallest = left;
      if (right < n && data[right].f < data[smallest].f) smallest = right;
      if (smallest === i) break;
      const tmp = data[i];
      data[i] = data[smallest];
      data[smallest] = tmp;
      i = smallest;
    }
    return top;
  }
}
