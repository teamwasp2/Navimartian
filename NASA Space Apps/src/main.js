import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  PRESETS,
  cellHazard,
  evaluateRoute,
  findPath,
  prepareTerrain,
  snapToPassable,
  stormInfluence,
} from './pathfinding/astar.js';
import { createRiskTexture, createSlopeTexture } from './render/overlays.js';

const WORLD = 100;
const BASE = import.meta.env.BASE_URL;
const previousDispose = window.__marsDispose;
if (typeof previousDispose === 'function') previousDispose();

const ui = {
  loading: document.querySelector('#loading'),
  loadingMessage: document.querySelector('#loading-message'),
  sourcePill: document.querySelector('#source-pill'),
  gridPill: document.querySelector('#grid-pill'),
  siteName: document.querySelector('#site-name'),
  siteSeg: document.querySelector('#site-seg'),
  stormBtn: document.querySelector('#storm-btn'),
  stormMove: document.querySelector('#storm-move'),
  stormBanner: document.querySelector('#storm-banner'),
  modeSeg: document.querySelector('#mode-seg'),
  radWeight: document.querySelector('#rad-weight'),
  slopeWeight: document.querySelector('#slope-weight'),
  dustWeight: document.querySelector('#dust-weight'),
  maxSlope: document.querySelector('#max-slope'),
  exaggeration: document.querySelector('#exaggeration'),
  legend: document.querySelector('#legend'),
  legendNote: document.querySelector('#legend-note'),
  destMode: document.querySelector('#dest-mode'),
  resetView: document.querySelector('#reset-view'),
  alert: document.querySelector('#route-alert'),
  tradeoff: document.querySelector('#tradeoff'),
  customCard: document.querySelector('#custom-card'),
  customMetrics: document.querySelector('#custom-metrics'),
  activeLabel: document.querySelector('#active-label'),
  simBtn: document.querySelector('#sim-btn'),
  playback: document.querySelector('#playback'),
  simBar: document.querySelector('#sim-bar'),
  simProgress: document.querySelector('#sim-progress'),
  simDistance: document.querySelector('#sim-distance'),
  simTime: document.querySelector('#sim-time'),
  simRad: document.querySelector('#sim-rad'),
  simO2: document.querySelector('#sim-o2'),
  simKwh: document.querySelector('#sim-kwh'),
  simHere: document.querySelector('#sim-here'),
  status: document.querySelector('#status-line'),
  tip: document.querySelector('#tooltip'),
  colSafe: document.querySelector('#col-safe'),
  colFast: document.querySelector('#col-fast'),
  roverSeg: document.querySelector('#rover-seg'),
  exportPlan: document.querySelector('#export-plan'),
};

const ROVER_PROFILES = {
  cargo: { maxSlope: 15, wSlope: 4.5, wDust: 3.5, wRad: 2.5 },
  scout: { maxSlope: 25, wSlope: 2.5, wDust: 1.5, wRad: 4.5 },
};

const SITES = {
  jezero: { label: 'Jezero Crater' },
  valles: { label: 'Valles Marineris' },
  arcadia: { label: 'Arcadia Planitia' },
  gale: { label: 'Gale Crater' },
  olympus: { label: 'Olympus Mons' },
};

const state = {
  site: 'jezero',
  mode: 'safest',
  layer: 'surface',
  wRad: 4,
  wSlope: 3.5,
  wDust: 2,
  maxSlope: 25,
  exaggeration: 3,
  destinationMode: false,
  roverProfile: null,
  switching: false,
  start: null,
  goal: null,
};

const storm = { active: false, col: 75, row: 75, radius: 22 };
const session = { view: null };
const siteCache = new Map();

const routes = { safest: null, fastest: null, custom: null };
let terrainRef = null;
const sim = { playing: false, u: 0, elapsed: 0 };
const pointer = new THREE.Vector2();
let pointerInside = false;
let pointerDown = null;
let frameId = 0;
let applyingPreset = false;

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

boot().catch((error) => {
  fail(error?.message || 'The terrain matrix could not be loaded.');
  console.error(error);
});

async function boot() {
  const loaded = await loadSite('jezero');
  terrainRef = loaded.terrain;
  ui.loadingMessage.textContent = 'Meshing the terrain and opening both corridors…';
  session.view = createView(loaded.terrain, loaded.texture);
  bindUi();
  adoptDefaults(loaded.terrain);
  recompute(terrainRef, session.view);
  paintChrome(loaded.terrain);
  ui.loading.hidden = true;
  session.view.start();
}

async function loadSite(id) {
  if (siteCache.has(id)) return siteCache.get(id);
  const response = await fetch(`${BASE}data/${id}_terrain.json`);
  if (!response.ok) {
    throw new Error(
      `Missing data/${id}_terrain.json (${response.status}). Run: python scripts/fetch_mars_data.py`,
    );
  }
  const terrain = prepareTerrain(await response.json());
  const texture = await new THREE.TextureLoader().loadAsync(`${BASE}data/${id}_texture.jpg`);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  const loaded = { terrain, texture };
  siteCache.set(id, loaded);
  return loaded;
}

function reliefScale(terrain) {
  const relief = Math.max(1, terrain.maxElev - terrain.minElev);
  const metersToWorld = WORLD / (terrain.cols - 1) / terrain.cellSizeMeters;
  const raw = 14 / (relief * metersToWorld);
  return Math.round(Math.min(24, Math.max(1, raw)) * 10) / 10;
}

function adoptDefaults(terrain) {
  const start = terrain.meta.defaultStart;
  const goal = terrain.meta.defaultTarget;
  if (!start || !goal) throw new Error('Terrain file has no default start and target.');
  const recommended = Number(terrain.meta.recommendedExaggeration);
  state.exaggeration = Number.isFinite(recommended) && recommended >= 1
    ? Math.min(24, recommended)
    : reliefScale(terrain);
  ui.exaggeration.value = String(state.exaggeration);
  syncSliderLabels();
  state.start = snapToPassable(terrain, start.row, start.col, state.maxSlope, storm);
  state.goal = snapToPassable(terrain, goal.row, goal.col, state.maxSlope, storm);
}

async function switchSite(id) {
  if (state.switching || id === state.site || !session.view) return;
  state.switching = true;
  ui.loading.hidden = false;
  ui.loadingMessage.textContent = `Acquiring ${SITES[id].label}…`;
  try {
    const loaded = await loadSite(id);
    storm.active = false;
    state.site = id;
    terrainRef = loaded.terrain;
    adoptDefaults(loaded.terrain);
    session.view.replaceTerrain(loaded.terrain, loaded.texture);
    resetSim();
    updateStormUi();
    recompute(terrainRef, session.view);
    paintChrome(loaded.terrain);
    ui.loading.hidden = true;
  } catch (error) {
    ui.loading.hidden = true;
    ui.status.textContent = error?.message || 'That mission site could not be loaded.';
    console.error(error);
  } finally {
    state.switching = false;
  }
}

function paintChrome(terrain) {
  const meta = terrain.meta;
  const live = Boolean(meta.live) || String(meta.dataSource || meta.source || '').includes('Trek');
  ui.siteName.textContent = SITES[state.site]?.label || meta.name || 'Mars';
  document.title = `Martian Map · ${ui.siteName.textContent}`;
  ui.sourcePill.textContent = live ? 'DATA: NASA MARS TREK WMTS (LIVE)' : 'DATA: SEEDED FALLBACK (OFFLINE)';
  ui.sourcePill.classList.toggle('live', live);
  ui.sourcePill.classList.toggle('offline', !live);
  ui.gridPill.textContent = `${terrain.rows} × ${terrain.cols} · ${Math.round(terrain.cellSizeMeters)} m`;
  for (const button of ui.siteSeg.querySelectorAll('button[data-site]')) {
    button.classList.toggle('is-on', button.dataset.site === state.site);
  }
  updateLegend();
}

function fail(message) {
  ui.loading.hidden = false;
  ui.loadingMessage.textContent = message;
}

function createView(initialTerrain, initialTexture) {
  let terrain = initialTerrain;
  let surfaceTexture = initialTexture;
  const viewport = document.querySelector('#viewport');
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x1a0d0a, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.12;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.setAttribute('aria-label', 'Jezero terrain');
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a0d0a);
  scene.fog = new THREE.FogExp2(0x1a0d0a, 0.0065);

  const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.15, 2000);
  const homePosition = new THREE.Vector3(72, 48, 78);
  const homeTarget = new THREE.Vector3(0, 1.2, 4);
  camera.position.copy(homePosition);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(homeTarget);
  controls.enableDamping = !reduceMotion;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minPolarAngle = 0.08;
  controls.minDistance = 8;
  controls.maxDistance = 190;
  controls.update();

  scene.add(new THREE.HemisphereLight(0xf0c2a4, 0x2a140e, 0.62));
  const sun = new THREE.DirectionalLight(0xffd2b4, 2.35);
  sun.position.set(-48, 72, 26);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const shadow = 62;
  sun.shadow.camera.left = -shadow;
  sun.shadow.camera.right = shadow;
  sun.shadow.camera.top = shadow;
  sun.shadow.camera.bottom = -shadow;
  sun.shadow.camera.near = 10;
  sun.shadow.camera.far = 180;
  sun.shadow.bias = -0.00015;
  sun.shadow.normalBias = 0.035;
  scene.add(sun);
  scene.add(sun.target);
  const fill = new THREE.DirectionalLight(0x8a5544, 0.45);
  fill.position.set(36, 22, -40);
  scene.add(fill);

  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(900, 28, 20),
    new THREE.MeshBasicMaterial({ map: skyTexture(), side: THREE.BackSide, fog: false, depthWrite: false }),
  );
  scene.add(sky);

  const grid = new THREE.GridHelper(130, 26, 0x8d5340, 0x3b241e);
  grid.position.y = -0.45;
  scene.add(grid);
  addCompass(scene);

  let geometry = buildGeometry(terrain);
  const anisotropy = renderer.capabilities.getMaxAnisotropy();
  surfaceTexture.anisotropy = anisotropy;
  const slopeMaps = { current: createSlopeTexture(terrain, state.maxSlope) };
  let riskTexture = createRiskTexture(terrain);
  slopeMaps.current.anisotropy = anisotropy;
  riskTexture.anisotropy = anisotropy;

  const material = new THREE.MeshStandardMaterial({
    map: surfaceTexture,
    roughness: 0.94,
    metalness: 0.03,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);

  const paths = new THREE.Group();
  scene.add(paths);
  const dashTexture = makeDashTexture();

  const startBeacon = makeBeacon(0x3dff9a);
  const goalBeacon = makeBeacon(0x3ee0ff);
  const reticle = makeReticle();
  const rover = makeRover();
  const stormGroup = makeStorm();
  scene.add(startBeacon, goalBeacon, reticle, rover, stormGroup);

  const raycaster = new THREE.Raycaster();
  const timer = new THREE.Timer();
  timer.connect(document);

  function verticalScale() {
    return (WORLD / (terrain.cols - 1) / terrain.cellSizeMeters) * state.exaggeration;
  }

  function cellToWorld(row, col, lift = 0) {
    const x = (col / (terrain.cols - 1) - 0.5) * WORLD;
    const z = (row / (terrain.rows - 1) - 0.5) * WORLD;
    const elev = terrain.elevation[row * terrain.cols + col];
    const y = (elev - terrain.minElev) * verticalScale() + lift;
    return new THREE.Vector3(x, y, z);
  }

  function worldToCell(x, z) {
    return {
      col: (x / WORLD + 0.5) * (terrain.cols - 1),
      row: (z / WORLD + 0.5) * (terrain.rows - 1),
    };
  }

  function refreshHeights() {
    const position = geometry.attributes.position;
    const scale = verticalScale();
    for (let i = 0; i < terrain.n; i += 1) {
      position.setY(i, (terrain.elevation[i] - terrain.minElev) * scale);
    }
    position.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  }

  function placeBeacons() {
    const a = cellToWorld(state.start.row, state.start.col, 0);
    const b = cellToWorld(state.goal.row, state.goal.col, 0);
    startBeacon.position.copy(a);
    goalBeacon.position.copy(b);
  }

  function setPaths(next) {
    clearGroup(paths);
    const selected = state.mode;
    addTube(next.safest, {
      color: 0x49f0ff,
      radius: 0.16,
      lift: 0.42,
      opacity: selected === 'fastest' ? 0.72 : 1,
      glow: true,
    });
    addDashed(next.fastest, 0.78);
    if (state.mode === 'custom') {
      addTube(next.custom, {
        color: 0xd7c4ff,
        radius: 0.11,
        lift: 0.58,
        opacity: 1,
        glow: false,
      });
    }
  }

  function addTube(route, style) {
    if (!route?.ok || route.path.length < 2) return;
    const curve = pathCurve(route.path, style.lift);
    if (!curve) return;
    const segments = Math.max(48, curve.points?.length ?? route.path.length * 2);
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, segments, style.radius, 6, false),
      new THREE.MeshStandardMaterial({
        color: style.color,
        emissive: style.color,
        emissiveIntensity: style.opacity > 0.9 ? 1.05 : 0.45,
        roughness: 0.32,
        metalness: 0.08,
        toneMapped: false,
      }),
    );
    tube.castShadow = false;
    paths.add(tube);
    if (style.glow) {
      paths.add(
        new THREE.Mesh(
          new THREE.TubeGeometry(curve, segments, style.radius * 2.1, 6, false),
          new THREE.MeshBasicMaterial({
            color: style.color,
            transparent: true,
            opacity: 0.16,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            toneMapped: false,
          }),
        ),
      );
    }
  }

  function addDashed(route, lift) {
    if (!route?.ok || route.path.length < 2) return;
    const curve = pathCurve(route.path, lift);
    if (!curve) return;
    const length = curve.getLength();
    dashTexture.repeat.set(Math.max(8, length / 1.35), 1);
    const mesh = new THREE.Mesh(
      new THREE.TubeGeometry(curve, Math.max(24, curve.points.length), 0.08, 5, false),
      new THREE.MeshStandardMaterial({
        color: 0xffbf3c,
        emissive: 0xffa000,
        emissiveIntensity: state.mode === 'fastest' ? 1.2 : 0.65,
        map: dashTexture,
        alphaTest: 0.45,
        roughness: 0.4,
        toneMapped: false,
      }),
    );
    paths.add(mesh);
  }

  function pathCurve(path, lift) {
    const points = [];
    for (const cell of path) {
      const point = cellToWorld(cell.row, cell.col, lift);
      const previous = points[points.length - 1];
      if (!previous || previous.distanceToSquared(point) > 1e-8) points.push(point);
    }
    if (points.length < 2) return null;
    const smooth = relaxPath(points);
    const curve = new THREE.CatmullRomCurve3(smooth, false, 'centripetal');
    curve.arcLengthDivisions = Math.max(80, smooth.length);
    curve.points = smooth;
    return curve;
  }

  function parkRover() {
    if (!state.start) return;
    const spot = cellToWorld(state.start.row, state.start.col, 0.2);
    rover.position.copy(spot);
    rover.visible = true;
    if (state.goal) {
      const goal = cellToWorld(state.goal.row, state.goal.col, 0.2);
      const tangent = goal.sub(spot);
      if (tangent.lengthSq() > 1e-6) rover.rotation.y = Math.atan2(tangent.x, tangent.z);
    }
  }

  function sampleBilinear(field, row, col) {
    const r = THREE.MathUtils.clamp(row, 0, terrain.rows - 1);
    const c = THREE.MathUtils.clamp(col, 0, terrain.cols - 1);
    const r0 = Math.floor(r);
    const c0 = Math.floor(c);
    const r1 = Math.min(r0 + 1, terrain.rows - 1);
    const c1 = Math.min(c0 + 1, terrain.cols - 1);
    const tr = r - r0;
    const tc = c - c0;
    const v00 = field[r0 * terrain.cols + c0];
    const v10 = field[r0 * terrain.cols + c1];
    const v01 = field[r1 * terrain.cols + c0];
    const v11 = field[r1 * terrain.cols + c1];
    return (v00 * (1 - tc) + v10 * tc) * (1 - tr) + (v01 * (1 - tc) + v11 * tc) * tr;
  }

  function hover(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    const flipX = event.clientX > window.innerWidth - 250;
    const flipY = event.clientY > window.innerHeight - 170;
    ui.tip.style.left = `${event.clientX}px`;
    ui.tip.style.top = `${event.clientY}px`;
    ui.tip.style.transform = `translate(${flipX ? 'calc(-100% - 14px)' : '14px'}, ${flipY ? 'calc(-100% - 14px)' : '14px'})`;
  }

  function updateHover() {
    if (!pointerInside) {
      reticle.visible = false;
      ui.tip.hidden = true;
      return;
    }
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(mesh, false)[0];
    if (!hit) {
      reticle.visible = false;
      ui.tip.hidden = true;
      return;
    }
    const cell = worldToCell(hit.point.x, hit.point.z);
    const slope = sampleBilinear(terrain.slope, cell.row, cell.col);
    const influence = stormInfluence(storm, cell.row, cell.col);
    const radiation = sampleBilinear(terrain.radiation, cell.row, cell.col) + 2.5 * influence.intensity;
    const dust = Math.min(1, sampleBilinear(terrain.dust, cell.row, cell.col) + 0.9 * influence.intensity);
    const elev = terrain.minElev + hit.point.y / verticalScale();
    const lat = terrain.bbox.north - (cell.row / (terrain.rows - 1)) * (terrain.bbox.north - terrain.bbox.south);
    const lon = terrain.bbox.west + (cell.col / (terrain.cols - 1)) * (terrain.bbox.east - terrain.bbox.west);
    const passable = slope <= state.maxSlope && !influence.blocked;
    const stormLine = influence.blocked
      ? '<br>Storm core · zero visibility'
      : influence.intensity > 0.02
        ? `<br>Storm intensity ${influence.intensity.toFixed(2)}`
        : '';
    reticle.visible = true;
    reticle.position.set(hit.point.x, hit.point.y + 0.08 + slope / 180, hit.point.z);
    ui.tip.hidden = false;
    ui.tip.innerHTML = `
      <b>SURFACE SAMPLE</b>
      ${formatLat(lat)} &nbsp; ${formatLon(lon)}<br>
      Elev ${formatNumber(elev, 0)} m<br>
      Slope ${slope.toFixed(1)}% <span class="${passable ? 'good' : 'bad'}">${passable ? 'PASSABLE' : 'IMPASSABLE'}</span><br>
      Rad ${radiation.toFixed(3)} mSv/day<br>
      Dust slip ${dust.toFixed(2)}
      ${stormLine}
    `;
  }

  function pickCell(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(mesh, false)[0];
    if (!hit) return null;
    const cell = worldToCell(hit.point.x, hit.point.z);
    return snapToPassable(terrain, cell.row, cell.col, state.maxSlope, storm);
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    pointerDown = { x: event.clientX, y: event.clientY, shift: event.shiftKey };
  }

  function onPointerUp(event) {
    if (!pointerDown || event.button !== 0) return;
    const dx = event.clientX - pointerDown.x;
    const dy = event.clientY - pointerDown.y;
    const shifted = pointerDown.shift || event.shiftKey;
    pointerDown = null;
    if (dx * dx + dy * dy > 25) return;
    const cell = pickCell(event);
    if (!cell) return;
    const setGoal = shifted || state.destinationMode;
    if (setGoal) state.goal = cell;
    else state.start = cell;
    sim.playing = false;
    sim.u = 0;
    sim.elapsed = 0;
    recompute(terrain, viewApi);
  }

  function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  renderer.domElement.addEventListener('pointermove', (event) => {
    pointerInside = true;
    hover(event);
  });
  renderer.domElement.addEventListener('pointerleave', () => {
    pointerInside = false;
  });
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  renderer.domElement.addEventListener('pointerup', onPointerUp);
  window.addEventListener('resize', onResize);

  function frame(timestamp) {
    frameId = requestAnimationFrame(frame);
    timer.update(timestamp);
    const dt = timer.getDelta();
    const t = timer.getElapsed();
    controls.update();
    if (!reduceMotion) {
      const pulse = 1 + Math.sin(t * 3) * 0.08;
      startBeacon.userData.head.scale.setScalar(pulse);
      goalBeacon.userData.head.scale.setScalar(1 + Math.sin(t * 3 + 1) * 0.08);
      reticle.scale.setScalar(1 + Math.sin(t * 5) * 0.06);
      if (storm.active) {
        const stormPulse = 0.5 + 0.5 * Math.sin(t * 2.6);
        stormGroup.userData.dome.material.opacity = 0.16 + stormPulse * 0.18;
        stormGroup.userData.halo.material.opacity = 0.05 + stormPulse * 0.08;
        stormGroup.userData.ring.material.opacity = 0.4 + stormPulse * 0.45;
        stormGroup.userData.core.material.opacity = 0.55 + stormPulse * 0.4;
      }
    }
    updateHover();
    if (sim.playing) stepSim(dt);
    renderer.render(scene, camera);
  }

  function stepSim(dt) {
    const route = activeRoute();
    if (!route?.ok || route.path.length < 2 || route.timeHours <= 0) {
      sim.playing = false;
      return;
    }
    const duration = 36 / Number(ui.playback.value || 3);
    sim.elapsed += dt;
    sim.u = Math.min(1, sim.elapsed / duration);
    poseRover(route, sim.u);
    renderSim(route, sim.u);
    if (sim.u >= 1) {
      sim.playing = false;
      ui.simBtn.textContent = 'Replay traverse';
    }
  }

  function poseRover(route, progress) {
    const hours = progress * route.timeHours;
    const cum = route.cum.hours;
    let index = 1;
    while (index < cum.length - 1 && cum[index] < hours) index += 1;
    const span = cum[index] - cum[index - 1] || 1;
    const f = THREE.MathUtils.clamp((hours - cum[index - 1]) / span, 0, 1);
    const a = route.path[index - 1];
    const b = route.path[index];
    const pa = cellToWorld(a.row, a.col, 0.22);
    const pb = cellToWorld(b.row, b.col, 0.22);
    rover.position.set(pa.x + (pb.x - pa.x) * f, pa.y + (pb.y - pa.y) * f, pa.z + (pb.z - pa.z) * f);
    const tx = pb.x - pa.x;
    const tz = pb.z - pa.z;
    if (tx * tx + tz * tz > 1e-8) rover.rotation.y = Math.atan2(tx, tz);
    rover.userData.spin += Math.hypot(tx, tz) * f * 0.4;
    for (const wheel of rover.userData.wheels) wheel.rotation.x = rover.userData.spin;
  }

  const viewApi = {
    start() {
      refreshHeights();
      placeBeacons();
      placeStorm();
      parkRover();
      frame();
    },
    refreshHeights,
    placeBeacons,
    setPaths,
    parkRover,
    poseRover,
    resetCamera() {
      camera.position.copy(homePosition);
      controls.target.copy(homeTarget);
      controls.update();
    },
    rebuildSlope() {
      const next = createSlopeTexture(terrain, state.maxSlope);
      next.anisotropy = anisotropy;
      slopeMaps.current.dispose();
      slopeMaps.current = next;
      if (state.layer === 'slope') applyLayer();
    },
    applyLayer,
    placeStorm,
    replaceTerrain,
  };

  function placeStorm() {
    stormGroup.visible = storm.active;
    if (!storm.active) return;
    const radiusWorld = (storm.radius / (terrain.cols - 1)) * WORLD;
    const center = cellToWorld(storm.row, storm.col, 0.25);
    stormGroup.position.set(center.x, center.y, center.z);
    stormGroup.scale.set(radiusWorld, radiusWorld * 0.62, radiusWorld);
  }

  function replaceTerrain(nextTerrain, nextTexture) {
    const sameGrid = nextTerrain.rows === terrain.rows && nextTerrain.cols === terrain.cols;
    terrain = nextTerrain;
    terrainRef = nextTerrain;
    surfaceTexture.dispose();
    surfaceTexture = nextTexture;
    surfaceTexture.colorSpace = THREE.SRGBColorSpace;
    surfaceTexture.anisotropy = anisotropy;
    slopeMaps.current.dispose();
    riskTexture.dispose();
    slopeMaps.current = createSlopeTexture(terrain, state.maxSlope);
    riskTexture = createRiskTexture(terrain);
    slopeMaps.current.anisotropy = anisotropy;
    riskTexture.anisotropy = anisotropy;
    if (!sameGrid) {
      geometry.dispose();
      geometry = buildGeometry(terrain);
      mesh.geometry = geometry;
    }
    refreshHeights();
    applyLayer();
    placeStorm();
    renderer.domElement.setAttribute('aria-label', `${terrain.meta.name || 'Mars'} terrain`);
  }

  function applyLayer() {
    const map = state.layer === 'slope' ? slopeMaps.current : state.layer === 'risk' ? riskTexture : surfaceTexture;
    material.map = map;
    material.emissiveMap = state.layer === 'surface' ? null : map;
    material.emissive.set(state.layer === 'surface' ? 0x000000 : 0xffffff);
    material.emissiveIntensity = state.layer === 'surface' ? 0 : 0.28;
    material.needsUpdate = true;
  }

  window.__marsDispose = () => {
    cancelAnimationFrame(frameId);
    renderer.domElement.removeEventListener('pointerdown', onPointerDown);
    renderer.domElement.removeEventListener('pointerup', onPointerUp);
    window.removeEventListener('resize', onResize);
    controls.dispose();
    timer.dispose();
    renderer.dispose();
    renderer.domElement.remove();
  };

  return viewApi;
}

function bindUi() {
  const view = () => session.view;
  ui.modeSeg.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-mode]');
    if (!button) return;
    applyingPreset = true;
    state.mode = button.dataset.mode;
    state.roverProfile = null;
    if (state.mode !== 'custom') {
      const preset = PRESETS[state.mode];
      state.wRad = preset.wRad;
      state.wSlope = preset.wSlope;
      state.wDust = preset.wDust;
      ui.radWeight.value = String(preset.wRad);
      ui.slopeWeight.value = String(preset.wSlope);
      ui.dustWeight.value = String(preset.wDust);
    }
    applyingPreset = false;
    syncSliderLabels();
    markModes();
    markProfiles();
    resetSim();
    recompute(terrainRef, view());
  });

  ui.roverSeg.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-profile]');
    if (!button || !terrainRef) return;
    applyRoverProfile(button.dataset.profile, view());
  });

  for (const [input, key] of [
    [ui.radWeight, 'wRad'],
    [ui.slopeWeight, 'wSlope'],
    [ui.dustWeight, 'wDust'],
  ]) {
    input.addEventListener('input', () => {
      state[key] = Number(input.value);
      if (!applyingPreset) {
        state.mode = 'custom';
        state.roverProfile = null;
      }
      syncSliderLabels();
      markModes();
      markProfiles();
      scheduleRecompute(terrainRef, view());
    });
  }

  ui.maxSlope.addEventListener('input', () => {
    state.maxSlope = Number(ui.maxSlope.value);
    if (!applyingPreset) state.roverProfile = null;
    syncSliderLabels();
    markProfiles();
    state.start = snapToPassable(terrainRef, state.start.row, state.start.col, state.maxSlope, storm);
    state.goal = snapToPassable(terrainRef, state.goal.row, state.goal.col, state.maxSlope, storm);
    view().rebuildSlope();
    updateLegend();
    resetSim();
    scheduleRecompute(terrainRef, view());
  });

  ui.exaggeration.addEventListener('input', () => {
    state.exaggeration = Number(ui.exaggeration.value);
    syncSliderLabels();
    view().refreshHeights();
    view().placeBeacons();
    view().placeStorm();
    view().setPaths(routes);
    if (!sim.playing) view().parkRover();
  });

  document.querySelector('#layer-seg').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-layer]');
    if (!button) return;
    state.layer = button.dataset.layer;
    for (const item of document.querySelectorAll('#layer-seg button')) {
      item.classList.toggle('is-on', item === button);
    }
    updateLegend();
    view().applyLayer();
  });

  ui.siteSeg.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-site]');
    if (!button) return;
    switchSite(button.dataset.site);
  });

  ui.stormBtn.addEventListener('click', () => {
    if (storm.active) {
      storm.active = false;
    } else {
      const mid = pathMidpoint(routes.safest);
      storm.active = true;
      storm.row = mid.row;
      storm.col = mid.col;
      storm.radius = 22;
    }
    settleStorm();
  });

  ui.stormMove.addEventListener('click', () => {
    if (!storm.active) return;
    const mid = pathMidpoint(routes.safest);
    storm.row = mid.row;
    storm.col = mid.col;
    settleStorm();
  });

  ui.destMode.addEventListener('click', () => {
    state.destinationMode = !state.destinationMode;
    ui.destMode.setAttribute('aria-pressed', String(state.destinationMode));
    ui.destMode.textContent = state.destinationMode ? 'Setting destination' : 'Set destination';
  });

  ui.resetView.addEventListener('click', () => view().resetCamera());

  ui.exportPlan.addEventListener('click', () => exportMissionPlan());

  ui.simBtn.addEventListener('click', () => {
    const route = activeRoute();
    if (!route?.ok || route.path.length < 2) return;
    if (sim.playing) {
      sim.playing = false;
      ui.simBtn.textContent = 'Resume traverse';
      return;
    }
    if (sim.u >= 1) {
      sim.u = 0;
      sim.elapsed = 0;
    }
    sim.playing = true;
    ui.simBtn.textContent = 'Pause traverse';
  });

  syncSliderLabels();
  markModes();
  markProfiles();
  updateLegend();
  updateStormUi();
}

function applyRoverProfile(id, view) {
  const profile = ROVER_PROFILES[id];
  if (!profile || !terrainRef) return;
  applyingPreset = true;
  state.roverProfile = id;
  state.mode = 'custom';
  state.maxSlope = profile.maxSlope;
  state.wRad = profile.wRad;
  state.wSlope = profile.wSlope;
  state.wDust = profile.wDust;
  ui.maxSlope.value = String(profile.maxSlope);
  ui.radWeight.value = String(profile.wRad);
  ui.slopeWeight.value = String(profile.wSlope);
  ui.dustWeight.value = String(profile.wDust);
  applyingPreset = false;
  syncSliderLabels();
  markModes();
  markProfiles();
  state.start = snapToPassable(terrainRef, state.start.row, state.start.col, state.maxSlope, storm);
  state.goal = snapToPassable(terrainRef, state.goal.row, state.goal.col, state.maxSlope, storm);
  view.rebuildSlope();
  updateLegend();
  resetSim();
  recompute(terrainRef, view);
}

function markProfiles() {
  for (const button of ui.roverSeg.querySelectorAll('button')) {
    button.classList.toggle('is-on', button.dataset.profile === state.roverProfile);
  }
}

function settleStorm() {
  if (storm.active && terrainRef) {
    state.start = snapToPassable(terrainRef, state.start.row, state.start.col, state.maxSlope, storm);
    state.goal = snapToPassable(terrainRef, state.goal.row, state.goal.col, state.maxSlope, storm);
  }
  session.view?.placeStorm();
  resetSim();
  if (terrainRef && session.view) recompute(terrainRef, session.view);
  updateStormUi();
}

function pathMidpoint(route) {
  if (!route?.path?.length) {
    return {
      row: Math.floor((terrainRef?.rows || 150) / 2),
      col: Math.floor((terrainRef?.cols || 150) / 2),
    };
  }
  return route.path[Math.floor((route.path.length - 1) / 2)];
}

function updateStormUi() {
  ui.stormBtn.setAttribute('aria-pressed', String(storm.active));
  ui.stormBtn.textContent = storm.active ? '✓ Clear Storm Hazard' : '⚡ Trigger Solar / Dust Storm';
  ui.stormMove.disabled = !storm.active;
  ui.stormBanner.hidden = !storm.active;
}

let recomputeFrame = 0;
function scheduleRecompute(terrain, view) {
  cancelAnimationFrame(recomputeFrame);
  recomputeFrame = requestAnimationFrame(() => recompute(terrain, view));
}

function recompute(terrain, view) {
  const weights = { wRad: state.wRad, wSlope: state.wSlope, wDust: state.wDust };
  const activeStorm = storm.active ? storm : null;
  routes.safest = evaluateRoute(
    terrain,
    findPath(terrain, state.start, state.goal, PRESETS.safest, state.maxSlope, activeStorm).path,
    activeStorm,
  );
  routes.fastest = evaluateRoute(
    terrain,
    findPath(terrain, state.start, state.goal, PRESETS.fastest, state.maxSlope, activeStorm).path,
    activeStorm,
  );
  routes.custom = evaluateRoute(
    terrain,
    findPath(terrain, state.start, state.goal, weights, state.maxSlope, activeStorm).path,
    activeStorm,
  );
  view.placeStorm();
  view.setPaths(routes);
  view.placeBeacons();
  if (!sim.playing) view.parkRover();
  renderComparison();
  renderSim(activeRoute(), sim.u);
}

function activeRoute() {
  if (state.mode === 'fastest') return routes.fastest;
  if (state.mode === 'custom') return routes.custom;
  return routes.safest;
}

function renderComparison() {
  fillColumn('safe', routes.safest);
  fillColumn('fast', routes.fastest);
  const blocked = !routes.safest.ok || !routes.fastest.ok;
  ui.alert.hidden = !blocked;
  ui.alert.textContent = blocked
    ? 'No viable corridor at the current slope limit. Raise the limit or move a beacon off a cliff.'
    : '';
  ui.tradeoff.textContent = tradeoffText(routes.safest, routes.fastest);
  const customOn = state.mode === 'custom';
  ui.customCard.hidden = !customOn;
  if (customOn) {
    const route = routes.custom;
    ui.customMetrics.innerHTML = route.ok
      ? metricChips(route)
      : '<div><dt>Route</dt><dd>Blocked</dd></div>';
  }
  ui.colSafe.className = state.mode === 'safest' ? 'col-on' : '';
  ui.colFast.className = state.mode === 'fastest' ? 'col-on col-fast' : '';
  ui.activeLabel.textContent =
    state.mode === 'fastest' ? 'Fastest route' : state.mode === 'custom' ? 'Custom weights' : 'Safest EVA';
  const route = activeRoute();
  ui.simBtn.disabled = !route?.ok || route.path.length < 2;
  if (!sim.playing && sim.u <= 0) ui.simBtn.textContent = 'Simulate EVA traverse';
  const both = routes.safest.ok && routes.fastest.ok;
  const stormNote = storm.active ? ' Storm reroute is live.' : '';
  ui.status.textContent = both
    ? `Corridors ready. Safest ${routes.safest.distanceKm.toFixed(2)} km, fastest ${routes.fastest.distanceKm.toFixed(2)} km.${stormNote}`
    : 'Waiting on a passable corridor.';
}

function fillColumn(prefix, route) {
  const cells = {
    distance: document.querySelector(`#${prefix}-distance`),
    time: document.querySelector(`#${prefix}-time`),
    rad: document.querySelector(`#${prefix}-rad`),
    o2: document.querySelector(`#${prefix}-o2`),
    kwh: document.querySelector(`#${prefix}-kwh`),
    score: document.querySelector(`#${prefix}-score`),
  };
  if (!route?.ok) {
    for (const cell of Object.values(cells)) cell.textContent = '—';
    cells.score.className = 'score';
    return;
  }
  cells.distance.textContent = `${route.distanceKm.toFixed(2)} km`;
  cells.time.textContent = `${route.timeHours.toFixed(2)} h`;
  cells.rad.textContent = `${route.radiationMsv.toFixed(3)} mSv`;
  cells.o2.textContent = `${route.oxygenKg.toFixed(2)} kg`;
  cells.kwh.textContent = `${route.batteryKwh.toFixed(2)} kWh`;
  cells.score.textContent = String(route.safetyScore);
  cells.score.className = `score ${scoreClass(route.safetyScore)}`;
}

function metricChips(route) {
  return [
    ['Distance', `${route.distanceKm.toFixed(2)} km`],
    ['Time', `${route.timeHours.toFixed(2)} h`],
    ['Dose', `${route.radiationMsv.toFixed(3)}`],
    ['O2', `${route.oxygenKg.toFixed(2)} kg`],
    ['Battery', `${route.batteryKwh.toFixed(2)}`],
    ['Safety', String(route.safetyScore)],
  ]
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`)
    .join('');
}

function renderSim(route, progress) {
  ui.simBar.style.width = `${Math.round(progress * 100)}%`;
  if (!route?.ok || !route.cum || route.path.length < 2) {
    ui.simProgress.textContent = 'Parked at start';
    ui.simDistance.textContent = '—';
    ui.simTime.textContent = '—';
    ui.simRad.textContent = '—';
    ui.simO2.textContent = '—';
    ui.simKwh.textContent = '—';
    ui.simHere.textContent = '—';
    return;
  }
  const sample = sampleCum(route, progress);
  ui.simProgress.textContent = progress <= 0 ? 'Parked at start' : `${Math.round(progress * 100)}%`;
  ui.simDistance.textContent = `${sample.dist.toFixed(2)} / ${route.distanceKm.toFixed(2)} km`;
  ui.simTime.textContent = `${sample.hours.toFixed(2)} / ${route.timeHours.toFixed(2)} h`;
  ui.simRad.textContent = `${sample.rad.toFixed(3)} mSv`;
  ui.simO2.textContent = `${sample.o2.toFixed(2)} kg`;
  ui.simKwh.textContent = `${sample.kwh.toFixed(2)} kWh`;
  ui.simHere.textContent = `slope ${sample.slope.toFixed(1)}% · ${sample.speed.toFixed(2)} m/s · ${sample.radiation.toFixed(2)} mSv/day`;
}

function sampleCum(route, progress) {
  const origin = route.path[0];
  if (progress <= 0) {
    return {
      dist: 0,
      hours: 0,
      rad: 0,
      o2: 0,
      kwh: 0,
      slope: cellHazard(terrainRef, origin.row, origin.col, storm.active ? storm : null).slope,
      radiation: cellHazard(terrainRef, origin.row, origin.col, storm.active ? storm : null).radiation,
      speed: route.cum.speed?.[0] ?? 2.4,
    };
  }
  const hours = progress * route.timeHours;
  const cum = route.cum.hours;
  let index = 1;
  while (index < cum.length - 1 && cum[index] < hours) index += 1;
  const span = cum[index] - cum[index - 1] || 1;
  const f = THREE.MathUtils.clamp((hours - cum[index - 1]) / span, 0, 1);
  const mix = (array) => array[index - 1] + (array[index] - array[index - 1]) * f;
  const node = route.path[index];
  const hazard = cellHazard(terrainRef, node.row, node.col, storm.active ? storm : null);
  return {
    dist: mix(route.cum.distKm),
    hours: mix(route.cum.hours),
    rad: mix(route.cum.rad),
    o2: mix(route.cum.o2),
    kwh: mix(route.cum.kwh),
    slope: hazard.slope,
    radiation: hazard.radiation,
    speed: route.cum.speed?.[index] ?? 2.4,
  };
}

function tradeoffText(safe, fast) {
  if (!safe?.ok || !fast?.ok) return 'One corridor is blocked at this slope limit.';
  const dKm = fast.distanceKm - safe.distanceKm;
  const dH = fast.timeHours - safe.timeHours;
  const dRad = fast.radiationMsv - safe.radiationMsv;
  return `Fastest route: ${signed(dKm, 'km')} distance, ${signed(dH, 'h')} time, ${signed(dRad, 'mSv')} dose. Safety ${fast.safetyScore} against safest ${safe.safetyScore}.`;
}

function signed(value, unit) {
  const digits = unit === 'mSv' ? 3 : 2;
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(digits)} ${unit}`;
}

function resetSim() {
  sim.playing = false;
  sim.u = 0;
  sim.elapsed = 0;
  ui.simBtn.textContent = 'Simulate EVA traverse';
}

function markModes() {
  ui.modeSeg.dataset.mode = state.mode;
  for (const button of ui.modeSeg.querySelectorAll('button')) {
    button.classList.toggle('is-on', button.dataset.mode === state.mode);
  }
}

function syncSliderLabels() {
  document.querySelector('#rad-weight-val').textContent = state.wRad.toFixed(1);
  document.querySelector('#slope-weight-val').textContent = state.wSlope.toFixed(1);
  document.querySelector('#dust-weight-val').textContent = state.wDust.toFixed(1);
  document.querySelector('#max-slope-val').textContent = `${Math.round(state.maxSlope)}%`;
  document.querySelector('#exaggeration-val').textContent = `${state.exaggeration.toFixed(1)}×`;
}

function updateLegend() {
  ui.legend.classList.remove('legend-hidden', 'legend-slope', 'legend-risk');
  if (state.layer === 'surface') {
    ui.legend.classList.add('legend-hidden');
    ui.legendNote.textContent = 'NASA-style natural color mosaic for this bbox.';
    return;
  }
  if (state.layer === 'slope') {
    ui.legend.classList.add('legend-slope');
    document.querySelector('#legend-left').textContent = 'Flat';
    document.querySelector('#legend-right').textContent = `Cliff ≥ ${Math.round(state.maxSlope)}%`;
    ui.legendNote.textContent = 'Crimson cells are above the rover slope limit.';
    return;
  }
  ui.legend.classList.add('legend-risk');
  document.querySelector('#legend-left').textContent = 'Shielded';
  document.querySelector('#legend-right').textContent = 'Exposed + dusty';
  ui.legendNote.textContent = 'Heat is radiation. A tan wash marks dust slip.';
}

function scoreClass(score) {
  if (score >= 75) return 'good';
  if (score >= 55) return 'mid';
  return 'bad';
}

function cellLatLon(terrain, row, col) {
  const lat = terrain.bbox.north - (row / (terrain.rows - 1)) * (terrain.bbox.north - terrain.bbox.south);
  const lon = terrain.bbox.west + (col / (terrain.cols - 1)) * (terrain.bbox.east - terrain.bbox.west);
  return {
    lat: Number(lat.toFixed(5)),
    lon: Number(lon.toFixed(5)),
  };
}

function routeTelemetry(route) {
  if (!route?.ok) return null;
  return {
    distanceKm: Number(route.distanceKm.toFixed(3)),
    timeHours: Number(route.timeHours.toFixed(3)),
    radiationMsv: Number(route.radiationMsv.toFixed(4)),
    oxygenKg: Number(route.oxygenKg.toFixed(3)),
    batteryKwh: Number(route.batteryKwh.toFixed(3)),
    safetyScore: route.safetyScore,
  };
}

function exportMissionPlan() {
  const terrain = terrainRef;
  const corridor = activeRoute();
  if (!terrain || !corridor) return;
  const meta = terrain.meta || {};
  const briefing = {
    generatedAt: new Date().toISOString(),
    missionSite: {
      id: state.site,
      name: meta.name || SITES[state.site]?.label,
      subtitle: meta.subtitle || '',
      dataSource: meta.dataSource || meta.source || '',
      bbox: terrain.bbox,
      cellSizeMeters: terrain.cellSizeMeters,
      rows: terrain.rows,
      cols: terrain.cols,
    },
    storm: {
      active: storm.active,
      col: storm.col,
      row: storm.row,
      radius: storm.radius,
    },
    activeCorridor: state.mode,
    comparison: {
      safest: routeTelemetry(routes.safest),
      fastest: routeTelemetry(routes.fastest),
    },
    waypoints: (corridor.path || []).map((cell, index) => ({
      index,
      ...cellLatLon(terrain, cell.row, cell.col),
    })),
  };
  const blob = new Blob([`${JSON.stringify(briefing, null, 2)}\n`], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `mars-eva-plan-${state.site}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function formatLat(lat) {
  return `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? 'N' : 'S'}`;
}

function formatLon(lon) {
  return `${Math.abs(lon).toFixed(4)}°${lon >= 0 ? 'E' : 'W'}`;
}

function formatNumber(value, digits) {
  return value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function relaxPath(points) {
  let current = points;
  for (let pass = 0; pass < 2; pass += 1) {
    if (current.length < 3) break;
    const next = [current[0]];
    for (let i = 0; i < current.length - 1; i += 1) {
      const a = current[i];
      const b = current[i + 1];
      next.push(new THREE.Vector3().lerpVectors(a, b, 0.25));
      next.push(new THREE.Vector3().lerpVectors(a, b, 0.75));
    }
    next.push(current[current.length - 1]);
    current = next;
  }
  return current;
}

function clearGroup(group) {
  for (const child of [...group.children]) {
    group.remove(child);
    child.geometry?.dispose();
    if (Array.isArray(child.material)) child.material.forEach((item) => item.dispose());
    else child.material?.dispose();
  }
}

function makeBeacon(color) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.85,
    roughness: 0.35,
    toneMapped: false,
  });
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 1.35, 10), material);
  pole.position.y = 0.75;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 18, 14), material);
  head.position.y = 1.52;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.34, 0.48, 40),
    new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.9, toneMapped: false }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.05;
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.02, 0.1, 8, 8, 1, true),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.14,
      side: THREE.DoubleSide,
      depthWrite: false,
      toneMapped: false,
    }),
  );
  beam.position.y = 4;
  group.add(pole, head, ring, beam);
  group.userData.head = head;
  group.scale.setScalar(2.35);
  return group;
}

function makeReticle() {
  const group = new THREE.Group();
  const material = new THREE.MeshBasicMaterial({
    color: 0xffc56b,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.95,
    toneMapped: false,
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.26, 0.36, 48), material);
  ring.rotation.x = -Math.PI / 2;
  const dot = new THREE.Mesh(new THREE.CircleGeometry(0.045, 16), material);
  dot.rotation.x = -Math.PI / 2;
  group.add(ring, dot);
  group.visible = false;
  return group;
}

function makeRover() {
  const group = new THREE.Group();
  const body = new THREE.MeshStandardMaterial({ color: 0xd9d3c7, roughness: 0.45, metalness: 0.18 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1b1b1b, roughness: 0.7 });
  const gold = new THREE.MeshStandardMaterial({
    color: 0xd08958,
    roughness: 0.4,
    metalness: 0.35,
    emissive: 0x5a2a10,
    emissiveIntensity: 0.25,
  });
  const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.22, 1.2), body);
  chassis.position.y = 0.4;
  chassis.castShadow = true;
  const deck = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.1, 0.7), gold);
  deck.position.set(0, 0.54, -0.05);
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 0.55, 8), gold);
  mast.position.set(0, 0.86, 0.28);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.1, 0.12), dark);
  head.position.set(0, 1.16, 0.32);
  const wheelGeo = new THREE.CylinderGeometry(0.16, 0.16, 0.1, 12);
  wheelGeo.rotateZ(Math.PI / 2);
  const wheels = [];
  for (const z of [-0.4, 0.02, 0.42]) {
    for (const x of [-0.32, 0.32]) {
      const wheel = new THREE.Mesh(wheelGeo, dark);
      wheel.position.set(x, 0.16, z);
      wheel.castShadow = true;
      wheels.push(wheel);
      group.add(wheel);
    }
  }
  group.add(chassis, deck, mast, head);
  group.userData.wheels = wheels;
  group.userData.spin = 0;
  group.scale.setScalar(2.15);
  return group;
}

function makeStorm() {
  const group = new THREE.Group();
  const dome = new THREE.Mesh(
    new THREE.SphereGeometry(1, 48, 28, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshBasicMaterial({
      color: 0xff3a2a,
      transparent: true,
      opacity: 0.28,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }),
  );
  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(1.12, 32, 18, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshBasicMaterial({
      color: 0xffb03a,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }),
  );
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.96, 1.05, 80),
    new THREE.MeshBasicMaterial({
      color: 0xffc14a,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.03;
  const core = new THREE.Mesh(
    new THREE.RingGeometry(0.4, 0.48, 64),
    new THREE.MeshBasicMaterial({
      color: 0xff2448,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }),
  );
  core.rotation.x = -Math.PI / 2;
  core.position.y = 0.05;
  group.add(dome, halo, ring, core);
  group.userData = { dome, halo, ring, core };
  group.visible = false;
  return group;
}

function makeDashTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 8;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 64, 8);
  ctx.fillStyle = '#ffbf3c';
  ctx.fillRect(0, 0, 28, 8);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  return texture;
}

function skyTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 8;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 512);
  gradient.addColorStop(0, '#1a0d0a');
  gradient.addColorStop(0.38, '#1a0d0a');
  gradient.addColorStop(0.5, '#a84b2c');
  gradient.addColorStop(0.62, '#1a0d0a');
  gradient.addColorStop(1, '#120806');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 8, 512);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function addCompass(scene) {
  const labels = [
    ['N', 0, -WORLD / 2 - 5],
    ['S', 0, WORLD / 2 + 5],
    ['E', WORLD / 2 + 5, 0],
    ['W', -WORLD / 2 - 5, 0],
  ];
  for (const [text, x, z] of labels) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffd2b4';
    ctx.font = '700 80px Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 64);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }),
    );
    sprite.position.set(x, 2.4, z);
    sprite.scale.set(7, 3.5, 1);
    scene.add(sprite);
  }
}

function buildGeometry(terrain) {
  const { rows, cols } = terrain;
  const geometry = new THREE.BufferGeometry();
  const position = new Float32Array(terrain.n * 3);
  const uv = new Float32Array(terrain.n * 2);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i = r * cols + c;
      position[i * 3] = (c / (cols - 1) - 0.5) * WORLD;
      position[i * 3 + 1] = 0;
      position[i * 3 + 2] = (r / (rows - 1) - 0.5) * WORLD;
      uv[i * 2] = c / (cols - 1);
      uv[i * 2 + 1] = 1 - r / (rows - 1);
    }
  }
  const index = new Uint32Array((rows - 1) * (cols - 1) * 6);
  let cursor = 0;
  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const a = r * cols + c;
      const b = a + 1;
      const d = a + cols;
      const e = d + 1;
      index[cursor++] = a;
      index[cursor++] = d;
      index[cursor++] = b;
      index[cursor++] = b;
      index[cursor++] = d;
      index[cursor++] = e;
    }
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();
  return geometry;
}
