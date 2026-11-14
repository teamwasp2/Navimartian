"""Download NASA Mars Trek WMTS tiles and export five EVA terrain packs.

Equirectangular tile math (default028mm, not Web Mercator), zoom z:

    columns = 2 ** (z + 1)
    rows    = 2 ** z
    tx = ((lon + 180) / 360) * columns
    ty = ((90 - lat) / 180) * rows

Zoom 8 is requested first (~230 m/px). Layers whose pyramid stops earlier
are fetched at the highest published zoom and the terminal says so.

Color drapes come from the Viking MDIM 2.1 mosaic. The elevation grid is
the grayscale MOLA DEM luminance, scaled onto each region's physical
elevation envelope, then smoothed with a 5-tap Gaussian (sigma 1.15) so the
zoom-5 DEM does not stair-step under vertical exaggeration. The MOLA/HRSC
shaded-relief layer is downloaded as the specified relief product and used
when the DEM tiles are unavailable. If Trek cannot be reached, a seeded
terrain is written for that region only.

Slope is percent grade from central finite differences. Radiation starts at
the region's baseline (Curiosity RAD is 0.64 mSv/day at Jezero) and rises
where the atmospheric column is thinner and on exposed ridges. Dust slip is
higher on flat, low basins.
"""

from __future__ import annotations

import heapq
import io
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "public" / "data"
ROWS = 150
COLS = 150
PREFERRED_ZOOM = 8
TILE_PX = 256
REQUEST_TIMEOUT = 15
MARS_RADIUS_M = 3_389_500.0
USER_AGENT = "NASASpaceApps2026-MartianMap/1.1 (educational eva-route-planner)"

TILE_URL = (
    "https://trek.nasa.gov/tiles/Mars/EQ/{layer}/1.0.0/default/default028mm/{z}/{y}/{x}.{ext}"
)

TEXTURE_LAYERS = (
    ("Mars_Viking_MDIM21_ClrMosaic_global_232m", "jpg"),
)
# Specified relief layer, then the brief's fallback name, then a live color-shade alias.
RELIEF_LAYERS = (
    ("Mars_MOLA_blend200ppx_HRSC_Shade_clon0dd_200mpp_lzw", "jpg"),
    ("Mars_MOLA_ClrShade_global_463m", "jpg"),
    ("Mars_MGS_MOLA_ClrShade_merge_global_463m", "jpg"),
)
# 8-bit MOLA elevation. Luminance tracks height, unlike a hillshade.
DEM_LAYERS = (
    ("Mars_MGS_MOLA_DEM_mosaic_global_463m_8", "png"),
)

REGIONS = (
    {
        "id": "jezero",
        "name": "Jezero Crater",
        "subtitle": "Delta and rim",
        "kind": "crater",
        "bbox": (77.30, 18.15, 77.90, 18.75),
        "elev_range": (-2906.0, -2178.0),
        "rad_baseline": 0.64,
        "dust_baseline": 0.35,
        "start_col_row": (35, 95),
        "start_name": "Crater Floor",
        "target_col_row": (112, 42),
        "target_name": "Neretva Vallis Delta",
    },
    {
        "id": "valles",
        "name": "Valles Marineris",
        "subtitle": "Coprates Chasma",
        "kind": "canyon",
        "bbox": (-60.50, -14.20, -59.50, -13.20),
        "elev_range": (-4800.0, 1200.0),
        "rad_baseline": 0.58,
        "dust_baseline": 0.45,
        "start_col_row": (25, 75),
        "start_name": "Canyon Floor West",
        "target_col_row": (125, 80),
        "target_name": "Canyon Floor East",
    },
    {
        "id": "arcadia",
        "name": "Arcadia Planitia",
        "subtitle": "Subsurface ice plains",
        "kind": "plains",
        "bbox": (-172.00, 39.00, -171.00, 40.00),
        "elev_range": (-4100.0, -3850.0),
        "rad_baseline": 0.72,
        "dust_baseline": 0.68,
        "start_col_row": (20, 30),
        "start_name": "Ice Mining Outpost",
        "target_col_row": (130, 120),
        "target_name": "Solar Relay Array",
    },
    {
        "id": "gale",
        "name": "Gale Crater",
        "subtitle": "Mount Sharp / Aeolis Mons",
        "kind": "mount",
        "bbox": (137.00, -5.80, 138.00, -4.80),
        "elev_range": (-4500.0, 500.0),
        "rad_baseline": 0.64,
        "dust_baseline": 0.42,
        "start_col_row": (25, 30),
        "start_name": "Aeolis Palus Floor",
        "target_col_row": (110, 115),
        "target_name": "Mount Sharp Foothills",
        "recommended_exaggeration": 2.2,
    },
    {
        "id": "olympus",
        "name": "Olympus Mons",
        "subtitle": "Summit caldera",
        "kind": "volcano",
        "bbox": (-134.50, 18.00, -133.50, 19.00),
        "elev_range": (18000.0, 21229.0),
        "rad_baseline": 0.88,
        "dust_baseline": 0.25,
        "start_col_row": (22, 85),
        "start_name": "Western Caldera Approach",
        "target_col_row": (125, 70),
        "target_name": "Summit Ridge Overlook",
        "recommended_exaggeration": 2.5,
    },
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "image/jpeg,image/png,*/*"})


def log(message: str) -> None:
    print(message, flush=True)


def fractional_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    tx = ((lon + 180.0) / 360.0) * (2 ** (zoom + 1))
    ty = ((90.0 - lat) / 180.0) * (2 ** zoom)
    return tx, ty


def cell_size_meters(bbox: tuple[float, float, float, float]) -> float:
    west, south, east, north = bbox
    mid = (south + north) * 0.5
    meters_per_deg = 2.0 * math.pi * MARS_RADIUS_M / 360.0
    dx = (east - west) * meters_per_deg * math.cos(math.radians(mid)) / (COLS - 1)
    dy = (north - south) * meters_per_deg / (ROWS - 1)
    return math.sqrt(max(dx, 1.0) * max(dy, 1.0))


def rc_to_lonlat(bbox, row: int, col: int) -> tuple[float, float]:
    west, south, east, north = bbox
    lon = west + col / (COLS - 1) * (east - west)
    lat = north - row / (ROWS - 1) * (north - south)
    return lon, lat


def http_get(url: str) -> bytes:
    last_error = None
    for _attempt in range(2):
        try:
            response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                raise requests.HTTPError("404", response=response)
            response.raise_for_status()
            return response.content
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            last_error = exc
    raise last_error


def looks_like_image(data: bytes, ext: str) -> bool:
    if len(data) < 16:
        return False
    if ext == "jpg":
        return data[:3] == b"\xff\xd8\xff"
    if ext == "png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    return False


def tile_url(layer: str, ext: str, zoom: int, y: int, x: int) -> str:
    return TILE_URL.format(layer=layer, ext=ext, z=zoom, y=y, x=x)


def highest_zoom(layer: str, ext: str, bbox, preferred: int = PREFERRED_ZOOM) -> int:
    west, south, east, north = bbox
    lon = (west + east) * 0.5
    lat = (south + north) * 0.5
    for zoom in range(preferred, 2, -1):
        tx, ty = fractional_tile(lon, lat, zoom)
        url = tile_url(layer, ext, zoom, int(math.floor(ty)), int(math.floor(tx)))
        try:
            data = http_get(url)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 404:
                continue
            raise
        if looks_like_image(data, ext):
            return zoom
    raise RuntimeError(f"{layer} has no image tiles from zoom {preferred} down to 3")


def coverage(bbox, zoom: int):
    west, south, east, north = bbox
    tx0, ty_north = fractional_tile(west, north, zoom)
    tx1, ty_south = fractional_tile(east, south, zoom)
    x0, x1 = int(math.floor(tx0)), int(math.floor(tx1))
    y0, y1 = int(math.floor(ty_north)), int(math.floor(ty_south))
    return tx0, ty_north, tx1, ty_south, x0, x1, y0, y1


def fetch_mosaic(layer: str, ext: str, bbox, zoom: int) -> Image.Image:
    _tx0, _ty0, _tx1, _ty1, x0, x1, y0, y1 = coverage(bbox, zoom)
    coords = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    images = {}

    def fetch_one(coord):
        x, y = coord
        url = tile_url(layer, ext, zoom, y, x)
        data = http_get(url)
        if not looks_like_image(data, ext):
            raise RuntimeError(f"Trek response was not a {ext} tile: {url}")
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            rgb.load()
            return coord, rgb

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one, coord) for coord in coords]
        for future in as_completed(futures):
            coord, image = future.result()
            images[coord] = image

    tile_w, tile_h = next(iter(images.values())).size
    canvas = Image.new("RGB", ((x1 - x0 + 1) * tile_w, (y1 - y0 + 1) * tile_h))
    for (x, y), image in images.items():
        canvas.paste(image, ((x - x0) * tile_w, (y - y0) * tile_h))
    log(
        f"  downloaded {len(images)} NASA Mars Trek WMTS tiles "
        f"[{layer}] zoom {zoom}  ({x0}..{x1}, {y0}..{y1})"
    )
    return canvas


def crop_resize(mosaic: Image.Image, bbox, zoom: int, size: tuple[int, int], resample) -> Image.Image:
    _tx0, _ty0, _tx1, _ty1, x0, _x1, y0, _y1 = coverage(bbox, zoom)
    west, south, east, north = bbox
    tx_west, ty_north = fractional_tile(west, north, zoom)
    tx_east, ty_south = fractional_tile(east, south, zoom)
    left = (tx_west - x0) * TILE_PX
    right = (tx_east - x0) * TILE_PX
    top = (ty_north - y0) * TILE_PX
    bottom = (ty_south - y0) * TILE_PX
    l0 = max(0, int(math.floor(left)))
    t0 = max(0, int(math.floor(top)))
    r1 = min(mosaic.width, int(math.ceil(right)))
    b1 = min(mosaic.height, int(math.ceil(bottom)))
    if r1 <= l0 or b1 <= t0:
        raise RuntimeError("bbox crop fell outside the stitched Trek mosaic")
    window = mosaic.crop((l0, t0, r1, b1))
    out_w, out_h = size
    scale_x = (right - left) / out_w
    scale_y = (bottom - top) / out_h
    matrix = (scale_x, 0.0, left - l0, 0.0, scale_y, top - t0)
    # Pillow's affine transform accepts nearest, bilinear, and bicubic only.
    if resample not in (Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC):
        resample = Image.Resampling.BICUBIC
    return window.transform(size, Image.Transform.AFFINE, matrix, resample=resample)


def luminance(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    if gray.shape != (ROWS, COLS):
        raise RuntimeError(f"expected a {ROWS}x{COLS} elevation sample, got {gray.shape}")
    return gray


def stretch_to_range(values: np.ndarray, elev_range: tuple[float, float]) -> np.ndarray:
    lo, hi = np.percentile(values, [2.0, 98.0])
    span = max(float(hi - lo), 1e-6)
    norm = np.clip((values - lo) / span, 0.0, 1.0)
    return elev_range[0] + norm * (elev_range[1] - elev_range[0])


def maybe_invert(norm: np.ndarray, kind: str) -> np.ndarray:
    """MOLA browse products are bright-high. Flip only if a crater or canyon is upside down."""
    if kind == "crater":
        center = float(norm[48:102, 48:102].mean())
        rim = float(np.concatenate([norm[:22, :].ravel(), norm[-22:, :].ravel(), norm[:, :22].ravel(), norm[:, -22:].ravel()]).mean())
        if center > rim + 0.03:
            log("  inverted DEM polarity so the crater floor sits below the rim")
            return 1.0 - norm
    elif kind == "canyon":
        floor = float(norm[58:96, :].mean())
        walls = float(np.concatenate([norm[:28, :].ravel(), norm[-28:, :].ravel()]).mean())
        if floor > walls + 0.03:
            log("  inverted DEM polarity so the canyon floor sits below the walls")
            return 1.0 - norm
    elif kind == "mount":
        north = float(norm[:40, :].mean())
        south = float(norm[-40:, :].mean())
        if north > south + 0.04:
            log("  inverted DEM polarity so Mount Sharp rises above Aeolis Palus")
            return 1.0 - norm
    elif kind == "volcano":
        center = float(norm[60:90, 60:90].mean())
        frame = np.concatenate(
            [
                norm[40:55, 40:110].ravel(),
                norm[95:110, 40:110].ravel(),
                norm[55:95, 40:55].ravel(),
                norm[55:95, 95:110].ravel(),
            ]
        )
        if center > float(frame.mean()) + 0.04:
            log("  inverted DEM polarity so the caldera sits below the summit rim")
            return 1.0 - norm
    return norm


def box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return array
    kernel = 2 * radius + 1
    padded = np.pad(array, ((0, 0), (radius, radius)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)), mode="constant")
    horizontal = (cumulative[:, kernel:] - cumulative[:, :-kernel]) / kernel
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=0), ((1, 0), (0, 0)), mode="constant")
    return (cumulative[kernel:, :] - cumulative[:-kernel, :]) / kernel


def gaussian_smooth(array: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    """Two separable 5-tap passes whose combined sigma is about 1.15."""
    step = sigma / math.sqrt(2.0)
    radius = 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / step) ** 2)
    kernel /= kernel.sum()
    blurred = np.asarray(array, dtype=np.float64)
    for _pass in range(2):
        padded = np.pad(blurred, ((0, 0), (radius, radius)), mode="edge")
        horizontal = np.zeros_like(blurred)
        width = blurred.shape[1]
        for index, weight in enumerate(kernel):
            horizontal += weight * padded[:, index : index + width]
        padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
        vertical = np.zeros_like(blurred)
        height = blurred.shape[0]
        for index, weight in enumerate(kernel):
            vertical += weight * padded[index : index + height, :]
        blurred = vertical
    return blurred


def pin_elevation(array: np.ndarray, elev_range: tuple[float, float]) -> np.ndarray:
    """Linearly restore the exact site elevation envelope after smoothing."""
    lo = float(np.min(array))
    hi = float(np.max(array))
    span = max(hi - lo, 1e-9)
    scale = (elev_range[1] - elev_range[0]) / span
    return elev_range[0] + (array - lo) * scale


def finish_elevation(elev: np.ndarray, elev_range: tuple[float, float]) -> np.ndarray:
    return pin_elevation(gaussian_smooth(elev), elev_range)


def compute_slope(elev: np.ndarray, cell_m: float) -> np.ndarray:
    dzdx = np.empty_like(elev)
    dzdy = np.empty_like(elev)
    dzdx[:, 1:-1] = (elev[:, 2:] - elev[:, :-2]) / (2 * cell_m)
    dzdy[1:-1, :] = (elev[2:, :] - elev[:-2, :]) / (2 * cell_m)
    dzdx[:, 0] = (elev[:, 1] - elev[:, 0]) / cell_m
    dzdx[:, -1] = (elev[:, -1] - elev[:, -2]) / cell_m
    dzdy[0, :] = (elev[1, :] - elev[0, :]) / cell_m
    dzdy[-1, :] = (elev[-1, :] - elev[-2, :]) / cell_m
    return np.sqrt(dzdx * dzdx + dzdy * dzdy) * 100.0


def snap_cell(slope: np.ndarray, row: int, col: int, max_slope: float = 25.0):
    rows, cols = slope.shape
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    if slope[row, col] <= max_slope:
        return row, col
    seen = np.zeros(rows * cols, dtype=bool)
    start = row * cols + col
    seen[start] = True
    queue = [start]
    head = 0
    while head < len(queue):
        index = queue[head]
        head += 1
        r, c = divmod(index, cols)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            ni = nr * cols + nc
            if seen[ni]:
                continue
            if slope[nr, nc] <= max_slope:
                return nr, nc
            seen[ni] = True
            queue.append(ni)
    return row, col


def reachable(slope: np.ndarray, start, goal, max_slope: float = 25.0) -> bool:
    return astar_path(slope, np.zeros_like(slope), np.zeros_like(slope), start, goal, 0.0, 0.0, 0.0, max_slope) is not None


def astar_path(slope, radiation, dust, start, goal, w_slope, w_rad, w_dust, max_slope=25.0):
    rows, cols = slope.shape
    sr, sc = start
    gr, gc = goal
    if slope[sr, sc] > max_slope or slope[gr, gc] > max_slope:
        return None
    min_factor = 1.0 + w_rad * float(np.min(radiation)) + w_dust * float(np.min(dust))
    sqrt2 = math.sqrt(2.0)

    def heuristic(row, col):
        dr = abs(row - gr)
        dc = abs(col - gc)
        smaller = dr if dr < dc else dc
        return (dr + dc + (sqrt2 - 2.0) * smaller) * min_factor

    start_i = sr * cols + sc
    goal_i = gr * cols + gc
    g_score = {start_i: 0.0}
    came = {}
    heap = [(heuristic(sr, sc), 0.0, start_i)]
    closed = set()
    directions = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, sqrt2),
        (1, -1, sqrt2),
        (-1, 1, sqrt2),
        (-1, -1, sqrt2),
    )
    while heap:
        _f, g_cost, index = heapq.heappop(heap)
        if index in closed or g_cost > g_score.get(index, 1e100) + 1e-9:
            continue
        if index == goal_i:
            path = []
            cursor = index
            while True:
                path.append((cursor // cols, cursor % cols))
                if cursor == start_i:
                    break
                cursor = came[cursor]
            path.reverse()
            return path
        closed.add(index)
        row, col = divmod(index, cols)
        for d_row, d_col, step in directions:
            nr, nc = row + d_row, col + d_col
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            if slope[nr, nc] > max_slope:
                continue
            if d_row and d_col and (slope[row, nc] > max_slope or slope[nr, col] > max_slope):
                continue
            ni = nr * cols + nc
            if ni in closed:
                continue
            factor = 1.0 + w_slope * (float(slope[nr, nc]) / 25.0) + w_rad * float(radiation[nr, nc]) + w_dust * float(dust[nr, nc])
            next_g = g_cost + step * factor
            if next_g + 1e-9 < g_score.get(ni, 1e100):
                g_score[ni] = next_g
                came[ni] = index
                heapq.heappush(heap, (next_g + heuristic(nr, nc), next_g, ni))
    return None


def soften_elevation(elev: np.ndarray, cell_m: float, elev_range, start, goal):
    """Keep real relief, but open a rover corridor if a raw upsample is a cliff field."""
    best = None
    for radius in (0, 1, 2, 4, 7, 10, 14):
        blurred = elev if radius == 0 else box_blur(elev, radius)
        scaled = stretch_to_range(blurred, elev_range)
        slope = compute_slope(scaled, cell_m)
        snapped_start = snap_cell(slope, start[0], start[1])
        snapped_goal = snap_cell(slope, goal[0], goal[1])
        open_fraction = float(np.mean(slope <= 25.0))
        connected = reachable(slope, snapped_start, snapped_goal)
        candidate = (scaled, slope, radius, connected, open_fraction, snapped_start, snapped_goal)
        best = candidate
        if connected and open_fraction >= 0.28:
            return candidate
    return best


def hazard_layers(elev, cell_m, region, exposure=None):
    slope = compute_slope(elev, cell_m)
    local = box_blur(elev, 5)
    relief = max(40.0, 0.05 * (region["elev_range"][1] - region["elev_range"][0]))
    prominence = np.clip((elev - local) / relief, 0.0, 1.0)
    steep = np.clip((slope - 5.0) / 18.0, 0.0, 1.0)
    ridge = np.clip(0.70 * prominence + 0.85 * steep, 0.0, 1.0)
    reference = float(np.percentile(elev, 15))
    pressure = np.exp(-(elev - reference) / 11100.0)
    atmosphere = np.power(np.clip(pressure, 0.35, 2.5), -0.35)
    radiation = region["rad_baseline"] * atmosphere + 0.48 * ridge
    if exposure is not None:
        radiation = radiation + 0.62 * exposure
    low = 1.0 - np.clip(
        (elev - region["elev_range"][0]) / (region["elev_range"][1] - region["elev_range"][0] + 1e-6),
        0.0,
        1.0,
    )
    flat = np.exp(-np.maximum(slope, 0.0) / 7.0)
    dust = region["dust_baseline"] * (0.45 + 0.85 * flat * (0.35 + 0.65 * low))
    dust = dust * (1.0 - 0.50 * ridge)
    if exposure is not None:
        dust = dust + 0.22 * exposure
    dust = np.clip(dust, 0.04, 0.96)
    radiation = np.clip(radiation - 0.03 * dust * (1.0 - ridge), 0.40, 2.60)
    return slope, radiation, dust


def value_noise(rows, cols, grid, rng):
    coarse = rng.random((grid + 1, grid + 1)).astype(np.float32)
    ys = np.linspace(0, grid, rows, dtype=np.float32)
    xs = np.linspace(0, grid, cols, dtype=np.float32)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, grid - 1)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, grid - 1)
    wy = (ys - y0.astype(np.float32))[:, None]
    wx = (xs - x0.astype(np.float32))[None, :]
    n00 = coarse[y0[:, None], x0[None, :]]
    n10 = coarse[y0[:, None], (x0 + 1)[None, :]]
    n01 = coarse[(y0 + 1)[:, None], x0[None, :]]
    n11 = coarse[(y0 + 1)[:, None], (x0 + 1)[None, :]]
    return n00 * (1 - wx) * (1 - wy) + n10 * wx * (1 - wy) + n01 * (1 - wx) * wy + n11 * wx * wy


def fbm(rows, cols, rng, octaves=4, base=3):
    out = np.zeros((rows, cols), dtype=np.float64)
    amplitude = 1.0
    total = 0.0
    for octave in range(octaves):
        grid = base * (2 ** octave)
        if grid >= max(rows, cols):
            break
        out += amplitude * value_noise(rows, cols, grid, rng)
        total += amplitude
        amplitude *= 0.5
    return out / max(total, 1e-6)


def _mesh():
    col, row = np.meshgrid(np.arange(COLS, dtype=np.float64), np.arange(ROWS, dtype=np.float64))
    return row, col


def _line_mask(row, col, start, goal, cross_sigma):
    sr, sc = start
    gr, gc = goal
    vr, vc = gr - sr, gc - sc
    length = math.hypot(vr, vc) or 1.0
    along = ((row - sr) * vr + (col - sc) * vc) / length
    cross = ((row - sr) * vc - (col - sc) * vr) / length
    on_segment = (along > length * 0.18) & (along < length * 0.82)
    ridge = np.exp(-0.5 * (cross / cross_sigma) ** 2) * on_segment
    return ridge


def synthetic_elevation(region, rng):
    row, col = _mesh()
    lo, hi = region["elev_range"]
    kind = region["kind"]
    if kind == "crater":
        floor = lo + 70.0
        crest = hi - 20.0
        radius = np.hypot(row - 78.0, col - 96.0)
        wall = np.clip((radius - 52.0) / 9.0, 0.0, 1.0)
        wall = wall * wall * (3.0 - 2.0 * wall)
        elev = floor + (crest - floor) * wall
        inlet = np.exp(-0.5 * ((row - 72.0) / 7.5) ** 2) * np.clip((48.0 - col) / 48.0, 0.0, 1.0)
        elev = elev * (1.0 - 0.92 * inlet) + floor * 0.92 * inlet
        elev += (fbm(ROWS, COLS, rng, 4, 3) - 0.5) * 36.0
    elif kind == "canyon":
        dist = np.abs(row - 77.0)
        wall = np.clip((dist - 16.0) / 9.0, 0.0, 1.0)
        wall = np.power(wall, 1.15)
        elev = lo + (hi - lo) * wall
        elev += (fbm(ROWS, COLS, rng, 3, 3) - 0.5) * 80.0
    elif kind == "mount":
        floor = lo + 90.0
        mound_r = np.hypot(row - 108.0, col - 100.0) / 70.0
        mound = 0.5 * (1.0 + np.cos(np.pi * np.clip(mound_r, 0.0, 1.0)))
        border = np.minimum(np.minimum(row, ROWS - 1.0 - row), np.minimum(col, COLS - 1.0 - col))
        rim = np.clip((12.0 - border) / 8.0, 0.0, 1.0) ** 2
        elev = floor + 2800.0 * mound + 1500.0 * rim
        elev += (fbm(ROWS, COLS, rng, 4, 3) - 0.5) * 36.0
    elif kind == "volcano":
        span = hi - lo
        elev = lo + span * (0.18 + 0.62 * (col / (COLS - 1.0)))
        elev += 160.0 * np.sin(row / 26.0)
        bowl = np.exp(-0.5 * (((row - 74.0) / 16.0) ** 2 + ((col - 76.0) / 14.0) ** 2))
        ring_r = np.hypot((row - 74.0) / 28.0, (col - 76.0) / 26.0)
        ring = np.exp(-0.5 * ((ring_r - 1.0) / 0.16) ** 2)
        north_gap = np.exp(-0.5 * ((row - 30.0) / 12.0) ** 2)
        elev = elev - 900.0 * bowl + 520.0 * ring * (1.0 - north_gap)
        elev += (fbm(ROWS, COLS, rng, 3, 3) - 0.5) * 50.0
    else:
        mid = (lo + hi) * 0.5
        elev = mid + (fbm(ROWS, COLS, rng, 4, 3) - 0.5) * (hi - lo) * 0.55
        elev += 18.0 * np.sin(col / 17.0) * np.cos(row / 23.0)
    start = (region["start_col_row"][1], region["start_col_row"][0])
    goal = (region["target_col_row"][1], region["target_col_row"][0])
    if kind == "canyon":
        mound = np.exp(-0.5 * (((row - 77.0) / 7.0) ** 2 + ((col - 75.0) / 10.0) ** 2))
        elev = elev + 280.0 * mound
        exposure = mound
    elif kind == "mount":
        exposure = _line_mask(row, col, (36.0, 48.0), (96.0, 88.0), 6.0)
        elev = elev + 80.0 * exposure
    elif kind == "volcano":
        exposure = np.exp(-0.5 * ((row - 96.0) / 8.0) ** 2) * np.clip((col - 30.0) / 90.0, 0.0, 1.0)
        elev = elev + 40.0 * exposure
    else:
        exposure = _line_mask(row, col, start, goal, 5.5 if kind == "crater" else 8.0)
        elev = elev + (90.0 if kind == "crater" else 28.0) * exposure
    return np.clip(elev, lo - 80.0, hi + 80.0), np.clip(exposure, 0.0, 1.0)


def render_synthetic_texture(elev, exposure, region) -> Image.Image:
    size = 512
    small = box_blur(elev, 1)
    field = np.array(Image.fromarray(small.astype(np.float32), mode="F").resize((size, size), Image.Resampling.BICUBIC), dtype=np.float64)
    exposure_i = np.array(
        Image.fromarray(exposure.astype(np.float32), mode="F").resize((size, size), Image.Resampling.BICUBIC),
        dtype=np.float64,
    )
    dzdx = np.zeros_like(field)
    dzdy = np.zeros_like(field)
    dzdx[:, 1:-1] = field[:, 2:] - field[:, :-2]
    dzdy[1:-1, :] = field[2:, :] - field[:-2, :]
    nx, ny, nz = -dzdx, -dzdy, np.full_like(field, 1400.0)
    normal = np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = np.clip((nx * -0.45 + ny * -0.25 + nz * 0.85) / normal, 0.0, 1.0)
    shade = 0.48 + 0.72 * shade
    span = np.percentile(field, 95) - np.percentile(field, 5) + 1e-6
    elev_n = np.clip((field - np.percentile(field, 5)) / span, 0.0, 1.0)
    if region["kind"] == "plains":
        red, grn, blu = 168 + 30 * elev_n, 142 + 24 * elev_n, 112 + 16 * elev_n
    elif region["kind"] == "canyon":
        red = 92 + 110 * elev_n
        grn = 48 + 78 * elev_n
        blu = 36 + 52 * elev_n
    elif region["kind"] == "volcano":
        red, grn, blu = 78 + 70 * elev_n, 62 + 48 * elev_n, 58 + 36 * elev_n
    elif region["kind"] == "mount":
        red, grn, blu = 148 + 62 * elev_n, 96 + 48 * elev_n, 58 + 28 * elev_n
    else:
        red, grn, blu = 132 + 70 * elev_n, 68 + 40 * elev_n, 42 + 24 * elev_n
    red = red * (1 - 0.25 * exposure_i) + 186 * exposure_i
    grn = grn * (1 - 0.25 * exposure_i) + 150 * exposure_i
    blu = blu * (1 - 0.25 * exposure_i) + 118 * exposure_i
    rng = np.random.default_rng(20261114 + ROWS)
    grain = 0.9 + 0.12 * fbm(size, size, rng, 4, 4)
    rgb = np.dstack(
        [
            np.clip(red * shade * grain, 0, 255).astype(np.uint8),
            np.clip(grn * shade * grain, 0, 255).astype(np.uint8),
            np.clip(blu * shade * grain, 0, 255).astype(np.uint8),
        ]
    )
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((1024, 1024), Image.Resampling.LANCZOS)


def acquire_layer(candidates, bbox, label: str):
    errors = []
    for layer, ext in candidates:
        try:
            zoom = highest_zoom(layer, ext, bbox, PREFERRED_ZOOM)
            if zoom != PREFERRED_ZOOM:
                log(f"  {label}: zoom {PREFERRED_ZOOM} is not published for {layer}; using zoom {zoom}")
            mosaic = fetch_mosaic(layer, ext, bbox, zoom)
            return mosaic, layer, ext, zoom
        except Exception as exc:
            errors.append(f"{layer}: {exc}")
            log(f"  {label} layer failed ({layer}): {exc}")
    raise RuntimeError(f"{label} unavailable: {'; '.join(errors)}")


def build_from_trek(region):
    bbox = region["bbox"]
    log(f"{region['id']}: requesting NASA Mars Trek WMTS tiles for {region['name']}")
    texture_mosaic, texture_layer, _ext, texture_zoom = acquire_layer(TEXTURE_LAYERS, bbox, "texture")
    texture = crop_resize(texture_mosaic, bbox, texture_zoom, (1024, 1024), Image.Resampling.LANCZOS)

    relief_note = None
    relief_zoom = None
    relief_layer = None
    try:
        relief_mosaic, relief_layer, _ext, relief_zoom = acquire_layer(RELIEF_LAYERS, bbox, "relief")
        relief_note = relief_layer
    except Exception as exc:
        log(f"  relief layer unavailable, elevation will use the DEM only ({exc})")

    dem_mosaic = None
    dem_layer = None
    dem_zoom = None
    try:
        dem_mosaic, dem_layer, _ext, dem_zoom = acquire_layer(DEM_LAYERS, bbox, "elevation")
    except Exception as exc:
        log(f"  grayscale DEM unavailable ({exc})")

    if dem_mosaic is None and relief_note is None:
        raise RuntimeError("neither the MOLA DEM nor the shaded-relief layer returned tiles")

    if dem_mosaic is not None:
        dem_image = crop_resize(dem_mosaic, bbox, dem_zoom, (ROWS, COLS), Image.Resampling.BILINEAR)
        samples = maybe_invert(luminance(dem_image), region["kind"])
        elevation_layer = dem_layer
        elevation_zoom = dem_zoom
        elevation_kind = "MOLA grayscale DEM luminance"
    else:
        shade = crop_resize(relief_mosaic, bbox, relief_zoom, (ROWS, COLS), Image.Resampling.BILINEAR)
        samples = luminance(shade)
        elevation_layer = relief_layer
        elevation_zoom = relief_zoom
        elevation_kind = "shaded-relief luminance (DEM tiles unavailable)"

    rough = finish_elevation(stretch_to_range(samples, region["elev_range"]), region["elev_range"])
    log("  gaussian-smoothed the 150x150 elevation grid (sigma 1.15) and pinned the site elevation range")
    start = (region["start_col_row"][1], region["start_col_row"][0])
    goal = (region["target_col_row"][1], region["target_col_row"][0])
    elev, _slope, blur, connected, open_fraction, snapped_start, snapped_goal = soften_elevation(
        rough, cell_size_meters(bbox), region["elev_range"], start, goal
    )
    if blur and connected:
        log(f"  smoothed elevation with a {blur}-cell box blur so a rover corridor stays open")
    elif not connected:
        log("  default sortie is still blocked after the corridor blur")
    log(
        f"  elevation grid: {elevation_kind}, scaled to "
        f"[{region['elev_range'][0]:.0f}, {region['elev_range'][1]:.0f}] m "
        f"(passable cells {open_fraction * 100:.1f}%)"
    )
    slope, radiation, dust = hazard_layers(elev, cell_size_meters(bbox), region, exposure=None)
    return {
        "elevation": elev,
        "slope": slope,
        "radiation": radiation,
        "dust": dust,
        "texture": texture,
        "data_source": "NASA Mars Trek WMTS",
        "live": True,
        "texture_layer": texture_layer,
        "texture_zoom": texture_zoom,
        "relief_layer": relief_note,
        "relief_zoom": relief_zoom,
        "elevation_layer": elevation_layer,
        "elevation_zoom": elevation_zoom,
        "elevation_kind": elevation_kind,
        "connected": connected,
        "snapped_start": snapped_start,
        "snapped_goal": snapped_goal,
    }


def build_fallback(region):
    rng = np.random.default_rng(20261114 + sum(ord(ch) for ch in region["id"]))
    elev, exposure = synthetic_elevation(region, rng)
    elev = finish_elevation(elev, region["elev_range"])
    cell = cell_size_meters(region["bbox"])
    slope, radiation, dust = hazard_layers(elev, cell, region, exposure=exposure)
    texture = render_synthetic_texture(elev, exposure, region)
    start = (region["start_col_row"][1], region["start_col_row"][0])
    goal = (region["target_col_row"][1], region["target_col_row"][0])
    return {
        "elevation": elev,
        "slope": slope,
        "radiation": radiation,
        "dust": dust,
        "texture": texture,
        "data_source": "seeded-mola-fallback",
        "live": False,
        "texture_layer": "synthetic-matched-to-dem",
        "texture_zoom": None,
        "relief_layer": None,
        "relief_zoom": None,
        "elevation_layer": "synthetic",
        "elevation_zoom": None,
        "elevation_kind": "seeded MOLA-calibrated terrain",
        "connected": reachable(slope, snap_cell(slope, *start), snap_cell(slope, *goal)),
        "snapped_start": snap_cell(slope, *start),
        "snapped_goal": snap_cell(slope, *goal),
    }


def round_grid(array: np.ndarray, digits: int):
    return np.round(array.astype(np.float64), digits).tolist()


def write_region(region, pack) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bbox = region["bbox"]
    west, south, east, north = bbox
    cell = round(cell_size_meters(bbox), 1)
    start_col, start_row = region["start_col_row"]
    goal_col, goal_row = region["target_col_row"]
    start_lon, start_lat = rc_to_lonlat(bbox, start_row, start_col)
    goal_lon, goal_lat = rc_to_lonlat(bbox, goal_row, goal_col)
    elev = pack["elevation"]
    slope = pack["slope"]
    radiation = pack["radiation"]
    dust = pack["dust"]
    detail = (
        f"Texture {pack['texture_layer']} zoom {pack['texture_zoom']}. "
        f"Elevation {pack['elevation_kind']} from {pack['elevation_layer']} zoom {pack['elevation_zoom']}. "
        f"Relief layer {pack['relief_layer']} zoom {pack['relief_zoom']}."
    )
    meta = {
        "id": region["id"],
        "name": region["name"],
        "subtitle": region["subtitle"],
        "dataSource": pack["data_source"],
        "live": pack["live"],
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "rows": ROWS,
        "cols": COLS,
        "cellSizeMeters": cell,
        "crs": "EPSG:4326",
        "rowOrder": "north-to-south",
        "elevationUnit": "meters-areoid",
        "slopeUnit": "percent",
        "radiationUnit": "mSv/day",
        "dustUnit": "slip-risk-0-1",
        "source": pack["data_source"],
        "sourceDetail": detail,
        "textureLayer": pack["texture_layer"],
        "textureZoom": pack["texture_zoom"],
        "reliefLayer": pack["relief_layer"],
        "reliefZoom": pack["relief_zoom"],
        "elevationLayer": pack["elevation_layer"],
        "elevationZoom": pack["elevation_zoom"],
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elevationSmoothing": "separable Gaussian, two 5-tap passes, sigma 1.15, then pinned to elevRange",
        "models": {
            "slope": f"percent grade from central finite differences on a {cell} m cell",
            "radiation": (
                f"{region['rad_baseline']} mSv/day regional baseline, Mars scale height 11.1 km, "
                "plus an exposed-ridge term"
            ),
            "dust": "higher on flat low basins, lower on rocky slopes; region dust baseline "
            f"{region['dust_baseline']}",
        },
        "stats": {
            "elevMin": round(float(elev.min()), 1),
            "elevMax": round(float(elev.max()), 1),
            "slopeMax": round(float(slope.max()), 2),
            "impassableFractionAt25": round(float(np.mean(slope > 25.0)), 4),
            "radiationMin": round(float(radiation.min()), 3),
            "radiationMax": round(float(radiation.max()), 3),
            "dustMin": round(float(dust.min()), 3),
            "dustMax": round(float(dust.max()), 3),
        },
        "defaultStart": {
            "name": region["start_name"],
            "col": int(start_col),
            "row": int(start_row),
            "lat": round(start_lat, 5),
            "lon": round(start_lon, 5),
        },
        "defaultTarget": {
            "name": region["target_name"],
            "col": int(goal_col),
            "row": int(goal_row),
            "lat": round(goal_lat, 5),
            "lon": round(goal_lon, 5),
        },
    }
    if region.get("recommended_exaggeration") is not None:
        meta["recommendedExaggeration"] = region["recommended_exaggeration"]
    texture_path = OUT_DIR / f"{region['id']}_texture.jpg"
    json_path = OUT_DIR / f"{region['id']}_terrain.json"
    pack["texture"].save(texture_path, format="JPEG", quality=90, optimize=True)
    payload = {
        "metadata": meta,
        "elevation": round_grid(elev, 1),
        "slope": round_grid(slope, 2),
        "radiation": round_grid(radiation, 3),
        "dust": round_grid(dust, 3),
    }
    json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    if region["id"] == "jezero":
        shutil.copyfile(texture_path, OUT_DIR / "mars_texture.jpg")
        shutil.copyfile(json_path, OUT_DIR / "mars_terrain.json")
    snapped = pack["snapped_start"], pack["snapped_goal"]
    safest = astar_path(slope, radiation, dust, snapped[0], snapped[1], 3.5, 4.0, 2.0)
    fastest = astar_path(slope, radiation, dust, snapped[0], snapped[1], 0.5, 0.3, 0.2)
    safe_n = len(safest) if safest else 0
    fast_n = len(fastest) if fastest else 0
    share = 0.0
    if safest and fastest:
        share = len(set(map(tuple, safest)) & set(map(tuple, fastest))) / max(len(safest), 1)
    status = "LIVE NASA Mars Trek WMTS" if pack["live"] else "SEEDED FALLBACK"
    log(
        f"{region['id']}: {status} | elev {meta['stats']['elevMin']:.0f}..{meta['stats']['elevMax']:.0f} m | "
        f"slope max {meta['stats']['slopeMax']:.1f}% | impassable {meta['stats']['impassableFractionAt25'] * 100:.1f}% | "
        f"rad {meta['stats']['radiationMin']:.3f}-{meta['stats']['radiationMax']:.3f} | "
        f"routes safest {safe_n} / fastest {fast_n} cells (shared {share:.0%})"
    )
    log(f"  wrote {json_path.name} and {texture_path.name}")


def write_icon() -> None:
    icon = Image.new("RGBA", (32, 32), (18, 10, 8, 255))
    pixels = icon.load()
    for py in range(32):
        for px in range(32):
            if (px - 16) ** 2 + (py - 16) ** 2 <= 110:
                pixels[px, py] = (196, 92, 48, 255)
            if 14 <= px <= 17 and py >= 8:
                pixels[px, py] = (255, 214, 170, 255)
    icon.save(ROOT / "public" / "favicon.ico", sizes=[(32, 32)])


def self_check() -> None:
    tx, ty = fractional_tile(0.0, 0.0, 8)
    if abs(tx - 256.0) > 1e-6 or abs(ty - 128.0) > 1e-6:
        raise RuntimeError(f"tile math failed at the origin: tx={tx}, ty={ty}")
    slope = np.zeros((8, 8))
    slope[1:7, 3] = 40.0
    radiation = np.full((8, 8), 0.64)
    dust = np.full((8, 8), 0.2)
    path = astar_path(slope, radiation, dust, (3, 1), (3, 6), 0.5, 0.3, 0.2)
    if not path or any(slope[row, col] > 25 for row, col in path):
        raise RuntimeError("A* self-check failed to walk around an impassable wall")
    spike = np.zeros((9, 9), dtype=np.float64)
    spike[4, 4] = 1.0
    blurred = gaussian_smooth(spike)
    if not (blurred[4, 4] > blurred[4, 3] > blurred[4, 2]) or abs(float(blurred.sum()) - 1.0) > 1e-6:
        raise RuntimeError("Gaussian smoothing self-check failed")
    pinned = pin_elevation(np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64), (-4500.0, 500.0))
    if abs(float(pinned.min()) + 4500.0) > 1e-6 or abs(float(pinned.max()) - 500.0) > 1e-6:
        raise RuntimeError("Elevation range pin self-check failed")
    log("Tile math, Gaussian smoothing, and A* self-check passed.")


def main() -> int:
    self_check()
    written = []
    for region in REGIONS:
        try:
            pack = build_from_trek(region)
        except Exception as exc:
            log(f"{region['id']}: Trek fetch failed ({exc}). Writing the seeded fallback.")
            pack = build_fallback(region)
        write_region(region, pack)
        written.append((region["id"], pack["live"], pack["data_source"]))
    write_icon()
    log("---")
    for region_id, live, source in written:
        state = "LIVE TILES" if live else "OFFLINE FALLBACK"
        log(f"{region_id}: dataSource = {source} ({state})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FAILED: {exc}")
        raise
