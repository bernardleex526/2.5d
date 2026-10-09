"""
Parametric Mountain Terrain Heightmap Generator for Gazebo Simulation
Designed for DEEP Robotics M20 Pro Wheeled-Legged Robot Mountain Patrol Testing.

Generates:
1. 16-bit uint16 and 8-bit uint8 grayscale heightmap PNG (dimensions 2^N + 1, e.g. 1025x1025).
2. Features:
   - Staging / Spawn flat base camp at (-30m, -30m) with smooth hermite blend;
   - Procedural rolling mountain hills & ridges;
   - Continuous stone stairs (连续石阶梯) with 24 steps (0.15m rise, 0.45m run, 3.6m elevation gain);
   - Lateral earthen shoulder/embankment smooth blend (eliminating knife-edge cliff drops);
   - Smooth entrance and exit landing platforms;
   - Parallel 16.7° bypass slope for wheeled vs stepping locomotion comparison;
   - Closed-loop mountain patrol trail for SLAM loop closure validation;
   - Standalone terrain textures (dirt, grass, rock, flat normal) for self-contained Gazebo rendering.
3. Terrain metadata JSON recording exact coordinates and dimensions.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image


def gaussian_smooth_2d(arr: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """Fast separable 2D Gaussian filter using pure NumPy."""
    if sigma <= 0.0:
        return arr.copy()
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    # Axis 0 convolution with edge padding
    padded_0 = np.pad(arr, ((radius, radius), (0, 0)), mode="edge")
    out_0 = np.zeros_like(arr, dtype=np.float64)
    for i, w in enumerate(kernel):
        out_0 += w * padded_0[i : i + arr.shape[0], :]

    # Axis 1 convolution with edge padding
    padded_1 = np.pad(out_0, ((0, 0), (radius, radius)), mode="edge")
    out_1 = np.zeros_like(arr, dtype=np.float64)
    for j, w in enumerate(kernel):
        out_1 += w * padded_1[:, j : j + arr.shape[1]]

    return out_1


def generate_terrain(
    grid_size: int = 1025,
    world_size_x: float = 100.0,
    world_size_y: float = 100.0,
    max_height: float = 15.0,
    base_height: float = 1.0,
    num_steps: int = 24,
    step_rise: float = 0.15,
    step_run: float = 0.45,
    stair_width: float = 3.2,
) -> tuple[np.ndarray, dict]:
    """
    Generate parametric heightmap array (grid_size x grid_size) normalized to [0, max_height].
    """
    # Create coordinate grid in meters, centered at (0, 0)
    xs = np.linspace(-world_size_x / 2.0, world_size_x / 2.0, grid_size)
    ys = np.linspace(-world_size_y / 2.0, world_size_y / 2.0, grid_size)
    X, Y = np.meshgrid(xs, ys)

    # 1. Base Rolling Hills (Multi-scale procedural terrain)
    # Low frequency mountain mass
    h_macro = (
        0.35 * np.sin(2.0 * np.pi * X / 120.0 + 0.5) * np.cos(2.0 * np.pi * Y / 110.0)
        + 0.40 * np.sin(2.0 * np.pi * (X * 0.7 - Y * 0.7) / 90.0)
        + 0.25 * np.cos(2.0 * np.pi * (X * 0.5 + Y * 0.8) / 80.0)
    )
    # Mid frequency rolling hills
    h_meso = (
        0.15 * np.sin(2.0 * np.pi * X / 35.0) * np.sin(2.0 * np.pi * Y / 40.0)
        + 0.12 * np.cos(2.0 * np.pi * (X + Y) / 30.0)
        + 0.08 * np.sin(2.0 * np.pi * (X - Y * 1.2) / 22.0)
    )
    # High frequency surface roughness / natural undulations
    h_micro = (
        0.03 * np.sin(2.0 * np.pi * X / 8.5) * np.cos(2.0 * np.pi * Y / 9.0)
        + 0.02 * np.cos(2.0 * np.pi * (X * 1.5 + Y) / 6.0)
    )

    # Combine natural terrain and normalize to [0, 1]
    h_raw = h_macro + h_meso + h_micro
    h_norm = (h_raw - h_raw.min()) / (h_raw.max() - h_raw.min())

    # Scale to elevation in meters
    elevation = base_height + h_norm * (max_height - base_height - 4.5)

    # Prominent mountain ridge / summit near (+22, +20)
    dist_summit = np.sqrt((X - 22.0) ** 2 + (Y - 20.0) ** 2)
    summit_bump = 4.2 * np.exp(-((dist_summit / 18.0) ** 2))
    elevation += summit_bump

    # Smooth the natural terrain first before engineering geometric features
    elevation = gaussian_smooth_2d(elevation, sigma=0.6)

    # 2. Base Station Flat Staging Platform (Spawn Area)
    spawn_center_x, spawn_center_y = -30.0, -30.0
    spawn_radius = 7.5
    dist_spawn = np.sqrt((X - spawn_center_x) ** 2 + (Y - spawn_center_y) ** 2)
    spawn_flat_elevation = base_height + 0.5

    # Smooth transition using cubic Hermite blend
    blend_inner = spawn_radius
    blend_outer = spawn_radius + 4.0
    t_spawn = np.clip((dist_spawn - blend_inner) / (blend_outer - blend_inner), 0.0, 1.0)
    w_spawn = 3.0 * t_spawn**2 - 2.0 * t_spawn**3
    elevation = (1.0 - w_spawn) * spawn_flat_elevation + w_spawn * elevation

    # 3. Continuous Stone Stairs (连续石阶梯) with Smooth Earthen Shoulder Blending
    stair_start_x = -6.0
    stair_start_y = 12.0
    stair_length = num_steps * step_run  # e.g. 24 * 0.45m = 10.8m
    stair_end_x = stair_start_x + stair_length

    # Determine base elevation at the foot of the stairs
    idx_x_start = np.argmin(np.abs(xs - stair_start_x))
    idx_y_start = np.argmin(np.abs(ys - stair_start_y))
    stair_base_z = float(elevation[idx_y_start, idx_x_start])

    # Longitudinal distance (s) and lateral distance (d)
    s_coord = X - stair_start_x
    d_coord = np.abs(Y - stair_start_y)

    # Stepped profile along the climbing direction
    step_indices = np.floor(s_coord / step_run)
    step_indices = np.clip(step_indices, 0, num_steps - 1)
    stair_elev_profile = stair_base_z + step_indices * step_rise

    # Tread core: perfectly flat across stair_width
    half_width = stair_width / 2.0  # 1.6m
    shoulder_width = 2.0  # 2.0m smooth embankment transition zone

    # Lateral cubic Hermite weight for shoulder blending
    t_d = np.clip((d_coord - half_width) / shoulder_width, 0.0, 1.0)
    w_shoulder = 3.0 * t_d**2 - 2.0 * t_d**3

    in_stair_length = (s_coord >= 0.0) & (s_coord <= stair_length)
    in_stair_corridor = in_stair_length & (d_coord <= half_width + shoulder_width)

    # Apply stepped profile on tread, and smoothly blend embankment on shoulders
    elevation_stair_corridor = (1.0 - w_shoulder) * stair_elev_profile + w_shoulder * elevation
    elevation[in_stair_corridor] = elevation_stair_corridor[in_stair_corridor]

    # Landing platforms at entrance and exit of stairs
    # Start Landing (at foot of stairs)
    dist_landing_start = np.sqrt((X - (stair_start_x - 1.8)) ** 2 + (Y - stair_start_y) ** 2)
    t_lstart = np.clip((dist_landing_start - 1.8) / 2.0, 0.0, 1.0)
    w_lstart = 3.0 * t_lstart**2 - 2.0 * t_lstart**3
    in_lstart_zone = dist_landing_start <= 3.8
    elevation[in_lstart_zone] = (1.0 - w_lstart[in_lstart_zone]) * stair_base_z + w_lstart[in_lstart_zone] * elevation[in_lstart_zone]

    # End Landing (at top of stairs)
    stair_top_z = stair_base_z + num_steps * step_rise
    dist_landing_end = np.sqrt((X - (stair_end_x + 1.8)) ** 2 + (Y - stair_start_y) ** 2)
    t_lend = np.clip((dist_landing_end - 1.8) / 2.0, 0.0, 1.0)
    w_lend = 3.0 * t_lend**2 - 2.0 * t_lend**3
    in_lend_zone = dist_landing_end <= 3.8
    elevation[in_lend_zone] = (1.0 - w_lend[in_lend_zone]) * stair_top_z + w_lend[in_lend_zone] * elevation[in_lend_zone]

    # Re-enforce exact flat treads within tread interior to ensure absolute flatness
    in_tread_exact = in_stair_length & (d_coord <= half_width)
    elevation[in_tread_exact] = stair_elev_profile[in_tread_exact]

    # 4. Alternative Bypass Gentle Slope (平缓绕行坡道)
    bypass_x_start, bypass_x_end = -6.0, 6.0
    bypass_y = 4.0
    bypass_width = 3.5
    bypass_s = X - bypass_x_start
    bypass_d = np.abs(Y - bypass_y)
    bypass_length = bypass_x_end - bypass_x_start
    bypass_slope = (num_steps * step_rise) / bypass_length
    bypass_profile = stair_base_z + bypass_s * bypass_slope

    bypass_half_w = bypass_width / 2.0
    bypass_shoulder = 1.8
    t_bp = np.clip((bypass_d - bypass_half_w) / bypass_shoulder, 0.0, 1.0)
    w_bp = 3.0 * t_bp**2 - 2.0 * t_bp**3

    in_bypass_len = (bypass_s >= 0.0) & (bypass_s <= bypass_length)
    in_bypass_corridor = in_bypass_len & (bypass_d <= bypass_half_w + bypass_shoulder)
    elevation[in_bypass_corridor] = (1.0 - w_bp[in_bypass_corridor]) * bypass_profile[in_bypass_corridor] + w_bp[in_bypass_corridor] * elevation[in_bypass_corridor]

    # 5. Perimeter Falloff at World Boundary
    border_dist = np.minimum(
        np.minimum(X - (-world_size_x / 2.0), world_size_x / 2.0 - X),
        np.minimum(Y - (-world_size_y / 2.0), world_size_y / 2.0 - Y),
    )
    border_fade_width = 6.0
    border_t = np.clip(border_dist / border_fade_width, 0.0, 1.0)
    border_w = 3.0 * border_t**2 - 2.0 * border_t**3
    elevation = border_w * elevation + (1.0 - border_w) * base_height

    # Clip to limits
    elevation = np.clip(elevation, 0.0, max_height)

    # 6. Winding Patrol Trail Loop Waypoints
    trail_waypoints = np.array([
        [-30.0, -30.0],
        [-22.0, -18.0],
        [-15.0, -8.0],
        [-10.0, 2.0],
        [-6.0, 12.0],   # Stair entrance
        [4.8, 12.0],    # Stair exit
        [6.3, 12.0],    # End landing
        [15.0, 16.0],
        [22.0, 12.0],
        [26.0, -2.0],
        [20.0, -18.0],
        [5.0, -26.0],
        [-15.0, -28.0],
        [-30.0, -30.0], # Loop closure return
    ])

    metadata = {
        "grid_size": grid_size,
        "world_size_x": world_size_x,
        "world_size_y": world_size_y,
        "world_size_z": max_height,
        "resolution_m_per_pixel": round(world_size_x / (grid_size - 1), 4),
        "spawn_point": {
            "x": spawn_center_x,
            "y": spawn_center_y,
            "z": round(float(spawn_flat_elevation), 3),
            "yaw_deg": 45.0,
        },
        "stairs": {
            "start_xyz": [stair_start_x, stair_start_y, round(stair_base_z, 3)],
            "end_xyz": [stair_end_x, stair_start_y, round(stair_top_z, 3)],
            "num_steps": num_steps,
            "step_run_m": step_run,
            "step_rise_m": step_rise,
            "stair_width_m": stair_width,
            "total_elevation_gain_m": round(num_steps * step_rise, 3),
        },
        "bypass_slope": {
            "start_xyz": [bypass_x_start, bypass_y, round(stair_base_z, 3)],
            "end_xyz": [bypass_x_end, bypass_y, round(stair_top_z, 3)],
            "slope_angle_deg": round(float(np.degrees(np.arctan(bypass_slope))), 2),
            "width_m": bypass_width,
        },
        "elevation_stats": {
            "min_m": round(float(np.min(elevation)), 4),
            "max_m": round(float(np.max(elevation)), 4),
            "mean_m": round(float(np.mean(elevation)), 4),
            "std_m": round(float(np.std(elevation)), 4),
        },
        "trail_waypoints": trail_waypoints.tolist(),
    }

    return elevation, metadata


def generate_procedural_textures(output_dir: str, tex_size: int = 512):
    """
    Generate seamless terrain textures for standalone Gazebo visual rendering.
    Creates:
    - dirt_diffuse.png: natural soil / earth
    - grass_diffuse.png: alpine meadow grass
    - rock_diffuse.png: mountain rock / stone
    - flat_normal.png: tangent-space flat normal map with subtle micro-roughness
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(12345)

    # 1. Dirt Diffuse Texture
    dirt = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    base_r, base_g, base_b = 115, 88, 62
    noise_d = rng.normal(0, 10, (tex_size, tex_size))
    dirt[:, :, 0] = np.clip(base_r + noise_d * 1.2, 0, 255).astype(np.uint8)
    dirt[:, :, 1] = np.clip(base_g + noise_d * 0.9, 0, 255).astype(np.uint8)
    dirt[:, :, 2] = np.clip(base_b + noise_d * 0.7, 0, 255).astype(np.uint8)
    Image.fromarray(dirt).save(os.path.join(output_dir, "dirt_diffuse.png"))

    # 2. Grass Diffuse Texture
    grass = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    base_gr_r, base_gr_g, base_gr_b = 75, 115, 52
    noise_g = rng.normal(0, 12, (tex_size, tex_size))
    grass[:, :, 0] = np.clip(base_gr_r + noise_g * 0.8, 0, 255).astype(np.uint8)
    grass[:, :, 1] = np.clip(base_gr_g + noise_g * 1.3, 0, 255).astype(np.uint8)
    grass[:, :, 2] = np.clip(base_gr_b + noise_g * 0.6, 0, 255).astype(np.uint8)
    Image.fromarray(grass).save(os.path.join(output_dir, "grass_diffuse.png"))

    # 3. Rock Diffuse Texture
    rock = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    base_rk_r, base_rk_g, base_rk_b = 110, 112, 115
    noise_r = rng.normal(0, 14, (tex_size, tex_size))
    rock[:, :, 0] = np.clip(base_rk_r + noise_r, 0, 255).astype(np.uint8)
    rock[:, :, 1] = np.clip(base_rk_g + noise_r * 0.98, 0, 255).astype(np.uint8)
    rock[:, :, 2] = np.clip(base_rk_b + noise_r * 0.95, 0, 255).astype(np.uint8)
    Image.fromarray(rock).save(os.path.join(output_dir, "rock_diffuse.png"))

    # 4. Flat Normal Map (RGB: 128, 128, 255 with subtle micro-perturbation)
    normal = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    normal[:, :, 0] = np.clip(128 + rng.normal(0, 2, (tex_size, tex_size)), 0, 255).astype(np.uint8)
    normal[:, :, 1] = np.clip(128 + rng.normal(0, 2, (tex_size, tex_size)), 0, 255).astype(np.uint8)
    normal[:, :, 2] = 255
    Image.fromarray(normal).save(os.path.join(output_dir, "flat_normal.png"))


def save_heightmaps(elevation: np.ndarray, metadata: dict, output_dir: str):
    """
    Save 16-bit PNG (standard Gazebo heightmap), 8-bit PNG (fallback compatibility),
    8-bit preview PNG, terrain textures, and metadata JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_z = metadata["world_size_z"]

    norm_elev = np.clip(elevation / max_z, 0.0, 1.0)

    # 16-bit uint16 PNG (0 to 65535) - Gazebo standard
    elev_16bit = (norm_elev * 65535.0).astype(np.uint16)
    img_16 = Image.fromarray(elev_16bit)
    path_16 = os.path.join(output_dir, "heightmap.png")
    img_16.save(path_16)
    print(f"[SUCCESS] Saved 16-bit heightmap to: {path_16}")

    # 8-bit uint8 PNG (0 to 255) - Gazebo 8-bit fallback
    elev_8bit = (norm_elev * 255.0).astype(np.uint8)
    img_8 = Image.fromarray(elev_8bit, mode="L")
    path_8bit = os.path.join(output_dir, "heightmap_8bit.png")
    img_8.save(path_8bit)
    print(f"[SUCCESS] Saved 8-bit heightmap to: {path_8bit}")

    # 8-bit PNG visual preview
    path_preview = os.path.join(output_dir, "heightmap_preview.png")
    img_8.save(path_preview)
    print(f"[SUCCESS] Saved 8-bit preview to: {path_preview}")

    # Generate procedural materials/textures
    generate_procedural_textures(output_dir)
    print(f"[SUCCESS] Saved procedural terrain textures to: {output_dir}")

    # Metadata JSON
    path_json = os.path.join(output_dir, "terrain_metadata.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Saved metadata to: {path_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parametric mountain heightmap for Gazebo")
    parser.add_argument("--grid-size", type=int, default=1025, choices=[513, 1025, 2049], help="Grid resolution ((2^N)+1)")
    parser.add_argument("--world-size", type=float, default=100.0, help="Horizontal terrain dimension in meters (default: 100m)")
    parser.add_argument("--max-height", type=float, default=15.0, help="Maximum height range in meters (default: 15m)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "models", "mountain_terrain", "materials", "textures")
    else:
        output_dir = args.output_dir

    print(f"Generating terrain: grid={args.grid_size}x{args.grid_size}, size={args.world_size}m, max_height={args.max_height}m")
    elev, meta = generate_terrain(
        grid_size=args.grid_size,
        world_size_x=args.world_size,
        world_size_y=args.world_size,
        max_height=args.max_height,
    )
    save_heightmaps(elev, meta, output_dir)
    print("\nElevation Stats:")
    print(f"  Min: {meta['elevation_stats']['min_m']} m")
    print(f"  Max: {meta['elevation_stats']['max_m']} m")
    print(f"  Mean: {meta['elevation_stats']['mean_m']} m")
    print(f"  Stairs elevation gain: {meta['stairs']['total_elevation_gain_m']} m ({meta['stairs']['num_steps']} steps)")
    print(f"  Bypass slope angle: {meta['bypass_slope']['slope_angle_deg']} deg")
    print(f"  Spawn Point: {meta['spawn_point']}")
