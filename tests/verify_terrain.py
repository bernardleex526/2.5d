"""
Comprehensive Verification Test Suite for Mountain Terrain Heightmap and Gazebo Simulation Package
Designed for DEEP Robotics M20 Pro Wheeled-Legged Robot Mountain Patrol Testing.
"""

import os
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


def test_simulation_assets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_root = os.path.dirname(script_dir)

    print("=== 1. Checking File Existence ===")
    required_files = [
        os.path.join(sim_root, "scripts", "generate_mountain_terrain.py"),
        os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "heightmap.png"),
        os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "heightmap_8bit.png"),
        os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "heightmap_preview.png"),
        os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "dirt_diffuse.png"),
        os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "grass_diffuse.png"),
        os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "rock_diffuse.png"),
        os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "flat_normal.png"),
        os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "terrain_metadata.json"),
        os.path.join(sim_root, "models", "mountain_terrain", "model.config"),
        os.path.join(sim_root, "models", "mountain_terrain", "model.sdf"),
        os.path.join(sim_root, "worlds", "mountain_patrol.world"),
        os.path.join(sim_root, "worlds", "mountain_patrol.sdf"),
    ]

    for fpath in required_files:
        assert os.path.exists(fpath), f"Missing file: {fpath}"
        size_kb = os.path.getsize(fpath) / 1024.0
        print(f"  [OK] {os.path.relpath(fpath, sim_root)} ({size_kb:.1f} KB)")

    print("\n=== 2. Validating Heightmap Image Properties ===")
    # 16-bit PNG
    hm_path_16 = os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "heightmap.png")
    with Image.open(hm_path_16) as img16:
        width, height = img16.size
        mode = img16.mode
        print(f"  [16-bit] Dimensions: {width} x {height}, Mode: {mode}")
        assert width == height, f"Heightmap must be square, got {width}x{height}"
        n_pow = np.log2(width - 1)
        assert np.isclose(n_pow, np.round(n_pow)), f"Width-1 must be power of 2, got {width}"
        assert mode == "I;16" or mode == "I", f"Expected 16-bit mode, got {mode}"
        arr16 = np.array(img16, dtype=np.uint16)
        min_v, max_v = arr16.min(), arr16.max()
        print(f"  [16-bit] Value range: [{min_v}, {max_v}] (uint16 bounds: [0, 65535])")
        assert min_v >= 0 and max_v <= 65535, "Value outside uint16 bounds"

    # 8-bit fallback PNG
    hm_path_8 = os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "heightmap_8bit.png")
    with Image.open(hm_path_8) as img8:
        w8, h8 = img8.size
        mode8 = img8.mode
        print(f"  [8-bit]  Dimensions: {w8} x {h8}, Mode: {mode8}")
        assert (w8, h8) == (width, height), "8-bit heightmap dimensions mismatch"
        assert mode8 == "L", f"Expected 8-bit grayscale mode 'L', got {mode8}"

    print("\n=== 3. Validating Terrain Features & Stair Geometry ===")
    meta_path = os.path.join(sim_root, "models", "mountain_terrain", "materials", "textures", "terrain_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Spawn point
    sp = meta["spawn_point"]
    print(f"  Spawn point: ({sp['x']}, {sp['y']}, {sp['z']})")
    assert sp["x"] == -30.0 and sp["y"] == -30.0, "Spawn coordinates mismatched"
    assert sp["z"] == 1.5, f"Spawn elevation expected 1.5, got {sp['z']}"

    # Stairs specs
    stairs = meta["stairs"]
    print(f"  Stairs: {stairs['num_steps']} steps, rise={stairs['step_rise_m']}m, run={stairs['step_run_m']}m")
    assert stairs["num_steps"] == 24, "Step count mismatch"
    assert np.isclose(stairs["total_elevation_gain_m"], 24 * 0.15), "Stair elevation gain mismatch"

    # Bypass slope
    slope = meta["bypass_slope"]
    print(f"  Bypass slope angle: {slope['slope_angle_deg']} deg")
    assert 16.0 <= slope["slope_angle_deg"] <= 17.5, f"Bypass slope angle out of bounds: {slope['slope_angle_deg']}"

    # Deep inspection of stair elevation grid
    world_z = meta["world_size_z"]
    elev_arr = arr16.astype(np.float64) / 65535.0 * world_z
    xs = np.linspace(-meta["world_size_x"] / 2.0, meta["world_size_x"] / 2.0, meta["grid_size"])
    ys = np.linspace(-meta["world_size_y"] / 2.0, meta["world_size_y"] / 2.0, meta["grid_size"])

    # Verify stair tread flatness across stair_width (lateral Y direction)
    stair_start_x = stairs["start_xyz"][0]
    stair_length = stairs["num_steps"] * stairs["step_run_m"]
    test_x_samples = [stair_start_x + 1.0, stair_start_x + 5.0, stair_start_x + 9.0]
    for tx in test_x_samples:
        ix = np.argmin(np.abs(xs - tx))
        # Tread interior: |y - 12.0| <= 1.4m (within stair_width=3.2m)
        mask_tread_y = np.abs(ys - stairs["start_xyz"][1]) <= 1.4
        tread_elevs = elev_arr[mask_tread_y, ix]
        std_elev = np.std(tread_elevs)
        assert std_elev < 1e-4, f"Tread at x={tx} is not flat! std={std_elev}"
    print("  [OK] Stair tread flatness verified (lateral std < 1e-4 m across tread).")

    # Verify absence of lateral knife-edge cliff drops across the entire stair corridor
    # At any x along the stairs, max adjacent-pixel difference across Y must be <= 0.15m
    for s_offset in np.linspace(0.5, stair_length - 0.5, 10):
        ix = np.argmin(np.abs(xs - (stair_start_x + s_offset)))
        corridor_y_mask = np.abs(ys - stairs["start_xyz"][1]) <= 4.0
        elev_corridor = elev_arr[corridor_y_mask, ix]
        max_jump = np.max(np.abs(np.diff(elev_corridor)))
        assert max_jump <= 0.15, f"Cliff jump detected in stair corridor at x={xs[ix]}: max_jump={max_jump:.3f}m"
    print("  [OK] Stair lateral shoulder continuity verified (zero cliff drops, max adjacent jump <= 0.15m).")

    # Verify landing platform elevations
    # Start landing: around x = stair_start_x - 1.8, y = 12.0
    ix_lstart = np.argmin(np.abs(xs - (stair_start_x - 1.8)))
    iy_stair = np.argmin(np.abs(ys - stairs["start_xyz"][1]))
    z_lstart = elev_arr[iy_stair, ix_lstart]
    assert np.isclose(z_lstart, stairs["start_xyz"][2], atol=0.05), f"Start landing z mismatch: {z_lstart} vs {stairs['start_xyz'][2]}"

    # End landing: around x = stair_end_x + 1.8, y = 12.0
    ix_lend = np.argmin(np.abs(xs - (stairs["end_xyz"][0] + 1.8)))
    z_lend = elev_arr[iy_stair, ix_lend]
    assert np.isclose(z_lend, stairs["end_xyz"][2], atol=0.05), f"End landing z mismatch: {z_lend} vs {stairs['end_xyz'][2]}"
    print("  [OK] Start and end landing platform elevations verified.")

    print("\n=== 4. Validating XML/SDF Files ===")
    xml_files = [
        os.path.join(sim_root, "models", "mountain_terrain", "model.config"),
        os.path.join(sim_root, "models", "mountain_terrain", "model.sdf"),
        os.path.join(sim_root, "worlds", "mountain_patrol.world"),
        os.path.join(sim_root, "worlds", "mountain_patrol.sdf"),
    ]
    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            print(f"  [OK] XML valid: {os.path.relpath(xf, sim_root)} (root tag: <{root.tag}>)")
        except Exception as e:
            raise AssertionError(f"XML parse error in {xf}: {e}")

    # Inspect model.sdf geometry consistency
    model_sdf = os.path.join(sim_root, "models", "mountain_terrain", "model.sdf")
    tree = ET.parse(model_sdf)
    root = tree.getroot()
    sizes = [elem.text.strip() for elem in root.iter("size") if elem.text]
    assert len(sizes) >= 2, "model.sdf must contain heightmap size definitions"
    assert sizes[0] == sizes[-1] == "100.0 100.0 15.0", f"Collision and visual heightmap sizes must match: {sizes}"
    print("  [OK] Collision and visual geometry sizes match (100.0 100.0 15.0).")

    # Check that fdir1 does not contain degenerate zero vector
    fdir1_elems = list(root.iter("fdir1"))
    for f in fdir1_elems:
        assert f.text.strip() != "0 0 0", "Degenerate zero-length vector in fdir1"
    print("  [OK] Friction parameters valid (no degenerate fdir1).")

    print("\n=== 5. Validating Generator CLI Parameterization ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        gen_script = os.path.join(sim_root, "scripts", "generate_mountain_terrain.py")
        cmd = [
            "python", gen_script,
            "--grid-size", "513",
            "--world-size", "100.0",
            "--max-height", "15.0",
            "--output-dir", tmpdir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Generator CLI failed with 513 grid: {res.stderr}"
        assert os.path.exists(os.path.join(tmpdir, "heightmap.png")), "CLI did not create heightmap.png"
        assert os.path.exists(os.path.join(tmpdir, "heightmap_8bit.png")), "CLI did not create heightmap_8bit.png"
        assert os.path.exists(os.path.join(tmpdir, "terrain_metadata.json")), "CLI did not create terrain_metadata.json"
        print("  [OK] Generator CLI verified with parameterized grid (513x513).")

    print("\n[ALL TESTS PASSED] Simulation assets and physics geometry verified 100% sound.")


if __name__ == "__main__":
    test_simulation_assets()
