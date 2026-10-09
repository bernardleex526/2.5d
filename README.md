# Mountain Patrol Gazebo Simulation Package

Simulation environment for DEEP Robotics M20 Pro (云深处 M20 Pro) Wheeled-Legged Robot mountain patrol, autonomous mapping, and gait switching.

## Features
- **Parametric Heightmap (1025 x 1025, 16-bit uint16 & 8-bit uint8 PNG)**: Continuous elevation modeling conforming to Gazebo $(2^N)+1$ requirements.
- **Staging Base Camp**: Flat spawn pad at `(-30, -30, 1.5)` with cubic Hermite blending for stable robot initialization.
- **Rolling Mountain Hills**: Multi-frequency procedural slopes (10°~25°) for testing 3D LIO and elevation mapping.
- **Continuous Stone Stairs (连续石阶梯)**: 24 steps with 0.15m riser height and 0.45m tread depth, climbing 3.6m vertically. Features 2.0m lateral earthen shoulder blending to eliminate knife-edge cliff drops and mesh singularities.
- **Entrance & Exit Landing Platforms**: Smooth landing pads at the foot and top of the staircase for safe gait transitions.
- **Bypass Gentle Slope (对比测试绕行坡道)**: 16.7° smooth slope for comparing wheeled rolling vs stepping locomotion.
- **Closed-Loop Patrol Trail**: Winding mountain trail with loop closure return to base station for SLAM PGO validation.
- **Self-Contained Materials**: Procedural seamless diffuse and normal textures (dirt, grass, rock, flat normal) eliminating dependencies on system media directories.

## Architecture & Technical Design Document
The complete system engineering plan, Requirements Traceability Matrix (RTM), ambiguous requirements clarification list, SLAM algorithm evaluation (`lightning-lm-deep-robotics` vs `Point-LIO + Scan Context + GTSAM`), Livox Mid-360 sensor integration, and 2.5D elevation mapping pipeline are documented in:
👉 [**docs/mountain_patrol_robot_plan.md**](docs/mountain_patrol_robot_plan.md)


## Directory Structure
```
mountain_patrol_sim/
├── scripts/
│   └── generate_mountain_terrain.py     # Parametric generator
├── models/
│   └── mountain_terrain/
│       ├── model.config                 # Gazebo model config
│       ├── model.sdf                    # Terrain collision & visual SDF
│       └── materials/textures/
│           ├── heightmap.png            # 16-bit uint16 heightmap for Gazebo
│           ├── heightmap_8bit.png       # 8-bit uint8 heightmap fallback
│           ├── heightmap_preview.png    # 8-bit visual preview
│           ├── dirt_diffuse.png         # Soil diffuse texture
│           ├── grass_diffuse.png        # Grass diffuse texture
│           ├── rock_diffuse.png         # Rock diffuse texture
│           ├── flat_normal.png          # Normal map texture
│           └── terrain_metadata.json    # Exact coordinates and parameters
├── worlds/
│   ├── mountain_patrol.world            # Gazebo Classic 11 world file
│   └── mountain_patrol.sdf              # Modern Gazebo (Harmonic/Fortress) SDF world
└── tests/
    └── verify_terrain.py                # Comprehensive asset & physics test suite
```

## How to Launch in Gazebo

### Gazebo Classic 11
```bash
# Add models directory to Gazebo model path
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)/models

# Launch the world
gazebo worlds/mountain_patrol.world --verbose
```

### Modern Gazebo (Gz Sim / Gazebo Harmonic)
```bash
# Add models directory to GZ resource path
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$(pwd)/models

# Launch with Gz Sim
gz sim -r worlds/mountain_patrol.sdf
```

## Running Verification Tests
```bash
python tests/verify_terrain.py
```
