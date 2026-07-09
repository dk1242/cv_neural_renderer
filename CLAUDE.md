# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VisionEngine is a from-scratch computer vision / SLAM pipeline in C++20. Core CV algorithms (convolution, Harris, FAST, ORB, RANSAC, epipolar geometry, pose recovery, triangulation) are implemented by hand rather than using OpenCV's equivalents — OpenCV is used only for `cv::Mat`, image I/O, and basic linear algebra primitives. There's also a small OpenGL 4.3 renderer for visualizing the resulting 3D map/trajectory.

## Build & run

```bash
./run.sh
```

This configures an out-of-source build in `build/`, builds with `-j$(nproc)`, and launches `build/app/VisionEngineApp`. It also sets `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` to force the discrete NVIDIA GPU on hybrid-GPU laptops — if OpenGL renders on the wrong GPU or fails, check this first (verify with `glxinfo | grep "OpenGL renderer"`).

Manual build:
```bash
cmake -S . -B build
cmake --build build -- -j$(nproc)
```

There is no test suite and no lint config in this repo currently.

### Dependencies

OpenCV, OpenGL, GLFW3, GLM (headers), and a vendored GLAD loader (`renderer/glad/`). On Fedora: `sudo dnf install cmake gcc-c++ glfw-devel glm-devel mesa-libGL-devel opencv-devel`. `renderer/SETUP_TEMPLATE.md` has the full OpenGL project-setup rationale (why GLAD must be generated for gl4.3+ core, PRIME offload details, shader version requirements) if bootstrapping OpenGL bits elsewhere.

## Architecture

Five static libraries plus one executable, wired together via `add_subdirectory` in the root `CMakeLists.txt`. Dependency order: `vision` and `geometry` are independent leaves; `slam` depends on both; `renderer` is independent (OpenCV + OpenGL/GLFW/GLM only); `app` links `vision`, `geometry`, `slam`, `renderer`.

- **`core/`** — header-only utilities (`image.hpp`, `logger.hpp`, `profiler.hpp`, `timer.hpp`). Its `CMakeLists.txt` is currently empty (no library target), so these headers aren't yet wired into the build via `find_package`/`target_link_libraries` — check before assuming `core::` symbols are linkable.
- **`vision/`** — low-level image processing and feature pipeline, hand-implemented: `convolution` → `gaussian`/`sobel` → `canny`; `harris` and `fast` corner detectors; `orb` descriptor extraction; `hamming` distance + `brute_force`/`descriptor_matcher` for matching; `image_pyramid`; `visualizer` for `cv::Mat` display. Public headers in `vision/include/vision/`, impls in `vision/src/`.
- **`geometry/`** — multi-view geometry, all `cv::Mat`-based: `homography`, `normalization` (point normalization for numerical stability), `fundamental_matrix` (normalized 8-point algorithm), `essential_matrix`, `pose_recovery` (decompose E → R, t), `triangulation`, and `ransac` (generic robust estimator used for fundamental matrix fitting). Headers are inconsistently `.h` vs `.hpp` in this directory — match whichever the file already uses when adding new ones.
- **`slam/`** — the actual visual-odometry pipeline (`visual_odometry.hpp/.cpp`), consuming `vision` (FAST + ORB + matcher) and `geometry` (RANSAC fundamental → essential → pose recovery → triangulation) frame-to-frame. `VisualOdometry::processFrame` is the main entry point; it maintains `Frame`/`CameraPose`/`MapPoint`/`RelativePose` state and accumulates `mapPoints()`/`cameraCenters()`/`currentPose()`. `keyframe.hpp` and `map.hpp` exist as separate small classes (`Keyframe`, `Map`) but are not currently used by `VisualOdometry` — it manages its own `trajectory_`/`mapPoints_` vectors directly.
- **`renderer/`** — self-contained OpenGL 4.3 core-profile point-cloud/trajectory viewer, structured per `renderer/SETUP_TEMPLATE.md`'s conventions: `shader` (compile/link wrapper), `vao`/`vbo` (buffer wrappers), `camera` (view/projection state) + `fly_camera` (WASD/mouse-look input), `opengl_renderer` (window/GL context lifecycle, draw loop). `glad/` is the vendored, generated GL 4.3 loader — do not hand-edit it. GLSL sources live in `renderer/shaders/` and are copied next to the binary at build time.
- **`app/`** — `main.cpp` is the current entry point: loads a KITTI-style image sequence from a hardcoded dataset path, runs `slam::VisualOdometry::processFrame` over it, then hands the resulting map points and final pose to `renderer::OpenGLRenderer` for an interactive 3D view. `app/epipolar_demo.cpp` exists but is **not** in `app/CMakeLists.txt`'s source list — it's not currently built.
- **`benchmarks/`** — `benchmark_convolution.cpp` and `benchmark_orb.cpp` exist but are not wired into any `CMakeLists.txt`/`add_subdirectory` — not currently built.

### Data flow through `VisualOdometry::processFrame`

1. Detect FAST corners + extract ORB descriptors on the incoming frame (`createFrame`).
2. Match descriptors against `previousFrame_` (`DescriptorMatcher`, Hamming distance).
3. Robustly estimate the fundamental matrix via RANSAC (`RansacFundamental`), then derive the essential matrix and recover relative `(R, t)` (`PoseRecovery`).
4. Triangulate inlier correspondences into 3D points, accumulated into `mapPoints_`.
5. Compose the relative pose onto `currentPose_`/`previousPose_` to build the trajectory.

Pose convention: `CameraPose` stores `Rwc`/`twc` (camera-to-world rotation/translation, i.e. camera pose *in* world frame), not the world-to-camera view matrix — `app/main.cpp`'s `buildViewPoseFromCameraPose` inverts this to build the OpenGL view matrix.
