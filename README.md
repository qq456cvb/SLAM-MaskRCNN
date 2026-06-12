# SLAM-MaskRCNN

Semantic 3D reconstruction of indoor scenes: **Mask R-CNN instance segmentation fused into a TSDF volume** with CUDA, on [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) sequences. The result is a colored 3D reconstruction in which each voxel also carries an object-instance label, rendered live by a CUDA ray-casting viewer.

## Pipeline

The pipeline has two stages:

### 1. Instance masks (`Mask_RCNN/`)

A vendored copy of [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) (TensorFlow/Keras) plus scripts to batch-generate per-frame instance masks:

- `mask_process.py` runs COCO-pretrained Mask R-CNN (`mask_rcnn_coco.h5` is downloaded automatically on first run) over a sequence's `rgb/` folder and writes label-encoded mask PNGs (pixel value = instance id).
- `dmask.py` / `mask_image.py` / `multi_mask_image.py` contain the detection helpers and visualization variants.
- Edit the `RGB_PATH` / `ROOT_PATH` / `OBJ` constants at the top of `mask_process.py` to point at your dataset before running.

### 2. TSDF fusion (`src/SfM_CUDA/`)

A CUDA implementation of Truncated Signed Distance Function volume fusion (`tsdf.cu`), extended to integrate instance labels:

- Depth, RGB and mask frames are timestamp-matched and back-projected using camera poses from the sequence's `groundtruth.txt`, so the focus is on semantic fusion rather than tracking.
- Per-voxel, the volume accumulates signed distance, color, and an object-instance histogram; `configuration.h` exposes the Mask R-CNN error-rate prior and the duplicate-instance merging threshold.
- `viewer.cu` ray-casts the fused volume into an OpenCV window, spinning the reconstruction once fusion finishes.
- `kernel.cpp` is the entry point: set the dataset paths, the camera intrinsics (defaults are TUM Freiburg-2: fx 520.9, fy 521.0, cx 325.1, cy 249.7), and the timestamp window there, then build (a `Makefile` lives in `build/`; needs CUDA and OpenCV).

`src/TSDF_CPP/` (OpenGL-shader TSDF prototype) and `src/TSDF_Python/` (NumPy prototype, plus an experimental two-view SfM in `src/main.py`) are earlier iterations kept for reference.

## Usage

1. Download a sequence from the [TUM RGB-D benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) (it must include `rgb/`, `depth/` and `groundtruth.txt`).
2. Generate masks: configure and run `python mask_process.py` inside `Mask_RCNN/` (TF 1.x, Keras, see `Mask_RCNN/requirements.txt`).
3. Fuse: point `kernel.cpp` at the `rgb`/`depth`/`mask` folders and the ground-truth trajectory, build `SfM_CUDA`, and run it to watch the labeled reconstruction appear.

## Acknowledgement

Instance segmentation builds on [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) (MIT license); the RGB-D data and trajectory format follow the TUM RGB-D benchmark.
