# SLAM-MaskRCNN

<!-- README refined by Cursor -->

RGB-D SLAM reconstruction with Mask R-CNN instance masks and TSDF fusion.

## Overview

This repository contains Python, Jupyter Notebook, C++, CUDA code from an older research, course, or prototype project. The README has been refreshed to make the repository easier to scan while preserving the original notes below.

## Repository Contents

- `Mask_RCNN/`
- `src/`

## Setup

- This legacy repo does not pin a full environment. Start from the language/toolchain implied by the source files, then install missing packages as reported by the runtime.

## Usage

- inspect the source directories listed below; many of these older repos were kept as research prototypes rather than packaged applications.

## Data and Artifacts

No new large artifact is stored in this repository. If a dataset or checkpoint is required, follow the links and notes in the original section below.

## Status

This is a `Batch B` cleanup pass for a legacy repository. Commands may require dependency/version adjustments on a modern machine.

## License

No explicit license file was found in this checkout; check the original project context before reusing code.

## Original Notes

# SLAM-MaskRCNN

This project implemented real-time indoor objects segmentation and 3D reconstruction.

We used fine-tuned [MaskRCNN](https://github.com/matterport/Mask_RCNN) doing instance segmentation for 51 different objects and build 3D model by Truncated Signed Distance Function Volume Reconstruction.

By now, there are two steps to execute the pipe line. 

First, download datasets from [RGB-D SLAM datasets](https://vision.in.tum.de/data/datasets/rgbd-dataset/download). Using [mask_process.py](https://github.com/qq456cvb/SLAM-MaskRCNN/blob/master/Mask_RCNN/mask_process.py) to generate mask images for specific datasets.

Second, change configuration in [kernel.cpp](https://github.com/qq456cvb/SLAM-MaskRCNN/blob/master/src/SfM_CUDA/kernel.cpp) to execute TSDF.
