# SLAM-MaskRCNN

This project implemented real-time indoor objects segmentation and 3D reconstruction.

We used fine-tuned [MaskRCNN](https://github.com/matterport/Mask_RCNN) doing instance segmentation for 51 different objects and build 3D model by Truncated Signed Distance Function Volume Reconstruction.

By now, there are two steps to execute the pipe line. 

First, download datasets from [RGB-D SLAM datasets](https://vision.in.tum.de/data/datasets/rgbd-dataset/download). Using [mask_process.py](https://github.com/qq456cvb/SLAM-MaskRCNN/blob/master/Mask_RCNN/mask_process.py) to generate mask images for specific datasets.

Second, change configuration in [kernel.cpp](https://github.com/qq456cvb/SLAM-MaskRCNN/blob/master/src/SfM_CUDA/kernel.cpp) to execute TSDF.
