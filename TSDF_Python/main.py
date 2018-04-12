import numpy as np
import cv2
import glob
import TSDF_Python.tsdf_utils as tsdf_utils
from TSDF_Python.tsdf import TSDF
#import MaskRCNN.model as modellib
import os
#import MaskRCNN.coco as coco
# import MaskRCNN.visualize as visualize
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ('BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush')

OBJ = 'desk'

ROOT_PATH = "D:\\rgb-datasets"
RGB_PATH = os.path.join(ROOT_PATH, OBJ, "rgb\\*.png")
DEPTH_PATH = os.path.join(ROOT_PATH, OBJ, "depth\\*.png")
MASK_PATH = os.path.join(ROOT_PATH, OBJ, "mask\\*.png")
TRAJ_PATH = os.path.join(ROOT_PATH, OBJ, "groundtruth.txt")
WRITE_PATH = "/Users/qq456cvb/Documents/tsdf-fusion/data/can/"
READ_PATH = "/Users/qq456cvb/Documents/tsdf-fusion/data/rgbd-frames/"


#class InferenceConfig(coco.CocoConfig):
   # Set batch size to 1 since we'll be running inference on
   # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
 #  GPU_COUNT = 1
 #  IMAGES_PER_GPU = 1

# def test():
#     tsdf = TSDF([585, 585, 320, 240])
#     for i in range(185, 190):
#         print(i)
#         rgb_img = cv2.imread(READ_PATH + 'frame-%06d.color.png' % i)
#         depth_img = cv2.imread(READ_PATH + 'frame-%06d.depth.png' % i, cv2.IMREAD_ANYDEPTH)
#         extrinsic = np.linalg.inv(np.loadtxt(READ_PATH + 'frame-%06d.pose.txt' % i, np.float64))
#         tsdf.parse_frame2(depth_img, rgb_img, extrinsic)
#         # cv2.imshow('depth', depth_img * 10)
#         # cv2.imshow('img', rgb_img)
#         # cv2.waitKey(0)
#     tsdf_utils.show_model(tsdf)


if __name__ == '__main__':
    rgb_fn = sorted(glob.glob(RGB_PATH))
    depth_fn = sorted(glob.glob(DEPTH_PATH))
    mask_fn = sorted(glob.glob(MASK_PATH))

    depth_timestamps = np.array([fn.split('\\')[-1].strip('.png')[5:] for fn in depth_fn]).astype(np.double)
    rgb_timestamps = np.array([fn.split('\\')[-1].strip('.png')[5:] for fn in rgb_fn]).astype(np.double)
    # mask_timestamps = np.array([fn.split('/')[-1].split('_')[0][5:] for fn in mask_fn]).astype(np.double)
    # depth_timestamps = np.sort(depth_timestamps)
    # mask_timestamps = np.sort(mask_timestamps)

    traj = tsdf_utils.read_traj(TRAJ_PATH)
    j = 0
    # TODO: add distortion
    tsdf = TSDF([520.9, 521.0, 325.1, 249.7])
    # np.savetxt(WRITE_PATH + "camera-intrinsics.txt", tsdf.intrinsic[:3, :3])
    dist = np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])
    begin = 68164
    end = 68164.37

    # model = modellib.MaskRCNN(mode="inference", model_dir='./', config=InferenceConfig())

    # Load weights trained on MS-COCO
    # model.load_weights("../mask_rcnn_coco.h5", by_name=True)

    for i in range(3000):
        if i >= depth_timestamps.shape[0]:
            break
        if depth_timestamps[i] < begin or depth_timestamps[i] > end:
            continue
        while depth_timestamps[i] < rgb_timestamps[j]:
            i += 1
        while rgb_timestamps[j] < depth_timestamps[i]:
            j += 1
        depth_img = cv2.imread(depth_fn[i], cv2.IMREAD_ANYDEPTH)
        # mask_img = cv2.imread(mask_fn[j], cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread(rgb_fn[j])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        #result = model.detect([rgb_img], verbose=0)[0]
        # visualize.display_instances(rgb_img, result['rois'], result['masks'], result['class_ids'],
        #                     class_names, result['scores'])
        #masks = result['masks']
        #masks = masks.astype(np.bool)
        masks = cv2.imread(mask_fn[j])
        # print(masks.shape)
        # cv2.imshow('test', masks)
        # cv2.waitKey(0)
        #     cv2.imshow('mask%d' % k, masks[:, :, k] * 255)
        #
        # # cv2.imshow('img', rgb_img)
        # cv2.waitKey(0)
        # rgb_img = tsdf_utils.fix_distortion(rgb_img, tsdf.intrinsic[:3, :3], dist)
        # depth_img = tsdf_utils.fix_distortion(depth_img, tsdf.intrinsic[:3, :3], dist)
        # cv2.imshow('img', rgb_img)
        # cv2.waitKey(0)

        # print(depth_timestamps[i])
        # print(mask_timestamps[j])
        # depth_img[mask_img == 0] = 0
        # rgb_img[mask_img == 0] = 0
        # _, mean_depth = tsdf_utils.filter_gaussian(depth_img)
        # mean_depth = 0.8
        # cv2.imshow('mask', mask_img)
        # cv2.imshow('depth', depth_img)
        # cv2.imshow('img', rgb_img)
        # cv2.waitKey(0)
        mean_depth = np.mean(depth_img[depth_img > 0])
        # print(mean_depth)

        extrinsic = None
        for k in range(traj.shape[0]):
            if traj[k, 0] < depth_timestamps[i]:
                continue
            # extrinsic = traj[k, 1:]
            print(traj[k, 0], depth_timestamps[i])
            t = (depth_timestamps[i] - traj[k-1, 0]) / (traj[k, 0] - traj[k-1, 0])
            assert 0 <= t <= 1
            extrinsic = np.concatenate([(traj[k, 1:4] - traj[k-1, 1:4]) * t + traj[k-1, 1:4],
                                        tsdf_utils.slerp(traj[k-1, -4:], traj[k, -4:], t)])

            break

        extrinsic = tsdf_utils.parse_pos(extrinsic)
        # write_file(depth_img, rgb_img, extrinsic, i)
        tsdf.parse_frame(depth_img, rgb_img, extrinsic, mean_depth, masks)

    tsdf_utils.show_model(tsdf)

