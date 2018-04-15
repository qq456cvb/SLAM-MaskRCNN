import os
ROOT_DIR = os.path.abspath("Mask_RCNN/")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import mrcnn.utils
import mrcnn.model as modellib
import coco

import cv2
import utils
from matplotlib import pyplot as plt
from TSDF_Python.main import RGB_PATH, ROOT_PATH, OBJ
import glob

# from MaskRCNN import visualize
import numpy as np

# import backend of keras to set GPU memory usage
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
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


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_mask():
    def preserve_small_objs(masks):
        areas = np.array([np.count_nonzero(masks[:, :, i]) for i in range(masks.shape[-1])])
        sorted_idx = np.argsort(areas)
        for i in range(len(sorted_idx)):
            for j in range(i + 1, len(sorted_idx)):
                if np.any(masks[:, :, sorted_idx[i]] & masks[:, :, sorted_idx[j]]):
                    masks[:, :, sorted_idx[j]][masks[:, :, sorted_idx[i]] & masks[:, :, sorted_idx[j]]] = False
        return masks

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    rgb_fn = sorted(glob.glob(RGB_PATH))
    for fn in rgb_fn:
        rgb_img = cv2.imread(fn)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        result = model.detect([rgb_img], verbose=0)[0]
        masks = result['masks']
        masks = masks.astype(np.bool)
        masks = preserve_small_objs(masks)
        cls = np.zeros([rgb_img.shape[0], rgb_img.shape[1]], np.uint8)
        for i in range(masks.shape[2]):
            cls[masks[:, :, i]] = i + 1
        print('processing: {}'.format(fn))
        write_path = os.path.join(ROOT_PATH, OBJ, "mask\\" + fn.split('\\')[-1])
        cv2.imshow('test', cls * 20)
        cv2.waitKey(10)
        cv2.imwrite(write_path, cls)


def slam():
    intrinsic = np.array([[520, 0, 360, 0],
                          [0, 520, 270, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    intrinsic = intrinsic[:3, :3]

    cap = cv2.VideoCapture('seq1.mp4')
    _, img1 = cap.read()
    interval = 30
    cnt = 0
    while cap.isOpened():
        cnt += 1
        if cnt % interval != 0:
            _, _ = cap.read()
            continue

        _, img2 = cap.read()
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        # cv2.imshow('img1', img1)
        # cv2.imshow('img2', img2)
        # cv2.waitKey(0)
        pts1, pts2 = utils.match(img1, img2)

        ess_mat, idx = cv2.findEssentialMat(pts1, pts2, intrinsic, cv2.RANSAC)
        print(ess_mat)

        pts1 = pts1[np.squeeze(idx.astype(np.bool)), :]
        pts2 = pts2[np.squeeze(idx.astype(np.bool)), :]
        img_pts = np.concatenate([np.expand_dims(pts1, 1), np.expand_dims(pts2, 1)], axis=1)
        print(img_pts.shape)
        RT = utils.estimate_RT_from_E(ess_mat, img_pts, intrinsic)
        print(RT)

        intrinsic_inv = np.linalg.inv(intrinsic)
        fund_mat = np.matmul(np.matmul(intrinsic_inv.transpose(), ess_mat), intrinsic_inv)
        H1 = np.zeros([3, 3])
        H2 = np.zeros_like(H1)
        cv2.stereoRectifyUncalibrated(pts1, pts2, fund_mat, (img1.shape[1], img1.shape[0]), H1, H2)
        print(H1)
        print(H2)
        img1_warped = cv2.warpPerspective(img1, H1, (0, 0))
        img2_warped = cv2.warpPerspective(img2, H2, (0, 0))
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img1_warped)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img2_warped)
        plt.show()
        # block_matcher = cv2.StereoBM_create(32, 15)
        block_matcher = cv2.StereoSGBM.create(numDisparities=32, blockSize=13, uniquenessRatio=8, speckleWindowSize=50,
                                              speckleRange=1, mode=cv2.STEREO_SGBM_MODE_HH4)
        # cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE, 0, 5)
        # disp = block_matcher.compute(img1_warped, img1_warped)
        disp = block_matcher.compute(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY))

        plt.imshow(disp.astype(np.float32))
        plt.show()

        # proj1 = np.eye(4)[:3, :]
        # proj2 = np.zeros([3, 4])
        # proj2[:3, :3] = RT[:3, :3].transpose()
        # proj2[:3, -1] = -np.dot(RT[:3, :3], RT[:3, -1])
        # proj1 = np.matmul(H1, proj1)
        # proj2 = np.matmul(H2, proj2)
        # x, y = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
        # pts_left = np.dstack([x, y]).reshape(-1, 2).transpose()
        # x = x + disp
        # pts_right = np.dstack([x, y]).reshape(-1, 2).transpose()
        # pts_all = cv2.triangulatePoints(proj1, proj2, pts_left[:, :1000], pts_right[:, :1000])

        # with open('test.ply', 'w+') as f:
        #     f.write('ply format ascii 1.0\n'
        #             'element vertex %d\n'
        #             'property float x\n'
        #             'property float y\n'
        #             'property float z\n'
        #             'end_header\n' % pts_all.shape[1])
        #     # for i in range(pts_all.shape[1]):
        #     #     f.write('%f %f %f\n' % (pts_all[0, i], pts_all[1, i], pts_all[2, i]))
        #     f.close()
        img1 = img2
        break
    # print(pts1.shape)
    # print(np.sum(np.diag(abs(np.matmul(np.matmul(np.hstack([pts2, np.ones([pts2.shape[0], 1])]), fund_mat),  np.hstack([pts1, np.ones([pts1.shape[0], 1])]).transpose())))))

    # cv2.drawKeypoints(img1, pts1, img1)
    # plt.imshow(img1)
    # plt.imshow(img2)
    # plt.show()

    # for i in range(1, 2):
    #     img = cv2.imread('test_images/%d.jpg' % i)
    #     img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #     r = model.detect([img], verbose=0)[0]
    #     print(r)
    #     visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
    #                                 class_names, r['scores'], figsize=(6, 8))


if __name__ == '__main__':

    get_mask()




