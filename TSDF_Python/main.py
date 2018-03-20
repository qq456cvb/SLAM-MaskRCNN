import numpy as np
import cv2
import glob
import tsdf_utils
from tsdf import TSDF


RGB_PATH = "D://rgb-datasets//cokecan//rgb//*.png"
DEPTH_PATH = "D://rgb-datasets//cokecan//depth//*.png"
MASK_PATH = "D://rgb-datasets//cokecan//gray_mask//*.png"


if __name__ == '__main__':
    rgb_fn = glob.glob(RGB_PATH)
    depth_fn = glob.glob(DEPTH_PATH)
    mask_fn = glob.glob(MASK_PATH)

    depth_timestamps = np.array([fn.split('\\')[-1].strip('.png')[5:] for fn in depth_fn]).astype(np.double)
    mask_timestamps = np.array([fn.split('\\')[-1].strip('.png')[5:] for fn in mask_fn]).astype(np.double)

    traj = tsdf_utils.read_traj("D://rgb-datasets//cokecan//groundtruth.txt")
    j = 0
    # TODO: add distortion
    tsdf = TSDF([520.9, 521.0, 325.1, 249.7])
    for i in range(20):
        if i > depth_timestamps.shape[0]:
            break
        while depth_timestamps[i] < mask_timestamps[j]:
            i += 1
        while mask_timestamps[j] < depth_timestamps[i]:
            j += 1
        depth_img = cv2.imread(depth_fn[i], cv2.IMREAD_ANYDEPTH)
        mask_img = cv2.imread(mask_fn[j], cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread(rgb_fn[j])

        print(depth_timestamps[i])
        print(mask_timestamps[j])
        depth_img[mask_img == 0] = 0
        rgb_img[mask_img == 0] = 0
        depth_img, mean_depth = tsdf_utils.filter_gaussian(depth_img)
        # cv2.imshow('mask', mask_img)
        cv2.imshow('depth', depth_img)
        cv2.imshow('img', rgb_img)
        cv2.waitKey(30)

        extrinsic = None
        for k in range(traj.shape[0]):
            if traj[k, 0] < depth_timestamps[i]:
                continue
            extrinsic = traj[k, 1:]
            break

        extrinsic = tsdf_utils.parse_pos(extrinsic)
        tsdf.parse_frame(depth_img, rgb_img, extrinsic, mean_depth)

    tsdf_utils.show_model(tsdf)

