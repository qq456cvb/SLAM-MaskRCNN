import numpy as np
import cv2
import glob
import TSDF_Python.tsdf_utils as tsdf_utils
from TSDF_Python.tsdf import TSDF

OBJ = 'can'

RGB_PATH = "/Users/qq456cvb/Documents/rgb-datasets/" + OBJ + "/rgb/*.png"
DEPTH_PATH = "/Users/qq456cvb/Documents/rgb-datasets/" + OBJ + "/depth/*.png"
MASK_PATH = "/Users/qq456cvb/Documents/rgb-datasets/" + OBJ + "/mask/gray_mask/*.png"
WRITE_PATH = "/Users/qq456cvb/Documents/tsdf-fusion/data/can/"
READ_PATH = "/Users/qq456cvb/Documents/tsdf-fusion/data/rgbd-frames/"


def write_file(depth_img, rgb_img, extrinsic, i):
    cv2.imwrite(WRITE_PATH + 'frame-%06d.color.png' % i, rgb_img)
    cv2.imwrite(WRITE_PATH + 'frame-%06d.depth.png' % i, depth_img)
    np.savetxt(WRITE_PATH + 'frame-%06d.pose.txt' % i, np.linalg.inv(extrinsic))


def test():
    tsdf = TSDF([585, 585, 320, 240])
    for i in range(185, 190):
        print(i)
        rgb_img = cv2.imread(READ_PATH + 'frame-%06d.color.png' % i)
        depth_img = cv2.imread(READ_PATH + 'frame-%06d.depth.png' % i, cv2.IMREAD_ANYDEPTH)
        extrinsic = np.linalg.inv(np.loadtxt(READ_PATH + 'frame-%06d.pose.txt' % i, np.float64))
        tsdf.parse_frame2(depth_img, rgb_img, extrinsic)
        # cv2.imshow('depth', depth_img * 10)
        # cv2.imshow('img', rgb_img)
        # cv2.waitKey(0)
    tsdf_utils.show_model(tsdf)


if __name__ == '__main__':
    # test()
    rgb_fn = sorted(glob.glob(RGB_PATH))
    depth_fn = sorted(glob.glob(DEPTH_PATH))
    mask_fn = sorted(glob.glob(MASK_PATH))

    depth_timestamps = np.array([fn.split('/')[-1].strip('.png')[5:] for fn in depth_fn]).astype(np.double)
    rgb_timestamps = np.array([fn.split('/')[-1].strip('.png')[5:] for fn in rgb_fn]).astype(np.double)
    # mask_timestamps = np.array([fn.split('/')[-1].split('_')[0][5:] for fn in mask_fn]).astype(np.double)
    # depth_timestamps = np.sort(depth_timestamps)
    # mask_timestamps = np.sort(mask_timestamps)

    traj = tsdf_utils.read_traj("/Users/qq456cvb/Documents/rgb-datasets/" + OBJ + "/groundtruth.txt")
    j = 0
    # TODO: add distortion
    tsdf = TSDF([520.9, 521.0, 325.1, 249.7])
    np.savetxt(WRITE_PATH + "camera-intrinsics.txt", tsdf.intrinsic[:3, :3])
    dist = np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])
    begin = 719223
    end = 71930
    # cv2.imshow('test', cv2.imread('/Users/qq456cvb/Documents/tsdf-fusion/data/rgbd-frames/frame-000150.depth.png') * 10)
    # cv2.waitKey(0)
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
        mask_img = cv2.imread(mask_fn[j], cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread(rgb_fn[j])

        # cv2.imshow('img', rgb_img)
        # cv2.waitKey(0)
        # rgb_img = tsdf_utils.fix_distortion(rgb_img, tsdf.intrinsic[:3, :3], dist)
        # depth_img = tsdf_utils.fix_distortion(depth_img, tsdf.intrinsic[:3, :3], dist)
        # cv2.imshow('img', rgb_img)
        # cv2.waitKey(0)

        # print(depth_timestamps[i])
        # print(mask_timestamps[j])
        depth_img[mask_img == 0] = 0
        rgb_img[mask_img == 0] = 0
        # mask = np.zeros([480, 640])
        # mask[120:-120, 160:-160] = 1
        # depth_img = depth_img * mask
        _, mean_depth = tsdf_utils.filter_gaussian(depth_img)
        # mean_depth = 0.8
        # cv2.imshow('mask', mask_img)
        # cv2.imshow('depth', depth_img)
        # cv2.imshow('img', rgb_img)
        # cv2.waitKey(0)

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

        extrinsic = tsdf_utils.transform44(extrinsic)
        # write_file(depth_img, rgb_img, extrinsic, i)
        tsdf.parse_frame(depth_img, rgb_img, extrinsic, mean_depth)

    tsdf_utils.show_model(tsdf)

