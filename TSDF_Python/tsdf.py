import numpy as np
import cv2


class TSDF:
    def __init__(self, intrinsics):
        self.intrinsic = np.eye(4)
        self.intrinsic[[[0, 1, 0, 1], [0, 1, 2, 2]]] = np.array(intrinsics)
        self.init = False
        self.tsdf_diff = None
        self.tsdf_wt = None
        self.tsdf_color = None
        self.mu = 0
        self.vol_dim = 64
        self.tex_dim = int(np.sqrt(pow(self.vol_dim, 3)))
        self.voxel = 0
        self.vol_start = None
        self.vol_end = None
        self.intrinsic_inv = np.linalg.inv(self.intrinsic)
        self.init_extrinsic_inv = None

    def parse_frame(self, depth, color, extrinsic, mean_depth):
        if not self.init:
            self.init = True
            self.init_extrinsic_inv = np.linalg.inv(extrinsic)
            rect = cv2.boundingRect(cv2.findNonZero(depth.astype(np.uint8)))
            tl = np.dot(self.intrinsic_inv[:3, :3], [rect[0], depth.shape[0] - 1 - rect[1], 1])
            br = np.dot(self.intrinsic_inv[:3, :3], [rect[0] + rect[2], depth.shape[0] - 1 - rect[1] - rect[3], 1])
            tl *= mean_depth / 5000
            br *= mean_depth / 5000

            half_side = np.sqrt(np.dot(tl[:2]-br[:2], tl[:2]-br[:2])) / 2
            center = (tl + br) / 2
            self.vol_start = center - half_side
            self.vol_end = center + half_side
            self.voxel = (self.vol_end - self.vol_start) / self.vol_dim
            self.mu = 2 * self.voxel[0]
            self.tsdf_diff = np.ones([self.tex_dim, self.tex_dim], np.float32) * self.mu
            self.tsdf_wt = np.zeros([self.tex_dim, self.tex_dim], np.int32)
            self.tsdf_color = np.zeros([self.tex_dim, self.tex_dim, 3], np.int32)
            self.parse_frame(depth, color, extrinsic, mean_depth)
        else:
            j, i = np.meshgrid(np.arange(self.tex_dim), np.arange(self.tex_dim))
            flattened_idx = i *self.tex_dim + j
            x_idx = flattened_idx // (self.vol_dim * self.vol_dim)
            y_idx = flattened_idx // self.vol_dim - x_idx * self.vol_dim
            z_idx = flattened_idx % self.vol_dim
            pos_inhomo = self.vol_start + np.dstack([x_idx, y_idx, z_idx]) * self.voxel
            pos_homo = np.concatenate([pos_inhomo, np.ones([self.tex_dim, self.tex_dim, 1])], axis=-1)
            proj = np.dot(np.matmul(extrinsic, self.init_extrinsic_inv), pos_homo.reshape([-1, 4]).transpose())
            pixel = np.dot(self.intrinsic, proj)
            pixel /= pixel[2, :]
            pixel = pixel.transpose().reshape([self.tex_dim, self.tex_dim, -1])
            x = pixel[:, :, 0].astype(np.int)
            y = pixel[:, :, 1].astype(np.int)
            # print(x.shape)
            x = x.reshape(-1)
            y = y.reshape(-1)
            mask = (x >= 0) & (x <= color.shape[1] - 1) & (y >= 0) & (y <= color.shape[0] - 1)

            diff = depth[[color.shape[0] - 1 - y.reshape(-1), x.reshape(-1)]] / 5000 - proj[2, :]
            diff[depth[[color.shape[0] - 1 - y.reshape(-1), x.reshape(-1)]] == 0] = self.mu

            diff = np.maximum(np.minimum(diff, self.mu), -self.mu)
            mask &= abs(diff) < self.mu
            weight = 1
            self.tsdf_wt = self.tsdf_wt.reshape(-1)
            self.tsdf_color = self.tsdf_color.reshape(-1, 3)
            self.tsdf_diff = self.tsdf_diff.reshape(-1)
            weight_mask = self.tsdf_wt > 0
            self.tsdf_diff[mask & weight_mask] = (self.tsdf_diff[mask & weight_mask] * self.tsdf_wt[mask & weight_mask] + weight * diff[mask & weight_mask]) / (self.tsdf_wt[mask & weight_mask] + weight)
            self.tsdf_color[mask & weight_mask] = (self.tsdf_color[mask & weight_mask] * np.expand_dims(self.tsdf_wt[mask & weight_mask], -1) + weight * color[[color.shape[0] - 1 - y.reshape(-1), x.reshape(-1)]][mask & weight_mask])\
                                                  / np.expand_dims(self.tsdf_wt[mask & weight_mask] + weight, -1)

            self.tsdf_diff[mask & ~weight_mask] = weight * diff[mask & ~weight_mask]
            self.tsdf_color[mask & ~weight_mask] = weight * color[[color.shape[0] - 1 - y.reshape(-1), x.reshape(-1)]][mask & ~weight_mask]
            self.tsdf_wt = self.tsdf_wt.reshape(self.tex_dim, -1)
            self.tsdf_color = self.tsdf_color.reshape(self.tex_dim, self.tex_dim, -1)
            self.tsdf_diff = self.tsdf_diff.reshape(self.tex_dim, -1)


            # for i in range(self.tex_dim):
            #     for j in range(self.tex_dim):
            #         flattened_idx = i * self.tex_dim + j
            #         x_idx = flattened_idx // (self.vol_dim * self.vol_dim)
            #         y_idx = flattened_idx // self.vol_dim - x_idx * self.vol_dim
            #         z_idx = flattened_idx % self.vol_dim
            #
            #         pos_inhomo = self.vol_start + np.array([x_idx, y_idx, z_idx]) * self.voxel
            #         proj = np.dot(np.matmul(self.init_extrinsic_inv, extrinsic), np.array([*pos_inhomo, 1]))
            #         pixel = np.dot(self.intrinsic, proj)
            #         pixel /= pixel[2]
            #         x, y = pixel[:2]
            #         if x < 0 or x > color.shape[1] - 1 or y < 0 or y > color.shape[0] - 1:
            #             continue
            #
            #         # TODO: add interpolation
            #         x = int(x)
            #         y = int(y)
            #         diff = depth[color.shape[0] - y - 1, x] / 5000 - proj[2]
            #         if depth[color.shape[0] - y - 1, x] == 0:
            #             diff = self.mu
            #
            #         diff = max(min(diff, self.mu), -self.mu)
            #         if abs(diff) < self.mu:
            #             weight = 1
            #             if self.tsdf_wt[i, j] != 0:
            #                 self.tsdf_diff[i, j] = self.tsdf_diff[i, j] * self.tsdf_wt[i, j] + weight * diff
            #                 self.tsdf_diff[i, j] /= (self.tsdf_wt[i, j] + weight)
            #
            #                 self.tsdf_color[i, j, :] = self.tsdf_color[i, j, :] * self.tsdf_wt[i, j] + weight * color[color.shape[0] - y - 1, x, :]
            #                 self.tsdf_color[i, j, :] //= (self.tsdf_wt[i, j] + weight)
            #             else:
            #                 self.tsdf_diff[i, j] = weight * diff
            #
            #                 self.tsdf_color[i, j, :] = weight * color[color.shape[0] - y - 1, x, :]
            #             self.tsdf_wt[i, j] += weight

            # cv2.imshow("tsdf", self.tsdf_wt.astype(np.uint8) * 200)
            # cv2.waitKey(30)


if __name__ == '__main__':
    pass