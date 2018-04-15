import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from PIL import Image

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    u, s, vh = np.linalg.svd(E)
    w = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    uwvt = np.matmul(np.matmul(u, w), vh)
    uwtvt = np.matmul(np.matmul(u, w.transpose()), vh)
    P = np.zeros([4, 3, 4])
    P[0, :, :3] = P[1, :, : 3] = uwvt
    P[2, :, :3] = P[3, :, : 3] = uwtvt
    P[0, :, 3] = P[2, :, 3] = u[:, -1]
    P[1, :, 3] = P[3, :, 3] = -u[:, -1]
    return P

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    M = image_points.shape[0]
    A = np.zeros([M*2, 4])
    for i in range(M):
        A[i * 2] = image_points[i, 0] * camera_matrices[i, 2, :] - camera_matrices[i, 0, :]
        A[i * 2 + 1] = image_points[i, 1] * camera_matrices[i, 2, :] - camera_matrices[i, 1, :]
    _, _, vh = np.linalg.svd(A)
    P = vh.transpose()[:, -1]
    if abs(P[-1]) < 1e-7:
        print('inf point')
        return P[:3]
    P /= P[-1]
    return P[:3]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    projected = np.matmul(camera_matrices.reshape(-1, 4), np.append(point_3d, 1))
    projected = projected.reshape(image_points.shape[0], -1)
    projected /= projected[:, -1:]
    return (projected[:, :2] - image_points).reshape(-1)

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    M = camera_matrices.shape[0]
    projected = np.matmul(camera_matrices.reshape(-1, 4), np.append(point_3d, 1)).reshape(M, -1)
    J = np.zeros([M*2, 3])
    J[::2, :] = -1 / (projected[:, 2:] * projected[:, 2:]) * camera_matrices[:, 2, :3] * projected[:, 0:1] + 1 / projected[:, 2:] * camera_matrices[:, 0, :3]
    J[1::2, :] = -1 / (projected[:, 2:] * projected[:, 2:]) * camera_matrices[:, 2, :3] * projected[:, 1:2] + 1 / projected[:, 2:] * camera_matrices[:, 1, :3]
    return J

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    P = linear_estimate_3d_point(image_points, camera_matrices)
    for _ in range(10):
        J = jacobian(P, camera_matrices)
        e = reprojection_error(P, image_points, camera_matrices)
        JtJ = np.matmul(J.transpose(), J)
        if np.linalg.matrix_rank(JtJ) == JtJ.shape[0]:
            P -= np.matmul(np.linalg.inv(JtJ), J.transpose()).dot(e)
        else:
            break
    return P

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    RTs = estimate_initial_RT(E)
    best_ratio = 0
    best_RT = None
    for i in range(RTs.shape[0]):
        RT = RTs[i]
        camera_matrice1 = np.zeros([3, 4])
        camera_matrice1[:, :3] = np.eye(3)
        camera_matrice1 = np.matmul(K, camera_matrice1)
        camera_matrice2 = np.matmul(K, RT)
        camera_matrices = np.concatenate([camera_matrice1.reshape(-1, 3, 4), camera_matrice2.reshape(-1, 3, 4)], axis=0)
        cnt = 0
        for n in range(image_points.shape[0]):
            P = nonlinear_estimate_3d_point(image_points[n], camera_matrices)
            # test for camera 1
            if P[-1] <= 0:
                continue
            # test for camera 2
            RT_inv = np.zeros([3, 4])
            RT_inv[:, :3] = RT[:, :3].transpose()
            RT_inv[:, -1] = -RT[:, -1]

            P_prime = np.matmul(RT_inv, np.append(P, 1))
            if P_prime[-1] <= 0:
                continue
            cnt += 1
        ratio = cnt / image_points.shape[0]
        if ratio > best_ratio:
            best_ratio = ratio
            best_RT = RT
    return best_RT


def match(img1, img2):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    matches = [(m, n) for i, (m, n) in enumerate(matches) if m.distance < 0.7*n.distance]

    # Need to draw only good matches, so create a mask
    matchesMask = [[1, 0] for i in range(len(matches))]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask=matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    # plt.imshow(img3,),plt.show()

    idx1 = [m.queryIdx for (m, _) in matches]
    idx2 = [m.trainIdx for (m, _) in matches]
    return np.array([kp1[idx].pt for idx in idx1]), np.array([kp2[idx].pt for idx in idx2])


def mloss(pt, fp, color1, color2, grad1, grad2, B, f, alpha=0, gamma=1, window_size=5):
    y, x = np.meshgrid(np.arange(pt[1] - window_size // 2, pt[1] + window_size // 2 + 1),
                             np.arange(pt[0] - window_size // 2, pt[0] + window_size // 2 + 1))
    y = y.reshape(-1)
    x = x.reshape(-1)
    valid = (y >= 0) & (y < color1.shape[0]) & (x >= 0) & (x < color1.shape[1])
    # fp = fp.reshape(-1)

    window_idxs = np.vstack([y[valid], x[valid]])
    # print(np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).shape)
    # print((color1[list(window_idxs)] - color1[pt[1], pt[0]]).shape)
    z = fp[0] * x + fp[1] * y + fp[2]
    d = B * f / z
    d = d[valid]
    window_idxs2 = window_idxs.copy()
    window_idxs2[1, :] -= d.reshape(-1).astype(np.int)
    valid = (window_idxs2[1, :] >= 0) & (window_idxs2[1, :] < color1.shape[1])
    valid = np.vstack([valid, valid])
    window_idxs = window_idxs[valid].reshape(2, -1)
    window_idxs2 = window_idxs2[valid].reshape(2, -1)
    if window_idxs2.size == 0:
        return np.inf

    weights = np.exp(-np.linalg.norm(color1[list(window_idxs)] - color1[pt[1], pt[0]], ord=1, axis=1) / gamma)

    rou = (1 - alpha) * np.linalg.norm(color1[list(window_idxs)] - color2[list(window_idxs2)], ord=1, axis=1) + \
        alpha * np.abs(grad1[list(window_idxs)] - grad2[list(window_idxs2)])
    loss = np.dot(weights, rou) / weights.shape[0] + (np.size(valid) - np.count_nonzero(valid)) * 1000
    # print((color1[list(window_idxs)] - color1[pt[1], pt[0]]).shape)
    # print(np.linalg.norm(color1[list(window_idxs)] - color1[pt[1], pt[0]], ord=1, axis=1).shape)
    # loss = np.dot(np.ones(window_idxs.shape[1]), np.square(grad1[list(window_idxs)] - grad2[list(window_idxs2)])) / window_idxs.shape[1]
    return loss


def PatchMatch(img1, img2, B, f, dmin, dmax):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    lap1 = cv2.Laplacian(gray1, cv2.CV_32F)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    lap2 = cv2.Laplacian(gray2, cv2.CV_32F)
    # mloss([3, 3], 1, 1, 1, img1, img1, gray1, gray1, lap1, lap1, 1, 520)
    xv, yv = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
    z0 = dmin + np.random.rand(img1.shape[0], img1.shape[1]) * (dmax - dmin)
    rand1 = np.random.rand(img1.shape[0], img1.shape[1])
    rand2 = np.random.rand(img1.shape[0], img1.shape[1])
    nx = np.cos(2 * math.pi * rand2) * np.sqrt(1 - rand1 * rand1)
    ny = np.sin(2 * math.pi * rand2) * np.sqrt(1 - rand1 * rand1)
    nz = rand1
    a = -nx / nz
    b = -ny / nz
    c = (nx * xv + ny * yv) / nz + z0
    fp = np.stack([a, b, c], -1)
    # cv2.imshow('depth', (fp[:, :, 0] * x + fp[:, :, 1] * y + fp[:, :, 2]) / 30)
    # cv2.waitKey(0)
    loss = np.zeros([img1.shape[0], img1.shape[1]], dtype=np.float32)
    for x in range(img1.shape[1]):
        for y in range(img1.shape[0]):
            loss[y, x] = mloss((x, y), fp[y, x], img1, img2, lap1, lap2, B, f)
    for it in range(5):
        # spatial propogation
        # depth = np.zeros([img1.shape[0], img1.shape[1]], dtype=np.float32)

        for x in range(img1.shape[1]):
            for y in range(img1.shape[0]):
                if y > 0 and x > 0:
                    # loss_max = 10000
                    # best_d = 0
                    # for k in range(2, 20):
                    #     fp[y, x, -1] = - fp[y, x, 0] * x - fp[y, x, 1] * y + k
                    #     loss = mloss((x, y), fp[y, x], img1, img2, lap1, lap2, B, f)
                    #     if loss < loss_max:
                    #         loss_max = loss
                    #         best_d = k
                    # fp[y, x, -1] = - fp[y, x, 0] * x - fp[y, x, 1] * y + best_d
                    # print(best_d)
                    # depth[y, x] = best_d
                    loss1 = mloss((x, y), fp[y - 1, x], img1, img2, lap1, lap2, B, f)
                    if loss[y, x] > loss1:
                        fp[y, x] = fp[y - 1, x]
                        loss[y, x] = loss1
                    loss2 = mloss((x, y), fp[y, x - 1], img1, img2, lap1, lap2, B, f)
                    if loss[y, x] > loss2:
                        fp[y, x] = fp[y, x - 1]
                        loss[y, x] = loss2
        for x in reversed(range(img1.shape[1])):
            for y in reversed(range(img1.shape[0])):
                if y < img1.shape[0] - 1 and x < img1.shape[1] - 1:
                    loss1 = mloss((x, y), fp[y + 1, x], img1, img2, lap1, lap2, B, f)
                    if loss[y, x] > loss1:
                        fp[y, x] = fp[y + 1, x]
                        loss[y, x] = loss1
                    loss2 = mloss((x, y), fp[y, x + 1], img1, img2, lap1, lap2, B, f)
                    if loss[y, x] > loss2:
                        fp[y, x] = fp[y, x + 1]
                        loss[y, x] = loss2
        # for x in reversed(range(img1.shape[1])):
        #     for y in reversed(range(img1.shape[0])):
        #         if y < img1.shape[0] - 1 and x < img1.shape[1] - 1:
        #             # loss_max = 10000
        #             # best_d = 0
        #             # for k in range(2, 20):
        #             #     fp[y, x, -1] = - fp[y, x, 0] * x - fp[y, x, 1] * y + k
        #             #     loss = mloss((x, y), fp[y, x], img1, img2, lap1, lap2, B, f)
        #             #     if loss < loss_max:
        #             #         loss_max = loss
        #             #         best_d = k
        #             # fp[y, x, -1] = - fp[y, x, 0] * x - fp[y, x, 1] * y + best_d
        #             # print(best_d)
        #             # depth[y, x] = best_d
        #             loss = mloss((x, y), fp[y, x], img1, img2, lap1, lap2, B, f)
        #             if loss > \
        #                     mloss((x, y + 1), fp[y + 1, x], img1, img2, lap1, lap2, B, f):
        #                 fp[y, x] = fp[y + 1, x]
        #             if loss > \
        #                     mloss((x + 1, y), fp[y, x + 1], img1, img2, lap1, lap2, B, f):
        #                 fp[y, x] = fp[y, x + 1]
        # cv2.imshow('depth', depth / 20)
        cv2.imshow('depth%d' % it, ((fp[:, :, 0] * xv + fp[:, :, 1] * yv + fp[:, :, 2]).astype(np.float32)) / 30)
        cv2.waitKey(30)
        # random improvement
        for x in range(img1.shape[1]):
            for y in range(img1.shape[0]):
                dz = (dmax - dmin) / 2
                dn = 1
                while dz > 0.1:
                    dz_rand = (np.random.rand() * 2 - 1) * dz
                    z_bk = fp[y, x, -1]
                    fp[y, x, -1] += dz_rand
                    potential = mloss((x, y), fp[y, x], img1, img2, lap1, lap2, B, f)
                    if loss[y, x] < potential:
                        fp[y, x, -1] = z_bk
                    else:
                        loss[y, x] = potential
                    dz /= 2
                # while dn > 0.1:
                #     dn_rand = np.random.rand(3) * dn
                #     n_rand = fp[y, x] + dn_rand
                #     n_rand = n_rand / np.linalg.norm(n_rand)
                #     fp_bk = fp[y, x].copy()
                #     fp[y, x] = np.array([-n_rand[0] / n_rand[2], -n_rand[1] / n_rand[2], (n_rand[0] * x + n_rand[1] * y) / n_rand[2] + fp[y, x, -1]])
                #     potential = mloss((x, y), fp[y, x], img1, img2, lap1, lap2, B, f)
                #     if loss[y, x] < potential:
                #         fp[y, x] = fp_bk
                #     else:
                #         loss[y, x] = potential
                #     dn /= 2
        cv2.imshow('improve%d' % it, ((fp[:, :, 0] * xv + fp[:, :, 1] * yv + fp[:, :, 2]).astype(np.float32)) / 30)
        cv2.waitKey(30)


if __name__ == '__main__':
    # Image.open('test_images/left.png').load()
    img1 = cv2.imread('test_images/left.png')
    img1 = img1[::2, ::2, :]
    img2 = cv2.imread('test_images/right.png')
    img2 = img2[::2, ::2, :]
    PatchMatch(img1, img2, 4, 30, 4, 30)