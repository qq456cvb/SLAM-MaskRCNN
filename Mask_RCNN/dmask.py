import numpy as np

def depth_filter(depth_image, masks, dep_range=6000):
    """Utilize depth to filter mask region, if depth smaller or larger than the median depth
        of current mask with a certain range, judge as non-mask
    :param depth_image: depth image of current frame
    :param target_mask: the mask need to be filtered
    :param target_roi: the minimal bounding box range of the mask
    :param dep_range: the depth which in (median val - dep_range) ~ (median val + dep_range)
    :return:
    """
    num_mask = masks.shape[2]
    new_masks = masks.copy()
    for i in range(num_mask):
        median = np.median(depth_image[masks[:, :, i]])
        std = np.std(depth_image[masks[:, :, i]])
        dep_filter = (depth_image < median - 5*std) | (depth_image > median + 5*std)
        new_masks[:, :, i][dep_filter] = False
    return new_masks

def preserve_small_objs(masks):
    """ Return non-overlapping masks from existing masks
    :param masks: existing masks
    :return: non-overalpping masks
    """
    areas = np.array([np.count_nonzero(masks[:, :, i]) for i in range(masks.shape[-1])])
    sorted_idx = np.argsort(areas)
    for i in range(len(sorted_idx)):
        for j in range(i + 1, len(sorted_idx)):
            if np.any(masks[:, :, sorted_idx[i]] & masks[:, :, sorted_idx[j]]):
                masks[:, :, sorted_idx[j]][masks[:, :, sorted_idx[i]] & masks[:, :, sorted_idx[j]]] = False
    return masks

def filter_tiny_objects(masks):
    """ Get rid of detection noise of objects
    :param masks: existing masks
    :return: clean masks    
    """
    areas = np.array([np.count_nonzero(masks[:, :, i]) for i in range(masks.shape[-1])])
    save_idx = []
    for idx, area in enumerate(areas):
        if area > 2000:
            save_idx.append(idx)
    #print("Before filter: {} After filter: {}".format(len(areas), len(save_idx)))
    return masks[:, :, save_idx]

def mask_detect(model, rgb_image, depth_image=None, noise_remove=True):
    result = model.detect([rgb_image], verbose=0)[0]
    masks = result['masks']
    masks = masks.astype(np.bool)
    if depth_image is not None:
        masks = depth_filter(depth_image, masks)
    if noise_remove:
        masks = filter_tiny_objects(masks)
    masks = preserve_small_objs(masks)
    cls = np.zeros([rgb_image.shape[0], rgb_image.shape[1]], np.uint8)
    for i in range(masks.shape[2]):
        cls[masks[:, :, i]] = i + 1
    return cls