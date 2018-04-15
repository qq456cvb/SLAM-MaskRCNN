import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "samples/coco/"))

import coco
# import utils
# import model as modellib
# import visualize
import mrcnn.utils
import mrcnn.model as modellib
from mrcnn import visualize

from os import listdir
from os.path import isfile, join
from skimage.feature import match_template



# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory to save raw images
image_dir = '/home/yzn/CV/Project_3d/rgbd_dataset_freiburg2_coke'
#image_dir = '/home/yzn/CV/Mask_RCNN/coke_test'


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


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

# Function for finding minimum rectangle bouding region
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array([rmin, cmin, rmax, cmax])

# detect objects in image and retrurn mask info
def mask_detect(model, rgb_image):
    results = model.detect([rgb_image], verbose=0)
    r = results[0]
    return r['rois'], r['masks'], r['class_ids'], r['scores']

# Function to pick up specific mask  from all categroies
def pick_mask(all_mask, class_ids, class_names, chosen_class_name):
    target_mask = None
    target_index = None
    if class_ids is None:
        pass
    else:
        for i in range(class_ids.shape[0]):
            if class_names[class_ids[i]] == chosen_class_name:
                target_mask = all_mask[:,:,i]
                target_index = i
                break
    return target_mask, target_index

# Utilize depth to filter mask, if depth smaller or larger than the median depth 
# value of current mask with a hand-chosen range, judge as non-mask
def depth_filter(depth_image, rgb_image, target_mask, target_roi, range=3000):
    rmin, cmin, rmax, cmax = target_roi
    new_mask = target_mask.copy()
    dep_target = depth_image[rmin:rmax, cmin:cmax]
    std_val = np.median(dep_target)
    dep_filter = (depth_image < std_val-range) | (depth_image > std_val+range)
    new_mask[dep_filter] = 0
    new_roi = bbox2(new_mask)
    return new_mask, new_roi


# This function template match current image with previous target mask, and execute Mask R-CNN
# to little expanded matched region, and return the mask of whole new image based on the result.
def template_match_mask_detect(model, rgb_image, pre_target, expand_ratio=0.25):
    row = rgb_image.shape[0]
    col = rgb_image.shape[1]
    matches = skimage.feature.match_template(rgb_image, pre_target)
    ij = np.unravel_index(np.argmax(matches), matches.shape)
    match_rmin, match_cmin = ij[:-1]
    h_target, w_target,_ = pre_target.shape
    crop_rmin = int(max(0,match_rmin - h_target * expand_ratio))
    crop_cmin = int(max(0,match_cmin - h_target * expand_ratio))
    crop_rmax = int(min(rgb_image.shape[0], match_rmin + h_target * (1 + expand_ratio)))
    crop_cmax = int(min(rgb_image.shape[1], match_cmin + w_target * (1 + expand_ratio)))

    #print(crop_rmin, crop_cmin, crop_rmax, crop_cmax)

    expand_target = rgb_image[crop_rmin:crop_rmax, crop_cmin:crop_cmax, :]

    # target_image = np.zeros(rgb_image.shape, dtype=np.uint8)
    # target_image[crop_rmin:crop_rmax, crop_cmin:crop_cmax, :] = expand_target
    # the rois and masks here only partial, need to expand to full iamge size
    rois, masks, class_ids, scores = mask_detect(model, expand_target)
    num_objects = class_ids.shape[0]
    if num_objects == 0:
        return None, None, None, None
    full_rois = rois + np.array([crop_rmin, crop_cmin, crop_rmin, crop_cmin])
    full_masks = np.zeros((row, col, num_objects), dtype=np.uint8)
    full_masks[crop_rmin:crop_rmax, crop_cmin:crop_cmax, :] = masks


    return full_rois, full_masks, class_ids, scores
    #return rois, masks, class_ids, scores

def save_mask_images(rgb_image, dir, rgb_file_name, class_name="none", target_mask=None):
    if class_name == "none" or class_name == "none-none":
        row = rgb_image.shape[0]
        col = rgb_image.shape[1]
        mask_image = gray_image = np.zeros((row, col, 3), dtype=np.uint8)
    else:
        rgb_mask = np.dstack([target_mask,target_mask,target_mask])
        mask_image = np.multiply(rgb_image, rgb_mask)
        gray_image = rgb_mask * 255
        gray_image = gray_image.astype(dtype=np.uint8)

    file_name = os.path.splitext(rgb_file_name)[0] + '_' +  class_name + os.path.splitext(rgb_file_name)[1]
    plt.imsave(join(join(dir,'rgb_mask'),file_name), mask_image)
    plt.imsave(join(join(dir,'gray_mask'),file_name), gray_image)   


def calc_overlap_ratio(direct_roi, match_roi):
    XA1, YA1, XA2, YA2 = direct_roi
    XB1, YB1, XB2, YB2 = match_roi
    SI= max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    SA = (XA2 - XA1) * (YA2 - YA1)
    SB = (XB2 - XB1) * (YB2 - YB1)
    ratio = SI / (SA + SB - SI)
    return ratio

def union_mask_roi(direct_roi, direct_mask, match_roi, match_mask):
    XA1, YA1, XA2, YA2 = direct_roi
    XB1, YB1, XB2, YB2 = match_roi
    X1 = min(XA1, XB1)
    Y1 = min(YA1, YB1)
    X2 = max(XA2, XB2)  
    Y2 = max(YA2, YB2)
    target_roi = np.array([X1, Y1, X2, Y2])
    target_mask = match_mask
    direct_filter = direct_mask > 0
    target_mask[direct_filter] = 1
    return target_mask, target_roi

def detect_class_record(detect_record, rgb_name, class_ids, class_names, prefix='direct'):
    log = rgb_name + " " + prefix + " detect: "
    if class_ids is None:
        pass
    else:
        for i in class_ids:
            log = log + class_names[i] + "  "
    detect_record.append(log)   


rgb_dir = join(image_dir,'rgb')
depth_dir = join(image_dir,'depth')
mask_dir = join(image_dir,'mask')
rgb_files = sorted(listdir(rgb_dir))
depth_files = sorted(listdir(depth_dir))

index = 0
pre_target = None
match_times = 0
oldest_match_limit = 5
pre_flag = False
detect_record = []

for file in rgb_files:

    curt_rgb = skimage.io.imread(join(rgb_dir,file))
    curt_depth = skimage.io.imread(join(depth_dir,depth_files[index]))
    index += 1

    if index%25 == 0:
      print('{}%'.format(index/25))

    target_mask = None
    target_roi = None
    final_class = None
    curt_rois, curt_masks, curt_classIds, curt_scores = mask_detect(model, curt_rgb)
    
    # direct detect record
    detect_class_record(detect_record, file, curt_classIds, class_names, 'direct')

    #num_class = curt_classIds.shape[0]
    possible_candidates = ['bottle', 'cup', 'vase']
    possible_match_candidates = ['bottle', 'cup', 'vase']

    # target object not occur in previous frame
    if pre_flag == False:
        for candidate_class in possible_candidates:
            target_mask, target_index = pick_mask(curt_masks, curt_classIds, class_names, candidate_class)
            if target_index != None:
                final_class = candidate_class
                break
        # If still could not detect object, pass
        if target_index is None:
            pre_target = None
            pre_flag = False
            save_mask_images(curt_rgb, mask_dir, file)
            continue
        target_mask, target_roi = depth_filter(curt_depth, curt_rgb, target_mask, curt_rois[target_index])
        save_mask_images(curt_rgb, mask_dir, file, final_class, target_mask)
        rmin, cmin, rmax, cmax = target_roi
        # template match target for next frame
        pre_target = curt_rgb[rmin:rmax, cmin:cmax, :]
        pre_flag = True
    else:
        direct_class = 'none'
        match_class = 'none'
        #print(file)
        match_rois, match_masks, match_classIds, match_scores = template_match_mask_detect(model, curt_rgb, pre_target)

        # template match detect record
        detect_class_record(detect_record, file, match_classIds, class_names, 'match')

        for candidate_class in possible_candidates:
            direct_mask, direct_index = pick_mask(curt_masks, curt_classIds, class_names, candidate_class)
            if direct_index != None:
                direct_class = candidate_class
                break            
        for candidate_class in possible_match_candidates:
            match_mask, match_index = pick_mask(match_masks, match_classIds, class_names, candidate_class)
            if match_index != None:
                match_class = candidate_class
                break
        if match_index is not None and direct_index is not None:
            direct_roi = curt_rois[direct_index]
            match_roi = match_rois[match_index]
            if calc_overlap_ratio(direct_roi, match_roi) < 0.2:
                target_mask = match_mask
                target_roi = match_roi
            else:
                target_mask, target_roi = union_mask_roi(direct_roi, direct_mask, match_roi, match_mask)
        elif match_index is not None:
            match_roi = match_rois[match_index]
            target_mask = match_mask
            target_roi = match_roi
        elif direct_index is not None:
            direct_roi = curt_rois[direct_index]
            target_mask = direct_mask
            target_roi = direct_roi
        else:
            save_mask_images(curt_rgb, mask_dir, file, class_name="none-none")
            if match_times > oldest_match_limit:
                pre_target = None
                pre_flag = False
                match_times = 0
            else: 
                match_times += 1
            continue
        
        match_times = 0
        final_class = direct_class + '-' + match_class
        target_mask, target_roi = depth_filter(curt_depth, curt_rgb, target_mask, target_roi)
        save_mask_images(curt_rgb, mask_dir, file, final_class, target_mask)
        rmin, cmin, rmax, cmax = target_roi
        # template match target for next frame
        pre_target = curt_rgb[rmin:rmax, cmin:cmax, :]
        pre_flag = True                                




detectlogfile = open('test_log/coke_log.txt', 'w')
for item in detect_record:
  detectlogfile.write("%s\n" % item)

    



  # results = model.detect([rgb_image], verbose=0)
  # index+=1
  # if index%25 == 0:
  #   print('{}%'.format(index/25))
  # r = results[0]
  # num_class = len(r['class_ids'])
  # for i in range(num_class):
  #   if class_names[r['class_ids'][i]] == 'bottle':
  #     t = r['masks'][:,:,i]
  #     s = np.dstack([t,t,t])
  #     mask_image = np.multiply(rgb_image, s)
  #     gray_image = s * 255
  #     plt.imsave(join(join(image_dir,'rgb_mask'),file), mask_image)
  #     plt.imsave(join(join(image_dir,'gray_mask'),file), gray_image)






