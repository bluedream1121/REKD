import os, cv2, math, json, torch
import numpy as np

from collections import Counter
import torch.nn.functional as F




##################### ================= For evaluation ========================= ###############

def warpPerspectivePoint(src_point, H):
    # normalize h
    H /= H[2][2]
    src_point = np.append(src_point, 1)

    dst_point = np.dot(H, src_point)
    dst_point /= dst_point[2]

    return dst_point[0:2]

def warpPerspectivePoints(src_points, H):
    # normalize H
    H /= H[2][2]

    ones = np.ones((src_points.shape[0], 1))
    points = np.append(src_points, ones, axis = 1)

    warpPoints = np.dot(H, points.T)
    warpPoints = warpPoints.T / warpPoints.T[:, 2][:,None]

    return warpPoints[:,0:2]

def GetGroundTruth(im1_path, im2_path):
    path = im1_path[:im1_path.rindex(os.sep)]
    im1_name = im1_path[im1_path.rindex(os.sep)+1:].split(".")[0]
    im2_name = im2_path[im2_path.rindex(os.sep)+1:].split(".")[0]
    H_path = path + os.sep + 'H_'+ im1_name + '_' + im2_name

    # load homography matrix
    H = np.fromfile(H_path, sep=" ")
    H.resize((3, 3))

    return H


def compute_repeatability(k1, k2, H, pixel_threshold=3):
    warp_points = warpPerspectivePoints(k1, H)
    k1_to_k2_pts = warp_points

    is_repeatable = []
    for (x1, y1) in k1_to_k2_pts:
        repeatable_check = False
        for (x2, y2) in k2:## warp kpts1 to image2 (using GT)
            distance = math.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
            if distance < pixel_threshold:
                is_repeatable.append(True)
                repeatable_check = True
                break
        if repeatable_check == False:
            is_repeatable.append(False)
    
    assert len(is_repeatable) == len(k1)

    return is_repeatable

def compute_repeatability_fast(k1, k2, H, pixel_threshold=3):

    warp_points = warpPerspectivePoints(k1, H) ## warp kpts1 to image2 (using GT)
    k1_to_k2_pts = warp_points

    repeatable_matrix = []
    k1_to_k2_pts = torch.tensor(k1_to_k2_pts).cuda()
    k2 = torch.tensor(k2).cuda()
    for kpt in k1_to_k2_pts:
        repeatable_check = False

        repeatable_matrix.append(torch.sqrt(torch.pow(kpt - k2, 2).sum(1)))
    repeatable_matrix = torch.stack(repeatable_matrix)

    is_repeatable = torch.relu(-repeatable_matrix + pixel_threshold).sum(1).bool()  ## check if any number in pixel threshold for each point

    assert len(is_repeatable) == len(k1)

    return is_repeatable.cpu().numpy().tolist()



##################### ================= For extraction ========================= ###############



## get hpatches image list and target scenes
def load_hpatches_images(dataset_dir, split):
    list_images_txt='datasets/HPatches_images.txt'
    split_path = 'datasets/splits.json'
    splits = json.load(open(split_path))
    target_scenes= splits[split]['test']

    ## open images of target images (debug, view, illum, all..)
    f = open(list_images_txt, "r")
    image_list = []
    for path_to_image in sorted(f.readlines()):
        scene_name = path_to_image.split('/')[-2]
        if scene_name not in target_scenes:
            continue
        image_list.append(os.path.join(dataset_dir, path_to_image).rstrip('\n'))
    
    target_scenes = [os.path.join(dataset_dir, 'hpatches-sequences-release', target_scene) for target_scene in target_scenes]

    return image_list, target_scenes


def make_save_dir(exp_name):
    results_dir='extracted_features/'
    save_feat_dir = os.path.join(results_dir, exp_name)

    create_result_dir(save_feat_dir)

    return save_feat_dir


def check_directory(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)


def create_result_dir(path):
    directories = path.split('/')
    tmp = ''
    for idx, dir in enumerate(directories):
        tmp += (dir + '/')
        if idx == len(directories)-1:
            continue
        check_directory(tmp)


def upsample_pyramid(image, upsampled_levels, scale_factor_levels):
    ## image np.array([C, H, W]), upsampled_levels int
    up_pyramid = []
    for j in range(upsampled_levels):
        factor = scale_factor_levels ** (upsampled_levels - j)
        up_image =  cv2.resize(image.transpose(1,2,0), dsize=(0,0), \
                        fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)     
        up_pyramid.append(up_image[np.newaxis])


    return up_pyramid

def check_coorinate_in_image(x, y, w, h):
    if x < 0: x=0
    elif x > w-1 : x=w-1

    if y <0 : y=0
    elif y > h-1 : y=h-1

    return x,y

def check_coorinates_in_image(im_pts, W, H):
    X_coords = im_pts[:, 0]
    Y_coords = im_pts[:, 1]
    scale_coords = im_pts[:, 2]
    score_coords = im_pts[:, 3]

    X_coords = np.where(X_coords < W-1, X_coords, W-1)
    X_coords = np.where(X_coords > 0, X_coords, 0)

    Y_coords = np.where(Y_coords < H-1, Y_coords, H-1)
    Y_coords = np.where(Y_coords > 0, Y_coords, 0)

    im_pts = np.stack([X_coords, Y_coords, scale_coords, score_coords]).T
    
    return im_pts

def assign_orientations_to_keypoints(im_pts, ori_values):
    ori_vectors = []
    for kpt in im_pts:
        x, y = round(kpt[0]-1), round(kpt[1]-1)
        ori_value = ori_values[0, y, x].cpu().numpy()
        ori_vectors.append(ori_value)        
    ori_vectors = np.array(ori_vectors)[: , np.newaxis]

    im_pts = np.concatenate([im_pts, ori_vectors], axis=1)
    return im_pts
