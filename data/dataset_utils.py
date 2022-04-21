import cv2
import numpy as np

perms = ((0, 1, 2), (0, 2, 1),
          (1, 0, 2), (1, 2, 0),
          (2, 0, 1), (2, 1, 0))


def color_distorsion(im_c):
    im_correction = colorDistorsion(im_c)
    im = cv2.cvtColor(im_correction, cv2.COLOR_BGR2GRAY)
    return im.reshape(im.shape[0], im.shape[1], 1)

def to_black_and_white(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return im.reshape(im.shape[0], im.shape[1], 1)


def colorDistorsion(image, lower=0.5, upper=1.5, delta=18.0, delta_brigtness=36):

    image = image.astype(float)

    if np.random.randint(2):
        delta = np.random.uniform(-delta_brigtness, delta_brigtness)
        image += delta
        image = check_margins(image)

    contrast = np.random.randint(2)
    if contrast:
        alpha = np.random.uniform(lower, upper)
        image *= alpha
        image = check_margins(image)

    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image.astype(float)

    if np.random.randint(2):
        image[:, :, 1] *= np.random.uniform(lower, upper)
        image = check_margins(image, axis=1)
    if np.random.randint(2):
        image[:, :, 0] += np.random.uniform(-delta, delta)
        image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = image.astype(float)

    if contrast:
        alpha = np.random.uniform(lower, upper)
        image *= alpha
        image = check_margins(image)

    if np.random.randint(2):
        swap = perms[np.random.randint(len(perms))]
        image = swap_channels(image, swap)  # shuffle channels

    return image.astype(np.uint8)


def check_margins(img, axis=-1):
    if axis == -1:
        img[img > 255.0] = 255.0
        img[img < 0.0] = 0.0
    else:
        img[:, :, axis][img[:, :, axis] > 255.0] = 255.0
        img[:, :, axis][img[:, :, axis] < 0.0] = 0.0
    return img


def swap_channels(image, swaps):
    image = image[:, :, swaps]
    return image


def generate_composed_homography(max_angle=45, min_scaling=0.5, max_scaling=2.0, max_shearing=0.8):

    # random sample
    # scale = np.random.uniform(min_scaling, max_scaling)
    scale = np.random.uniform(1.0, 1.0)
    angle = np.random.uniform(-max_angle, max_angle)
    # shear = np.random.uniform(-max_shearing, max_shearing)
    shear = np.random.uniform(0, 0)    
   
    # scale transform
    scale_mat = np.eye(3)
    scale_mat[0, 0] = scale
    scale_mat[1, 1] = scale
    # rotation transform
    angle = np.deg2rad(angle)
    rotation_mat = np.eye(3)
    rotation_mat[0, 0] = np.cos(angle)
    rotation_mat[0, 1] = -np.sin(angle)
    rotation_mat[1, 0] = np.sin(angle)
    rotation_mat[1, 1] = np.cos(angle)
    # shear transform
    shear_mat = np.eye(3)
    shear_mat[0, 1] = shear

    # compose transforms
    # h = np.matmul(shear_mat, np.matmul(scale_mat, rotation_mat))
    h = np.matmul(scale_mat, rotation_mat)

    return h, scale, np.rad2deg(angle), shear



