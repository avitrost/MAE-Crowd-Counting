
import os
import glob
import numpy as np
from cv2 import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from mae import *

RESIZE_SHAPE = IMAGE_SIZE

def get_image_paths():
    """
    This function returns lists containing the file path for each train
    and test image, as well as lists with the label of each train and
    test image. By default both lists will be 1500x1, where each
    entry is a char array (or string).
    """
    train_image_paths = glob.glob('../../utils/data/eset/jhu_crowd_v2.0/train/images/*.jpg')
    test_image_paths = glob.glob('../../utils/data/eset/jhu_crowd_v2.0/test/images/*.jpg')
    val_image_paths = glob.glob('../../utils/data/eset/jhu_crowd_v2.0/val/images/*.jpg')
    train_density_paths = glob.glob('../../utils/data/density/train/*.png')
    test_density_paths = glob.glob('../../utils/data/density/test/*.png')    
    val_density_paths = glob.glob('../../utils/data/density/val/*.png')
    return train_image_paths, test_image_paths, val_image_paths, train_density_paths, test_density_paths, val_density_paths

def get_images_from_paths(paths):
    images = []
    for path in paths:
        image = imread(path)[...,::-1]
        image = resize(image, (RESIZE_SHAPE, RESIZE_SHAPE, 3))
        images.append(image)
    return images

def count_from_density_map(density_map: np.ndarray) -> int:
    return round(np.sum(np.clip(density_map, a_min=0, a_max=None)))