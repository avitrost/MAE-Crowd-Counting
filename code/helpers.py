
import os
import glob

def get_image_paths():
    """
    This function returns lists containing the file path for each train
    and test image, as well as lists with the label of each train and
    test image. By default both lists will be 1500x1, where each
    entry is a char array (or string).
    """
    train_image_paths = glob.glob('data/eset/jhu_crowd_v2.0/train/images/*.jpg')
    test_image_paths = glob.glob('data/eset/jhu_crowd_v2.0/test/images/*.jpg')
    val_image_paths = glob.glob('data/eset/jhu_crowd_v2.0/val/images/*.jpg')
    train_density_paths = glob.glob('data/density/train/*.jpg')
    test_density_paths = glob.glob('data/density/test/*.jpg')    
    val_density_paths = glob.glob('data/density/val/*.jpg')
    return train_image_paths, test_image_paths, val_image_paths, train_density_paths, test_density_paths, val_density_paths