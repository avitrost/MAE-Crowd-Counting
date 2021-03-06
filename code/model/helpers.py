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
    train_density_paths = sorted(glob.glob('../../utils/data/density/train/*.png'), key=len)
    test_density_paths = sorted(glob.glob('../../utils/data/density/test/*.png'), key=len)
    val_density_paths = sorted(glob.glob('../../utils/data/density/val/*.png'), key=len)
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

def save_numpy(size):
    train_image_paths, test_image_paths, val_image_paths, train_density_paths, test_density_paths, val_density_paths = get_image_paths()
    # images = []
    # for i, path in enumerate(train_image_paths):  
    #     print(i)
    #     image = imread(path)[...,::-1]
    #     image = resize(image, (size, size, 3))
    #     images.append(image)
    # filename = 'train_images' + str(size) + 'x' + str(size) + '.npy'
    # np.save(filename, images)
    # images = []
    # for i, path in enumerate(test_image_paths):
    #     print(i)
    #     image = imread(path)[...,::-1]
    #     image = resize(image, (size, size, 3))
    #     images.append(image)
    # filename = 'test_images' + str(size) + 'x' + str(size) + '.npy'
    # np.save(filename, images)
    # images = []
    # for i, path in enumerate(val_image_paths):
    #     print(i)
    #     image = imread(path)[...,::-1]
    #     image = resize(image, (size, size, 3))
    #     images.append(image)
    # filename = 'val_images' + str(size) + 'x' + str(size) + '.npy'
    # np.save(filename, images)
    images = []
    for i, path in enumerate(train_density_paths):
        print(i)
        image = imread(path)[...,::-1]
        image = resize(image, (size, size, 3))
        images.append(image)
    filename = 'train_density' + str(size) + 'x' + str(size) + '.npy'
    np.save(filename, images)
    images = []
    for i, path in enumerate(test_density_paths):
        print(i)
        image = imread(path)[...,::-1]
        image = resize(image, (size, size, 3))
        images.append(image)
    filename = 'test_density' + str(size) + 'x' + str(size) + '.npy'
    np.save(filename, images)
    images = []
    for i, path in enumerate(val_density_paths):
        print(i)
        image = imread(path)[...,::-1]
        image = resize(image, (size, size, 3))
        images.append(image)
    filename = 'val_density' + str(size) + 'x' + str(size) + '.npy'
    np.save(filename, images)

def get_images_from_file(type):
    if type == 'train':
        return np.load('train_images48x48.npy')
    if type == 'test':
        return np.load('test_images48x48.npy')
    if type == 'val':
        return np.load('val_images48x48.npy')
    if type == 'train_density':
        return np.load('train_density48x48.npy')
    if type == 'test_density':
        return np.load('test_density48x48.npy')
    if type == 'val_density':
        return np.load('val_density48x48.npy')
# if __name__ == "__main__":
#     #train_image_paths, test_image_paths, val_image_paths, train_density_paths, test_density_paths, val_density_paths = get_image_paths()
#     #print(test_density_paths)
#     save_numpy(64)

def get_count_from_numpy():
    train_density_paths = sorted(glob.glob('../../utils/data/density_v2.0/train/*.npy'), key=len)
    test_density_paths = sorted(glob.glob('../../utils/data/density_v2.0/test/*.npy'), key=len)
    val_density_paths = sorted(glob.glob('../../utils/data/density_v2.0/val/*.npy'), key=len)

    for i, path in enumerate(train_density_paths):
        density_map = np.load(path)

        c = count_from_density_map(density_map)

        plt.title('count=%i' %c)

        plt.axis('off')
        plt.imshow(density_map, cmap='jet')
        plt.show()
        plt.clf()

def save_ground_truths():
    train_path = glob.glob('../../utils/data/eset/jhu_crowd_v2.0/train/image_labels.txt')
    test_path = glob.glob('../../utils/data/eset/jhu_crowd_v2.0/test/image_labels.txt')
    val_path = glob.glob('../../utils/data/eset/jhu_crowd_v2.0/val/image_labels.txt')
    train = open(train_path[0], 'r')
    lines = train.readlines()
    gts = []
    for line in lines:
        print(float(line.split(',')[1]))
        gts.append(float(line.split(',')[1]))
    print(np.max(gts))
    print(np.min(gts))
    print(np.mean(gts))
    print(np.var(gts))
    np.save('train_gt.npy', gts)
    test = open(test_path[0], 'r')
    lines = test.readlines()
    gts = []
    for line in lines:
        gts.append(float(line.split(',')[1]))
    np.save('test_gt.npy', gts)
    val = open(val_path[0], 'r')
    lines = val.readlines()
    gts = []
    for line in lines:
        gts.append(float(line.split(',')[1]))
    np.save('val_gt.npy', gts)

def get_ground_truths(type):
    if type == 'train':
        return np.load('train_gt.npy')
    if type == 'test':
        return np.load('test_gt.npy')
    if type == 'val':
        return np.load('val_gt.npy')

if __name__ == "__main__":
    #train_image_paths, test_image_paths, val_image_paths, train_density_paths, test_density_paths, val_density_paths = get_image_paths()
    #print(test_density_paths)
    #save_numpy(48)
    save_ground_truths()

