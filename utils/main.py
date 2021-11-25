
from helpers import get_image_paths
train_image_paths, test_image_paths, val_image_paths, train_density_paths, test_density_paths, val_density_paths = get_image_paths()

print(len(train_image_paths))
print(len(test_image_paths))
print(len(val_image_paths))