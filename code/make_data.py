import random
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from PIL import Image

DATASET_ROOT = Path("./jhu_crowd_v2.0") # Should point to a directory containing extracted JHU-CROWD++ data set
CACHE_ROOT = Path("./cache") # Directory to cache generated density maps
MODEL_SAVE_PATH = "./model.pt"
CHECKPOINTS_DIR = "./checkpoints"
SEED = 42

def load_dataset_paths(dataset_root: Path, img_extensions: set) -> List[Tuple[Path, Path]]:
    images_path = dataset_root / "images"
    gt_path = dataset_root / "gt"

    img_paths = [p for p in images_path.iterdir() if p.suffix in img_extensions]
    gt_paths = [gt_path / f"{img_path.stem}.txt" for img_path in img_paths]
    return list(zip(img_paths, gt_paths))


def load_gt(path: Path) -> np.ndarray:
    with path.open("r") as f:
        gt = [list(map(int, line.rstrip().split())) for line in f]
    assert all([len(target) == 6 for target in gt]), f"Wrong target format {path}"
    return np.array(gt, dtype=np.int32)


def load_image(img_path: Union[Path, str]) -> Image.Image:
    return Image.open(img_path).convert("RGB")


def draw_gt_labels(img: np.ndarray, gt_labels: np.ndarray) -> None:   
    img = np.array(img)
    for gt_label in gt_labels:
        keypoint = gt_label[:2]
        xy_min = keypoint - gt_label[2:4]
        xy_max = keypoint + gt_label[2:4]
        
        img = cv2.circle(img, tuple(keypoint), 4, (255, 0, 0), -1)
        img = cv2.rectangle(img, tuple(xy_min), tuple(xy_max), (0, 255, 0))
    return img


def gauss2D(shape: Tuple[int, int], sigma: float = 0.5) -> np.ndarray:
    my, mx = [(x - 1) / 2 for x in shape]
    y, x = np.ogrid[-my:my + 1, -mx:mx + 1]
    gmap = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    normalizer = gmap.sum()
    if normalizer != 0:
        gmap /= normalizer
    return gmap


def generate_density_map(
    size: Tuple[int, int],
    keypoints: np.ndarray,
    kernel_size: int = 30,
    sigma: float = 8
) -> np.ndarray:
    w, h = size
    keypoints = keypoints.astype(np.int32)
    density_map = np.zeros((h, w), dtype=np.float32)

    for keypoint in keypoints:
        keypoint = np.clip(keypoint, a_min=1, a_max=[w, h])
        x1, y1 = np.clip(np.array(keypoint - kernel_size // 2), a_min=1, a_max=[w, h])
        x2, y2 = np.clip(np.array(keypoint + kernel_size // 2), a_min=1, a_max=[w, h])
        gmap = gauss2D((y2 - y1 + 1, x2 - x1 + 1), sigma)
        density_map[y1 - 1:y2, x1 - 1:x2] = density_map[y1 - 1:y2, x1 - 1:x2] + gmap

    return density_map


def random_crop(
    img: Image.Image,
    density_map: np.ndarray,
    input_h: int = 512,
    input_w: int = 512
) -> Tuple[Image.Image, np.ndarray]:
    img_w, img_h = img.size

    padded_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    padded_density_map = np.zeros((input_h, input_w), dtype=np.float32)

    crop_min_x = np.random.randint(0, max(1, img_w - input_w))
    crop_min_y = np.random.randint(0, max(1, img_h - input_h))
    crop_max_x = min(crop_min_x + input_w, img_w)
    crop_max_y = min(crop_min_y + input_h, img_h)

    crop_w, crop_h = min(input_w, img_w), min(input_h, img_h)
    padded_img[:crop_h, :crop_w, :] = np.array(img)[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :]
    padded_density_map[:crop_h, :crop_w] = density_map[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    return Image.fromarray(padded_img), padded_density_map


def resize_img(img: Image.Image, min_size: int = 512, max_size: int = 1536) -> Tuple[Image.Image, float]:
    resize_ratio = 1.0
    if img.size[0] > max_size or img.size[1] > max_size:
        resize_ratio = max_size / img.size[0] if img.size[0] > max_size else max_size / img.size[1]
        resized_w = int(np.ceil(img.size[0] * resize_ratio))
        resized_h = int(np.ceil(img.size[1] * resize_ratio))
        img = img.resize((resized_w, resized_h))

    if img.size[0] < min_size or img.size[1] < min_size:
        resize_ratio = min_size / img.size[0] if img.size[0] < min_size else min_size / img.size[1]
        resized_w = int(np.ceil(img.size[0] * resize_ratio))
        resized_h = int(np.ceil(img.size[1] * resize_ratio))
        img = img.resize((resized_w, resized_h))
    
    return img, resize_ratio


def scale_density_map(density_map: np.ndarray, scale_factor: int = 1):
    scaled_density_map = cv2.resize(
        density_map,
        (density_map.shape[1] // scale_factor, density_map.shape[0] // scale_factor),
        interpolation=cv2.INTER_CUBIC
    )
    scaled_density_map *= (scale_factor ** 2)
    return scaled_density_map


def seed_worker(worker_id: int) -> None:
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class JHUCrowdDataset(Dataset):
    img_extensions = {".jpg"}

    def __init__(
        self,
        dataset_root: Path,
        subset_name: str = "train",
        transform: Optional[Callable[[Image.Image, np.ndarray], Tuple[Tensor, np.ndarray]]] = None,
        scale_factor: int = 1,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        min_crowd_size: int = 0,
        force_pregenerate: bool = False,
        num_workers: int = 8,
        cache_root: Path = Path("./cache"),
        cache: bool = True
    ) -> None:
        super(JHUCrowdDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset_name = subset_name
        self.transform = transform
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.max_size = max_size
        self.min_crowd_size = min_crowd_size
        self.num_workers = num_workers
        self.cache_dir = cache_root / subset_name
        self.cache = cache

        img_labels_df = self.load_img_labels(dataset_root / subset_name)
        self.img_labels_df = img_labels_df[img_labels_df["count"] >= min_crowd_size]

        if force_pregenerate and cache:
            self.pregenerate_density_maps()

    def load_img_labels(self, dataset_root: Path) -> pd.DataFrame:
        # Load image_labels.txt
        img_info_df = pd.read_csv(
            dataset_root / "image_labels.txt",
            names=["name", "count", "scene_type", "weather_condition", "distractor"],
            dtype={"name": str, "count": int, "scene_type": str, "weather_condition": int, "distractor": int}
        )
        img_info_df.set_index("name", inplace=True)

        # Load image & gt path combinations"
        img_gt_paths = load_dataset_paths(dataset_root, self.img_extensions)
        names = [img_path.stem for img_path, _ in img_gt_paths]

        img_gt_paths_df = pd.DataFrame(img_gt_paths, columns=["img_path", "gt_path"], index=names)
        gt_labels = pd.Series([
            load_gt(gt_path) for gt_path in img_gt_paths_df["gt_path"]], name="gt_labels", index=names)
        gt_counts = pd.Series([len(gt) for gt in gt_labels], index=names)

        # Check integrity
        assert len(img_info_df) == len(img_gt_paths)
        assert len(img_gt_paths_df.index.difference(img_info_df.index)) == 0
        assert all(img_info_df["count"].eq(gt_counts))

        return pd.concat([img_gt_paths_df, gt_labels, img_info_df], axis=1)

    def pregenerate_density_maps(self) -> None:
        with Pool(self.num_workers) as p:
            p.map(self.__getitem__, range(len(self)))

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.ndarray]:
        sample = self.img_labels_df.iloc[index]

        cached_img_path = self.cache_dir / f"{sample.name}.jpg"
        cached_density_path = self.cache_dir / f"{sample.name}.npy"
        if self.cache and cached_density_path.exists() and cached_img_path.exists():
            img = load_image(cached_img_path)
            density_map = np.load(str(cached_density_path))
        else:
            img = load_image(sample["img_path"])
            keypoints = np.empty((0, 2))
            if len(sample["gt_labels"] > 0):
                keypoints = sample["gt_labels"][:, :2]
            
            if self.min_size is not None and self.max_size is not None:
                img, resize_ratio = resize_img(img, min_size=self.min_size, max_size=self.max_size)
                keypoints = (keypoints * resize_ratio).astype(np.int32)
            density_map = generate_density_map(img.size, keypoints)

            if self.cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(self.cache_dir / f"{sample.name}.npy"), density_map)
                img.save(str(self.cache_dir / f"{sample.name}.jpg"))

        if self.transform is not None:
            img, density_map = self.transform(img, density_map)

        density_map = scale_density_map(density_map, self.scale_factor)

        return img, density_map

    def __len__(self) -> int:
        return len(self.img_labels_df)

    @staticmethod
    def collate_fn(samples: List[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        img_batch = torch.stack([sample[0] for sample in samples], 0)
        density_map_batch = torch.stack([torch.from_numpy(sample[1]).unsqueeze(0) for sample in samples], 0)
        return [img_batch, density_map_batch]

train_dataset = JHUCrowdDataset(dataset_root=DATASET_ROOT, subset_name="train", min_crowd_size=50)
valid_dataset = JHUCrowdDataset(dataset_root=DATASET_ROOT, subset_name="val", min_crowd_size=50)
test_dataset = JHUCrowdDataset(dataset_root=DATASET_ROOT, subset_name="test", min_crowd_size=50)

all_img_labels_df = pd.concat([
    train_dataset.img_labels_df,
    valid_dataset.img_labels_df,
    test_dataset.img_labels_df
])

print("Train data set size:", len(train_dataset))
print("Valid data set size:", len(valid_dataset))
print("Test data set size:", len(test_dataset))
print("Total size:", len(all_img_labels_df))

train_dataset.img_labels_df.head()

random_indices = np.random.choice(train_dataset.img_labels_df.index, 6)

fig, ax = plt.subplots(3, 2, figsize=(16, 16))

for i, index in enumerate(random_indices):
    sample = train_dataset.img_labels_df.loc[index]
    img = load_image(sample["img_path"])
    gt_labels = sample["gt_labels"]
    ax[i // 2, i % 2].imshow(draw_gt_labels(img, gt_labels))

train_gt_labels = np.array([gt for gt_labels in train_dataset.img_labels_df["gt_labels"] for gt in gt_labels])
valid_gt_labels = np.array([gt for gt_labels in valid_dataset.img_labels_df["gt_labels"] for gt in gt_labels])
test_gt_labels = np.array([gt for gt_labels in test_dataset.img_labels_df["gt_labels"] for gt in gt_labels])
all_gt_labels = np.concatenate([train_gt_labels, valid_gt_labels, test_gt_labels])

print("Total number of train annotations:", len(train_gt_labels))
print("Total number of valid annotations:", len(valid_gt_labels))
print("Total number of test annotations:", len(test_gt_labels))
print("Total number of annotations:", len(all_gt_labels))

train_occlusion_levels = dict(zip(*np.unique(train_gt_labels[:, 4], return_counts=True)))
valid_occlusion_levels = dict(zip(*np.unique(valid_gt_labels[:, 4], return_counts=True)))
test_occlusion_levels = dict(zip(*np.unique(test_gt_labels[:, 4], return_counts=True)))

labels = ["Train", "Valid", "Test"]
visible_counts = np.array([
    train_occlusion_levels[1],
    valid_occlusion_levels[1],
    test_occlusion_levels[1]
])
partial_occlusion_counts = np.array([
    train_occlusion_levels[2],
    valid_occlusion_levels[2],
    test_occlusion_levels[2]
])
full_occlusion_counts = np.array([
    train_occlusion_levels[3],
    valid_occlusion_levels[3],
    test_occlusion_levels[3]
])

train_dataset.img_labels_df["count"].describe()

valid_dataset.img_labels_df["count"].describe()

test_dataset.img_labels_df["count"].describe()

all_img_labels_df["count"].describe()

img_resolutions = []
for img_path in all_img_labels_df["img_path"]:
    img = load_image(img_path)
    img_resolutions.append(img.size)
img_resolutions = np.array(img_resolutions)

min_resolution = np.min(img_resolutions, axis=0)
max_resolution = np.max(img_resolutions, axis=0)
mean_resolution = np.mean(img_resolutions, axis=0, dtype=np.int32)
median_resolution = np.median(img_resolutions, axis=0)

print("Minimum image width:", min_resolution[0])
print("Minimum image height:", min_resolution[1])
print("Maximum image width:", max_resolution[0])
print("Maximum image height:", max_resolution[1])
print(f"Mean image resolution: {mean_resolution[0]}x{mean_resolution[1]}")
print(f"Median image resolution: {median_resolution[0]}x{median_resolution[1]}")

scene_types_counts = dict(zip(*np.unique(all_img_labels_df["scene_type"], return_counts=True)))

for scene_type, count in scene_types_counts.items():
    print(scene_type, ":", count, end=", ")

    scene_types = np.array(list(scene_types_counts.keys()))
scene_type_counts = np.array(list(scene_types_counts.values()))

random_indices = np.random.randint(0, len(train_dataset), 6)

fig, ax = plt.subplots(3, 2, figsize=(16, 16))

for i, index in enumerate(random_indices):
    img, density_map = train_dataset[index]
    ax[i // 2, i % 2].imshow(img)
    ax[i // 2, i % 2].imshow(density_map, cmap="jet", alpha=0.5)


checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINTS_DIR,
    filename="crowdnet-{epoch:02d}-{valid_mae:.2f}",
    monitor="valid_mae",
    mode="min",
    verbose=True,
    save_top_k=5
)
trainer = pl.Trainer(
    gpus=[0],
    fast_dev_run=False,
    max_epochs=50,
    checkpoint_callback=True,
    callbacks=[checkpoint_callback, MetricsCallback()],
    deterministic=True,
    progress_bar_refresh_rate=0,
    num_sanity_val_steps=0
)

def count_from_density_map(density_map: np.ndarray) -> int:
    return round(np.sum(np.clip(density_map, a_min=0, a_max=None)))

def predict_density_map(img: Image.Image, model: nn.Module) -> np.ndarray:
    img, resize_ratio = resize_img(img)
    img_input = preprocess_transform(img).unsqueeze(dim=0)
    density_map = model(img_input).detach().cpu().numpy()[0][0]
    return density_map

random_indices = np.random.randint(0, len(test_dataset), 10)

fig, axes = plt.subplots(10, 3, figsize=(16, 32))

column_names = ["image", "gt", "pred"]
for ax, col in zip(axes[0], column_names):
    ax.set_title(col)

for i, index in enumerate(random_indices):
    img, density_map = train_dataset[index]

    gt_count = count_from_density_map(density_map)
    density_map = scale_density_map(density_map, 8)
    
    
    axes[i, 0].imshow(img)
    axes[i, 1].imshow(density_map, cmap="jet")
    
    axes[i, 1].text(
        x=4,
        y=10,
        s=f"GT count: {gt_count}",
        bbox={"facecolor": "black", "alpha": 0.7, "pad": 4},
        c="white",
        weight="bold"
    )
    
    axes[i, 2].text(
        x=4,
        y=10,
        bbox={"facecolor": "black", "alpha": 0.7, "pad": 4},
        c="white",
        weight="bold"
    )
