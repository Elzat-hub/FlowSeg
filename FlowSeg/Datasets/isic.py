import torch
import torch.utils.data
import os
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from mpi4py import MPI 

from .transforms import Compose, ToTensor, ToPILImage, Normalize, Resize, CenterCrop, \
                      RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, ColorJitter

def cv2_loader(path, is_mask):
    if is_mask:
        img = imageio.imread(path)
        img[img > 0] = 1
    else:
        img = imageio.imread(path)
    return img

class LesionAwareCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, mask):
        # 1. 找到病灶区域的边界框
        mask_np = np.array(mask)
        # 找到所有非零像素的坐标
        y_coords, x_coords = np.where(mask_np > 0)

        # 如果没有病灶，就退化为中心裁剪
        if len(y_coords) == 0:
            w, h = img.size
            left = (w - self.crop_size) // 2
            top = (h - self.crop_size) // 2
            right = left + self.crop_size
            bottom = top + self.crop_size
        else:
            # 找到最小和最大的 x, y 坐标
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            # 计算边界框的中心点
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # 计算裁剪框的左上角坐标
            left = int(center_x - self.crop_size / 2)
            top = int(center_y - self.crop_size / 2)

            # 确保裁剪框不会超出图像边界
            left = max(0, left)
            top = max(0, top)

            if left + self.crop_size > img.size[0]:
                left = img.size[0] - self.crop_size
            if top + self.crop_size > img.size[1]:
                top = img.size[1] - self.crop_size
            
            right = left + self.crop_size
            bottom = top + self.crop_size
        
        # 2. 进行裁剪
        img = img.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))

        return img, mask

def get_isic_transform(image_size):
    lesion_crop = LesionAwareCrop(crop_size=image_size)

    transform_train = Compose([
        ToPILImage(), 
        Resize((256, 256)),  
        # lesion_crop,
        RandomHorizontalFlip(), 
        RandomVerticalFlip(),   
        RandomAffine(int(22), scale=(float(0.75), float(1.25))), 
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), 
        ToTensor(), 
        Normalize(mean=[175.80, 134.73, 116.28], std=[39.79, 44.34, 48.63])
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((256, 256)), 
        # lesion_crop,
        ToTensor(),
        Normalize(mean=[175.80, 134.73, 116.28], std=[39.79, 44.34, 48.63])
    ])
    return transform_train, transform_test

def create_dataset(mode="train", image_size=256):

    datadir = '/media/data1/yili/data/ISIC'

    transform_train, transform_test = get_isic_transform(image_size)

    if mode == "train":
        return ISICDataset(datadir, train=True, transform=transform_train, image_size=image_size)
    else: 
        return ISICDataset(datadir, train=False, transform=transform_test, image_size=image_size)
    
def load_data(
    *, data_dir, batch_size, image_size, class_name, class_cond=False, expansion, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """

    dataset = create_dataset(mode="train")   # 'train' or 'test'

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True  
        )   
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
        )
    while True:
        yield from loader

class ISICDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=cv2_loader, pSize=8, image_size=256):
        self.root = root
        
        if train:
            self.imgs_root = os.path.join(self.root, 'train/image')
            self.masks_root = os.path.join(self.root, 'train/label')
        else:
            self.imgs_root = os.path.join(self.root, 'test/image')
            self.masks_root = os.path.join(self.root, 'test/label')
        self.image_size = image_size
        
        all_image_files = sorted([f for f in os.listdir(self.imgs_root) if f.endswith('.jpg')])
        self.paths = [Path(f).stem for f in all_image_files]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.pSize = pSize
        
        self.imgs = []
        self.masks = []
        self.mean = torch.from_numpy(np.array([175.80, 134.73, 116.28]))
        self.std = torch.from_numpy(np.array([39.79, 44.34, 48.63]))

        shard = MPI.COMM_WORLD.Get_rank()
        num_shards = MPI.COMM_WORLD.Get_size()

        print(f"Loading ISIC data from: {self.imgs_root} and {self.masks_root}")
        for file_stem in tqdm(self.paths, desc="Loading ISIC data"):
            img_path = os.path.join(self.imgs_root, f"{file_stem}.jpg")
            mask_path = os.path.join(self.masks_root, f"{file_stem}_Segmentation.png") 

            self.imgs.append(self.loader(img_path, is_mask=False))
            self.masks.append(self.loader(mask_path, is_mask=True))

        self.imgs = self.imgs[shard::num_shards]
        self.masks = self.masks[shard::num_shards]
        self.paths = self.paths[shard::num_shards] 

        print(f'Num of data for this shard: {len(self.paths)}')

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]

        img, mask = self.transform(img, mask)
        out_dict = {"conditioned_image": img}
        mask = 2 * mask - 1.0 

        return mask.unsqueeze(0), out_dict, f"{Path(self.paths[index]).stem}_{index}"

    def __len__(self):
        return len(self.paths)

if __name__ == "__main__":
    
    print("--- Testing Train Dataset ---")
    train_dataset = create_dataset(
        mode='train', 
        image_size=256,
    )

    ds = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0, 
        shuffle=True,
        drop_last=True
    )
    pbar = tqdm(ds)
    mean0_list = []
    mean1_list = []
    mean2_list = []
    std0_list = []
    std1_list = []
    std2_list = []
    for i, (mask, out_dict, _) in enumerate(pbar):
        img = out_dict["conditioned_image"]
        plt.imshow(img.squeeze().permute(1,2,0).numpy().astype(np.uint8))
        plt.show()

        plt.imshow(mask.squeeze().numpy(), cmap='gray')
        plt.show()
        a = img.mean(dim=(0, 2, 3))
        b = img.std(dim=(0, 2, 3))
        mean0_list.append(a[0].item())
        mean1_list.append(a[1].item())
        mean2_list.append(a[2].item())
        std0_list.append(b[0].item())
        std1_list.append(b[1].item())
        std2_list.append(b[2].item())
    print(np.mean(mean0_list))
    print(np.mean(mean1_list))
    print(np.mean(mean2_list))

    print(np.mean(std0_list))
    print(np.mean(std1_list))
    print(np.mean(std2_list))
    # pbar_train = tqdm(train_dataloader, desc="Processing Train Data")
    # for i, (mask, out_dict, img_name) in enumerate(pbar_train):
    #     img = out_dict["conditioned_image"]
    #     # mask 是 (B, H, W) 或 (B, 1, H, W)
    #     # img 是 (B, C, H, W)

    #     print(f"Train Batch {i}: Image shape {img.shape}, Mask shape {mask.shape}, Image name: {img_name[0]}")
        
    #     # 打印经过 Normalize 后的均值和标准差，理论上应该接近 0 和 1
    #     # 这可以验证 Normalize 是否工作正常
    #     mean_after_norm = img.mean(dim=(0, 2, 3))
    #     std_after_norm = img.std(dim=(0, 2, 3))
    #     # print(f"    Mean (after Normalize): {mean_after_norm.tolist()}")
    #     # print(f"    Std (after Normalize): {std_after_norm.tolist()}")

    #     if i == 5: # 只打印前几批数据
    #         break
    # print("--- Train Dataset Test Complete ---")


    # # 测试测试集加载
    # print("\n--- Testing Test Dataset ---")
    # test_dataset = create_dataset(
    #     mode='test', # 明确指定模式
    #     image_size=256,
    # )

    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     num_workers=0, # 测试时可以将 num_workers 设为 0 以方便调试
    #     shuffle=False,
    #     drop_last=True
    # )

    # pbar_test = tqdm(test_dataloader, desc="Processing Test Data")
    # for i, (mask, out_dict, img_name) in enumerate(pbar_test):
    #     img = out_dict["conditioned_image"]
        
    #     print(f"Test Batch {i}: Image shape {img.shape}, Mask shape {mask.shape}, Image name: {img_name[0]}")
        
    #     mean_after_norm = img.mean(dim=(0, 2, 3))
    #     std_after_norm = img.std(dim=(0, 2, 3))
    #     print(f"    Mean (after Normalize): {mean_after_norm.tolist()}")
    #     print(f"    Std (after Normalize): {std_after_norm.tolist()}")

    #     if i == 5: # 只打印前几批数据
    #         break
    # print("--- Test Dataset Test Complete ---")