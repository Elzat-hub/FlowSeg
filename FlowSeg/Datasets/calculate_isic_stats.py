import sys
sys.path.append('..')
import torch
import torch.utils.data
import os
import imageio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Datasets.transforms import Compose, ToTensor, ToPILImage, Resize, CenterCrop


def cv2_loader(path, is_mask):
    if is_mask:
        img = imageio.imread(path)
        img[img > 0] = 1
    else:
        img = imageio.imread(path)
    return img

class IsicMeanStdDataset(torch.utils.data.Dataset):
    def __init__(self, root, loader=cv2_loader):
        self.root = root
        self.loader = loader
        
        self.imgs_root = os.path.join(self.root, 'train/image') 
        
        all_image_files = sorted([f for f in os.listdir(self.imgs_root) if f.endswith('.jpg')])
        self.paths = [Path(f).stem for f in all_image_files]

        print(f"Collecting image paths from: {self.imgs_root}")
        self.image_paths = []
        for file_stem in tqdm(self.paths, desc="Preparing image paths"):
            img_path = os.path.join(self.imgs_root, f"{file_stem}.jpg")
            self.image_paths.append(img_path)

        self.transform = Compose([
            ToPILImage(),
            Resize((512, 512)),  
            CenterCrop((256, 256)), 
            ToTensor(), 
        ])

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = self.loader(img_path, is_mask=False)

        img_tensor, _ = self.transform(img, np.zeros_like(img[:,:,0])) 
        return img_tensor

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    ISIC_DATA_ROOT = '/media/data1/yili/data/ISIC'

    dataset = IsicMeanStdDataset(root=ISIC_DATA_ROOT)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32, 
        num_workers=8, 
        shuffle=False
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    print("Calculating mean and standard deviation for ISIC training set...")
    for images in tqdm(dataloader):
        batch_samples = images.size(0)
        mean += images.mean(dim=(0, 2, 3)) * batch_samples
        std += images.std(dim=(0, 2, 3)) * batch_samples
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    print("\n--- ISIC Dataset Mean and Standard Deviation ---")
    print(f"Mean (R, G, B): {mean.tolist()}")
    print(f"Std (R, G, B): {std.tolist()}")
    print("\nThese values should be used in the Normalize transform in isic.py.")
    print("Example: Normalize(mean=[{:.2f}, {:.2f}, {:.2f}], std=[{:.2f}, {:.2f}, {:.2f}])".format(
          mean[0], mean[1], mean[2], std[0], std[1], std[2]))