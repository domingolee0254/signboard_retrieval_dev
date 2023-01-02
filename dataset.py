import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from os.path import join
from PIL import Image

class AugmentedDataset(Dataset):
    def __init__(self, img_paths, img_size):
        self.img_paths = img_paths
        self.img_size = img_size
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(f'{img_path}')
        img = self.transform(image=img)
        return {'path': img_path, 'img': img['image']}

class KeepLatioDataset(Dataset):
    def __init__(self, img_paths, img_size):
        self.img_paths = img_paths
        self.img_size = img_size
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0, value=(0,0,0)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(f'{img_path}')
        img = self.transform(image=img)
        return {'path': img_path, 'img': img['image']}

if __name__ == '__main__':
    img_paths = './imgs/near_duplicate.jpg'
    img = cv2.imread(f'{img_paths}')
    transform = A.Compose([
        A.LongestMaxSize(max_size=224, interpolation=0, p=1.0),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0,0,0)),
        A.Resize(224,224)
        ])
    augmented_image = transform(image=img)['image']
    tmp = Image.fromarray(augmented_image)
    tmp.save("./geeks.jpg")
