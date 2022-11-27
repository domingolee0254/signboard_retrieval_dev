import argparse
import os
import torch
import albumentations as A
import cv2 as cv
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
######
#1. load data 
#2. transformation
#3. save images
######

########data load and transform########## 
#out: images
def transform_data(args, files, file_list):
    for idx, ith_img in enumerate(file_list):
        print(f"ith_img[] is {ith_img[9:-4]}") # q_v_45
        img=cv.imread(ith_img)
        #img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if args.aug_ver == 'rotate':
            transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                              interpolation=1, border_mode=0, value=0, p=1)
        ])
        elif args.aug_ver == 'contrast':
            transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.2, 0.2), p=1.0)
        ])
        elif args.aug_ver == 'v_flip':
            transform = A.Compose([
            A.VerticalFlip(p=1)
        ])
        elif args.aug_ver == 'h_flip':
            transform = A.Compose([
            A.HorizontalFlip(p=1)
        ])
        elif args.aug_ver == 'blur':
            transform = A.Compose([
            A.GaussianBlur(blur_limit=(19, 21), sigma_limit=10, always_apply=True, p=1)
        ])
        elif args.aug_ver == 'color':
            transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=1, always_apply=True, p=1)
        ])    
        augmented_image = transform(image=img)['image']
        #print(augmented_image)
        cv.imwrite("./02.save_dir/new_rotate/"+ ith_img[9:-4] + "_" + args.aug_ver + ".png", augmented_image)    

        #cv.imwrite(args.save_dir + "/" + args.aug_ver +"/"+ ith_img[13:-4] + "_" + args.aug_ver + ".png", augmented_image)    
####

def main(args, files, files_list):
    raw_imgs = transform_data(args, files, files_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default='./origin/')
    parser.add_argument('-av', '--aug_ver', type=str, default='')
    parser.add_argument('-sd', '--save_dir', type=str, default='./02.save_dir')
    args = parser.parse_args()

    files_list = []
    for (root, dirs, files) in os.walk(args.data_dir):
        if len(files) > 0:  ##files is a list with files_name
            for file_name in files:
                tmp_fname_string = root + file_name
                files_list.append(tmp_fname_string)
    main(args, files, files_list)
    #print(f"files_list: {files_list}")
    #print(f"img_path {img_path}")

