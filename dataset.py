import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from utils import random_erase


def get_diag_mask(mask):
    diag_mask = np.asarray(mask.copy())

    hs = cv2.getStructuringElement(cv2.MORPH_RECT, (diag_mask.shape[1], 1))
    horizontal = cv2.erode(diag_mask, hs)
    horizontal = cv2.dilate(horizontal, hs)

    vs = cv2.getStructuringElement(cv2.MORPH_RECT, (1, diag_mask.shape[0]))
    vertical = cv2.erode(diag_mask, vs)
    vertical = cv2.dilate(vertical, vs)

    diag_mask = diag_mask - np.clip(vertical + horizontal, a_min=0, a_max=255)

    return diag_mask


class TrainValSet(Dataset):

    def __init__(self, path, set_type, ratio, rotate=True, flip=True, resize=None, diag_mask=True, random_crops=0):
        super(Dataset, self).__init__()

        # Get image and ground truth paths
        images_path = Path(path) / 'training' / 'images'
        gt_path = Path(path) / 'training' / 'groundtruth'

        # Listing the images and ground truth files
        self.images = [
            images_path / item
            for item in os.listdir(images_path)
            if item.endswith('.png')
        ]
        self.images.sort()

        self.gt = [
            gt_path / item
            for item in os.listdir(gt_path)
            if item.endswith('.png')
        ]
        self.gt.sort()

        # divide to validation and training set based on the value of set_type
        idx = int(len(self.images) * ratio)
        if set_type == 'train':
            self.images = self.images[idx:]
            self.gt = self.gt[idx:]
        elif set_type == 'val':
            self.images = self.images[:idx]
            self.gt = self.gt[:idx]
        else:
            raise Exception("set_type is not correct")

        self.set_type = set_type
        self.rotate = rotate
        self.flip = flip
        self.resize = resize
        self.diag_mask = diag_mask
        self.random_crops = random_crops

    def transform(self, img, mask, index):
        """
        Augmenting the dataset by doing random flip or random rotate and justify dataset by resizing
        """

        # Resize
        if self.resize:
            img = TF.resize(img, self.resize)
            mask = TF.resize(mask, self.resize)

        # Do a vertical or horizontal flip randomly
        if self.flip and random.random() > 0.33:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            else:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        # First apply a rotate based on diag_angles to extend dataset with non-horizontal and non-vertical roads and
        # then do a random rotate
        if self.rotate:
            diag_angles = [0, 15, 30, 45, 60, 75]
            img = TF.rotate(img, diag_angles[index % 6])
            mask = TF.rotate(mask, diag_angles[index % 6])
            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        to_tensor = transforms.ToTensor()

        diag_mask = -1
        if self.diag_mask and self.set_type == 'train':
            diag_mask = get_diag_mask(mask)
            diag_mask = to_tensor(diag_mask).round().long()

        img, mask = to_tensor(img), to_tensor(mask)

        # Erasing random rectangles from the image
        img = random_erase(img, n=self.random_crops, rgb='noise')

        return img, mask.round().long(), diag_mask

    def __getitem__(self, index):
        if self.rotate:
            img, mask = self.images[index // 6], self.gt[index // 6]
        else:
            img, mask = self.images[index], self.gt[index]

        # Read image and ground truth files
        img = Image.open(img)
        mask = Image.open(mask)

        # Apply dataset augmentation transforms if needed
        img, mask, diag_mask = self.transform(img, mask, index)

        if self.set_type == 'train':
            return img, mask, diag_mask
        return img, mask

    def __len__(self):
        if self.rotate:
            return len(self.images) * 6
        return len(self.images)


class TestSet(Dataset):

    def __init__(self, path):
        super(Dataset, self).__init__()

        # Get image and ground truth paths
        images_path = os.path.join(path, 'test_set_images')

        self.images = [
            os.path.join(images_path, item, item + '.png')
            for item in os.listdir(images_path)
            if os.path.isdir(os.path.join(images_path, item))
        ]
        self.images.sort(key=lambda x: int(os.path.split(x)[-1][5:-4]))

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img)
        img = transforms.ToTensor()(img)

        return img

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = TrainValSet(path='./dataset', set_type='train', ratio=0.2)
    ds = DataLoader(dataset=ds)
    for img, mask, diag_mask in ds:
        print(img.shape, mask.shape, diag_mask.shape)
    print(len(ds))

    ds = TrainValSet(path='./dataset', set_type='val', ratio=0.2)
    ds = DataLoader(dataset=ds)
    for img, mask in ds:
        print(img.shape, mask.shape)
    print(len(ds))

    ds = TestSet(path='./dataset')
    ds = DataLoader(dataset=ds)
    for img in ds:
        print(img.shape)
    print(len(ds))
