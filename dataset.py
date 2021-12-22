import os
import random
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from utils import random_erase


class TrainValSet(Dataset):

    def __init__(self, path, set_type, ratio, rotate=True, flip=True, resize=None, random_crops=0):
        super(Dataset, self).__init__()

        # Get image and ground truth paths
        images_path = Path(path) / 'training' / 'images'
        gt_path = Path(path) / 'training' / 'groundtruth'

        # Listing the images and ground truth file paths
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

        # Divide to validation and training set based on the value of set_type
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
        self.random_crops = random_crops

    def transform(self, img, mask, index):
        """
        Augmenting the dataset by doing random flip or rotate or justify dataset by resizing
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

        # Transforming from PIL type to torch.tensor and normalizing the data to range [0, 1]
        to_tensor = transforms.ToTensor()
        img, mask = to_tensor(img), to_tensor(mask)

        # Erasing random rectangles from the image
        img = random_erase(img, n=self.random_crops, rgb='noise')

        return img, mask.round().long()

    def __getitem__(self, index):
        if self.rotate:
            # Getting the each image 6 times, each time with a different diagonal rotation
            img, mask = self.images[index // 6], self.gt[index // 6]
        else:
            img, mask = self.images[index], self.gt[index]

        # Read image and ground truth files
        img = Image.open(img)
        mask = Image.open(mask)

        # Apply dataset augmentation transforms if needed
        img, mask = self.transform(img, mask, index)

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

        # Transforming from PIL type to torch.tensor and normalizing the data to range [0, 1]
        img = transforms.ToTensor()(img)

        return img

    def __len__(self):
        return len(self.images)

