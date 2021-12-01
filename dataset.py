import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image


class TrainValSet(Dataset):

    def __init__(self, path, set_type, ratio, rotate=True, flip=True):
        super(Dataset, self).__init__()

        # Get image and ground truth paths
        images_path = os.path.join(path, 'training', 'images')
        gt_path = os.path.join(path, 'training', 'groundtruth')

        self.images = [
            os.path.join(images_path, item)
            for item in os.listdir(images_path)
            if item.endswith('.png')
        ]
        self.images.sort()

        self.gt = [
            os.path.join(gt_path, item)
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

        self.rotate = rotate
        self.flip = flip

    def transform(self, img, mask):
        """
        Augmenting the dataset by doing random flip or random rotate
        """

        # Do a vertical or horizontal flip randomly
        if self.flip and random.random() > 0.33:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            else:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        # Do a random rotate
        if self.rotate:
            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        to_tensor = transforms.ToTensor()

        img, mask = to_tensor(img), to_tensor(mask)

        return img, mask

    def __getitem__(self, index):
        img, mask = self.images[index], self.gt[index]

        # Read image and ground truth files
        img = Image.open(img)
        mask = Image.open(mask)

        # Apply dataset augmentation transforms if needed
        img, mask = self.transform(img, mask)

        return img, mask.round().long()

    def __len__(self):
        return len(self.images)


class TestSet(Dataset):

    def __init__(self, path, resize=None):
        super(Dataset, self).__init__()

        # Get image and ground truth paths
        images_path = os.path.join(path, 'test_set_images')

        self.images = [
            os.path.join(images_path, item, item + '.png')
            for item in os.listdir(images_path)
            if os.path.isdir(os.path.join(images_path, item))
        ]
        self.images.sort(key=lambda x: int(os.path.split(x)[-1][5:-4]))

        self.resize = resize

    def transform(self, img):
        """
        Resize the image if needed
        """
        if self.resize:
            img = TF.resize(img, self.resize)
        img = transforms.ToTensor()(img)

        return img

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img)
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = TrainValSet(path='./dataset', set_type='train', ratio=0.2)
    ds = DataLoader(dataset=ds)
    for img, mask in ds:
        print(img.shape, mask.shape)
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
