import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import PIL.Image as Image
from sklearn.metrics import f1_score


def load_model(model, optimizer, args):
    """
    Loads states for weights and the optimizer
    """
    checkpoint = torch.load(args.weight_path, map_location=torch.device('cuda' if args.cuda else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def save_model(model, optimizer, path, args):
    """
    Saves the model and the optimizer states
    """
    save_path = os.path.join(path, args.experiment_name + '.pt')
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)


# Dice loss derived from https://github.com/mateuszbuda/brain-segmentation-pytorch
def dice_loss(output, mask, smooth=1.0):
    output = output[:, 0].contiguous().view(-1)
    mask = mask[:, 0].contiguous().view(-1)
    intersection = (output * mask).sum()
    dsc = (2. * intersection + smooth) / (
            output.sum() + mask.sum() + smooth
    )
    return 1. - dsc


def fgsm_update(img, output, mask, update_max_norm=0.25):
    data_grad = img.grad
    idx = (output > 0.5) == mask
    return torch.clamp(img + idx * update_max_norm * torch.sign(data_grad), min=0, max=1)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_score(output, mask, threshold=0.5):
    """
    Calculate F1 score of the prediction
    """
    labels_ = output > threshold
    mask_ = np.reshape(mask.cpu().numpy(), (mask.shape[0], -1))
    labels_ = np.reshape(labels_.cpu().numpy(), (labels_.shape[0], -1))
    # Calculating f1_score
    f_score = f1_score(mask_, labels_, average='macro', zero_division=0)

    return f_score


def save_image(output, idx, path, threshold=0.5):
    labels = (output > threshold).squeeze().cpu().numpy()
    img = Image.fromarray((labels * 255).astype(np.uint8))
    img.save(os.path.join(path, 'satImage_{:03d}.png'.format(idx)))


def save_image_overlap(output, img, idx, path):
    labels = (output > 0.5).cpu().numpy().squeeze()
    img = img.cpu().numpy().squeeze() * 255
    img[2, labels] = 150
    img = np.transpose(img, (1, 2, 0)).astype('uint8')
    img = Image.fromarray(img.astype(np.uint8), 'RGB')
    img.save(os.path.join(path, 'satImage_overlap_{:03d}.png'.format(idx)))


foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


cols = {'train': {'loss': [], 'f1-score': []},
        'val': {'loss': [], 'f1-score': []}}


def save_track(path, args, train_loss=None, train_f1=None, val_loss=None, val_f1=None):
    if train_loss is not None:
        cols['train']['loss'].append(train_loss)
    if train_f1 is not None:
        cols['train']['f1-score'].append(train_f1)

    if val_loss is not None:
        cols['val']['loss'].append(val_loss)
    if val_f1 is not None:
        cols['val']['f1-score'].append(val_f1)

    df = pd.DataFrame.from_dict(cols['train'])
    df.to_csv(os.path.join(path, args.experiment_name + "_train_tracking.csv"))

    df = pd.DataFrame.from_dict(cols['val'])
    df.to_csv(os.path.join(path, args.experiment_name + "_val_tracking.csv"))
