import os
import re
import torch
import numpy as np
import matplotlib.image as mpimg
import PIL.Image as Image
from sklearn.metrics import f1_score, jaccard_score


def load_model(model, optimizer, args):
    """
    Loads states for weights and the optimizer
    """
    checkpoint = torch.load(args.weight_path, map_location=torch.device('cuda' if args.cuda else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_scores(output, mask):
    """
    Calculate different scores such as F1 score and IoU for the prediction
    """
    labels = torch.argmax(output, dim=1)
    mask_ = np.reshape(mask.cpu().numpy(), (mask.shape[0], -1))
    labels_ = np.reshape(labels.cpu().numpy(), (labels.shape[0], -1))
    # Calculating f1_score
    f_score = f1_score(mask_, labels_, average='macro')
    # Calculating IoU score for segmentation
    IoU = jaccard_score(mask_, labels_, average='macro')

    return f_score, IoU


def save_model(model, optimizer, path, args):
    """
    Saves the model and the optimizer states
    """
    save_path = os.path.join(path, args.experiment_name + '.pt')
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)


def save_image(output, idx, path):
    labels = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    img = Image.fromarray(labels.astype(np.uint8))
    img.save(os.path.join(path, 'satImage_{:03d}.png'.format(idx)))


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
