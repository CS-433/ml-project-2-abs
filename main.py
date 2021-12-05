import argparse
import os
import torch
import dataset
from model import UNet
from torch.utils.data import DataLoader
from utils import get_score, load_model, create_folder, save_model, save_image, masks_to_submission, save_track, \
    dice_loss, save_image_overlap, fgsm_update

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="dataset path")
parser.add_argument('--validation_ratio', type=float, default=None,
                    help="the ratio of validation dataset size to the whole dataset. if not set then there will be no validation and the whole dataset is used for training")
parser.add_argument('--rotate', type=bool, default=True, help="do rotate while training")
parser.add_argument('--flip', type=bool, default=True, help="do flip while training")
parser.add_argument('--resize', type=int, default=None, help="the resize value for test images")
parser.add_argument('--batch_size', type=int, default=8, help="the batch size for the training")
parser.add_argument('--cuda', type=int, default=1, help="0 or 1, if 1 then the model uses gpu for the training")
parser.add_argument('--lr', type=float, default=0.001, help="the learning rate value")
parser.add_argument('--weight_path', type=str, default=None,
                    help="the path to saved weights. if not specified there will be no weight loaded")
parser.add_argument('--experiment_name', type=str, default="Road Segmentation", help="the name of the experiment")
parser.add_argument('--train', type=bool, default=True, help="if true then training is done")
parser.add_argument('--test', type=bool, default=True, help="if true then test is done")
parser.add_argument('--epochs', type=int, default=100, help="number of epoch")
parser.add_argument('--save_weights', type=bool, default=False, help="if true then the weights are saved in each epoch")
parser.add_argument('--loss', type=str, default="dice",
                    help="selects the loss type. the accepted values are \"dice\", \"cross entropy\" and \"dice + cross entropy\"")
parser.add_argument('--loss_weight', type=float, default=0,
                    help="if non-zero then an extra weighed loss is calculated for non-vertical-horizontal pixels in mask using the given value as the weight")
parser.add_argument('--adversarial_bound', type=float, default=0,
                    help="if non-zero then the training is done using adversarial attack, where epsilon is the given value")


def main(args):
    # Dataset initialization
    ratio = args.validation_ratio if args.validation_ratio else 0
    train_dataset = dataset.TrainValSet(path=args.path, set_type='train', ratio=ratio, rotate=args.rotate, flip=args.flip, diag_mask=args.loss_weight != 0)
    test_dataset = dataset.TestSet(path=args.path, resize=args.resize)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args.validation_ratio:
        val_dataset = dataset.TrainValSet(path=args.path, set_type='val', ratio=ratio, rotate=args.rotate, flip=args.flip)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    # Model initialization
    model = UNet(n_channels=3, n_classes=1)
    model = model.cuda() if args.cuda else model
    # Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    # Loading state dict for weights and optimizer state
    if args.weight_path:
        load_model(model, optimizer, args)
    # Scheduler initialization for reduction of learning rate during the training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7)

    experiment_path = os.path.join('./experiments', args.experiment_name)
    create_folder(experiment_path)

    # Loss function initialization
    if args.loss == 'dice':
        criterion = dice_loss
    elif args.loss == 'cross entropy':
        criterion = torch.nn.BCELoss(reduction='mean')
        criterion = criterion.cuda() if args.cuda else criterion
    elif args.loss == 'dice + cross entropy':
        ce = torch.nn.BCELoss(reduction='mean')
        ce = ce.cuda() if args.cuda else ce
        criterion = lambda output_, mask_: ce(output_, mask_) + dice_loss(output_, mask_)
    else:
        raise Exception("the give loss value is not defined")

    if args.train:
        for epoch in range(args.epochs):
            # Training
            model.train()
            train_loss = []
            train_f1 = []
            for img, mask, diag_mask in train_loader:
                img = img.cuda().float() if args.cuda else img.float()
                mask = mask.cuda() if args.cuda else mask
                if args.loss_weight:
                    diag_mask = diag_mask.cuda() if args.cuda else diag_mask

                optimizer.zero_grad()

                if args.adversarial_bound != 0:
                    img.requires_grad = True

                    output = model(img)
                    loss = criterion(output, mask)

                    model.zero_grad()
                    loss.backward()

                    img = fgsm_update(img, output, mask, update_max_norm=args.adversarial_bound)

                output = model(img)
                loss = criterion(output, mask)
                if args.loss_weight:
                    loss += args.loss_weight*dice_loss(output*diag_mask, diag_mask)

                loss.backward()
                optimizer.step()

                f1_score = get_score(output, mask)

                train_loss.append(loss.item())
                train_f1.append(f1_score)

            save_track(experiment_path, args,
                       train_loss=sum(train_loss) / len(train_loss),
                       train_f1=sum(train_f1) / len(train_f1))

            if args.validation_ratio:
                # Validation
                model.eval()
                val_loss = []
                val_f1 = []
                with torch.no_grad():
                    for img, mask in val_loader:
                        img = img.cuda().float() if args.cuda else img.float()
                        mask = mask.cuda() if args.cuda else mask

                        output = model(img)
                        loss = criterion(output, mask)
                        f1_score = get_score(output, mask)

                        val_loss.append(loss.item())
                        val_f1.append(f1_score)

                val_loss_to_track = sum(val_loss) / len(val_loss)
                val_f1_to_track = sum(val_f1) / len(val_f1)
                print('Epoch : {} | Loss = {:.4f}, F1 Score = {:.4f}'.format(epoch, val_loss_to_track, val_f1_to_track))
                save_track(experiment_path, args, val_loss=val_loss_to_track, val_f1=val_f1_to_track)

                scheduler.step(val_loss_to_track)

            if args.save_weights:
                save_model(model, optimizer, experiment_path, args)

    if args.test:
        # Testing
        results_path = os.path.join(experiment_path, 'results')
        create_folder(results_path)
        model.eval()
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                img = img.cuda().float() if args.cuda else img.float()

                output = model(img)

                save_image(output, i + 1, results_path)
                save_image_overlap(output, img, i + 1, results_path)

        submission_filename = os.path.join(results_path, args.experiment_name + '.csv')
        image_filenames = []
        for i in range(1, 51):
            image_filename = results_path + '/satImage_' + '%.3d' % i + '.png'
            print(image_filename)
            image_filenames.append(image_filename)
        masks_to_submission(submission_filename, *image_filenames)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
