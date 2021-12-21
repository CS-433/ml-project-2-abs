import argparse
import dataset
from model import UNet, UNet06, WNet0404
from torch.utils.data import DataLoader
from utils import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path',       type=str,   default='./dataset',    
    help="Dataset path")
parser.add_argument('--model',      type=str,   default="UNet",         
    help="Selects the model. Acceptable values: 'UNet', 'UNet06', 'WNet0404'")
parser.add_argument('--validation_ratio', type=float, default=None,     
    help="The ratio of validation dataset size to the whole dataset. If not set then there will be no validation and the whole dataset is used for training")
parser.add_argument('--rotate',     type=bool,  default=True,           
    help="Do rotate while training")
parser.add_argument('--flip',       type=bool,  default=True,           
    help="Do flip while training")
parser.add_argument('--resize',     type=int,   default=None,           
    help="The resize value for test images")
parser.add_argument('--random_crops', type=int, default=0,              
    help="Number of random crops for data augmentation")
parser.add_argument('--batch_size', type=int,   default=8,              
    help="The batch size for the training")
parser.add_argument('--cuda',       type=int,   default=1,              
    help="0 or 1, if 1 then the model uses gpu for the training")
parser.add_argument('--lr',         type=float, default=0.001,          
    help="The learning rate value")
parser.add_argument('--weight_path', type=str,  default=None,           
    help="The path to saved weights. if not specified there will be no weight loaded")
parser.add_argument('--experiment_name', type=str, default="NotSpec",   
    help="The name of the experiment")
parser.add_argument('--train',      type=bool,  default=True,           
    help="If true then training is done")
parser.add_argument('--test',       type=bool,  default=True,           
    help="If true then test is done")
parser.add_argument('--epochs',     type=int,   default=100,            
    help="Number of epoch")
parser.add_argument('--save_weights', type=bool, default=False,         
    help="If true then the weights are saved in each epoch")
parser.add_argument('--loss',       type=str,   default="dice",         
    help="Selects the loss type. the accepted values are 'dice', 'cross entropy' and 'dice + cross entropy'")
parser.add_argument('--adversarial_bound', type=float, default=0,       
    help="If non-zero then the training is done using adversarial attack, where epsilon is the given value")


def main(args):

    # Dataset initialization
    ratio = args.validation_ratio if args.validation_ratio else 0
    train_dataset = dataset.TrainValSet(
        path=args.path,set_type='train',
        ratio=ratio,rotate=args.rotate,
        flip=args.flip,
        resize=args.resize,
        random_crops=args.random_crops,
        )
    test_dataset = dataset.TestSet(path=args.path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args.validation_ratio:
        val_dataset = dataset.TrainValSet(
            path=args.path,
            set_type='val',
            ratio=ratio,
            rotate=args.rotate,
            flip=args.flip,
            resize=args.resize)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    # Model initialization
    if args.model == 'UNet':
        model = UNet(n_channels=3, n_classes=1)
    elif args.model == 'UNet06':
        model = UNet06(n_channels=3, n_classes=1)
    elif args.model == 'WNet0404':
        model = WNet0404(n_channels=3, n_classes=1)
    else:
        raise Exception("The given model does not exist.")
    model = model.cuda() if args.cuda else model

    # Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # Loading state dict for weights and optimizer state
    if args.weight_path:
        load_model(model, optimizer, args)
    
    # Scheduler initialization for reduction of learning rate during the training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7)

    # Creating the experiment path
    experiment_path = os.path.join('./experiments', args.experiment_name)
    create_folder(experiment_path)

    # Loss function initialization
    if args.loss == 'dice':
        criterion = dice_loss
    elif args.loss == 'dice_patches':
        criterion = lambda output, mask: dice_loss(mask_to_patches(output), mask_to_patches(mask))
    elif args.loss == 'cross entropy':
        criterion = torch.nn.BCELoss(reduction='mean')
        criterion = criterion.cuda() if args.cuda else criterion
    elif args.loss == 'dice + cross entropy':
        ce = torch.nn.BCELoss(reduction='mean')
        ce = ce.cuda() if args.cuda else ce
        criterion = lambda output_, mask_: ce(output_, mask_) + dice_loss(output_, mask_)
    else:
        raise Exception("The given loss function does not exist.")

    # Training
    if args.train:
        for epoch in range(args.epochs):
            model.train()
            train_loss = []
            train_f1 = []
            train_f1_patches = []
            for img, mask in train_loader:
                
                img = img.cuda().float() if args.cuda else img.float()
                mask = mask.cuda() if args.cuda else mask

                optimizer.zero_grad()

                # Adversarial attack
                if args.adversarial_bound != 0:
                    img.requires_grad = True
                    output = model(img)
                    loss = criterion(output, mask)
                    model.zero_grad()
                    loss.backward()
                    img = fgsm_update(img, output, mask, update_max_norm=args.adversarial_bound)

                output = model(img)
                loss = criterion(output, mask)
                loss.backward()
                optimizer.step()

                f1_score, f1_patches = get_score(output, mask), get_score_patches(output, mask)
                train_loss.append(loss.item())
                train_f1.append(f1_score)
                train_f1_patches.append(f1_patches)

            save_track(
                experiment_path,
                args,
                train_loss=sum(train_loss) / len(train_loss),
                train_f1=sum(train_f1) / len(train_f1),
                train_f1_patch=sum(train_f1_patches) / len(train_f1_patches),
                       )
            
            # Validation
            if args.validation_ratio:
                model.eval()
                val_loss = []
                val_f1 = []
                val_f1_patches = []
                with torch.no_grad():
                    for img, mask in val_loader:
                        
                        img = img.cuda().float() if args.cuda else img.float()
                        mask = mask.cuda() if args.cuda else mask

                        output = model(img)
                        loss = criterion(output, mask)
                        f1_score, f1_patches = get_score(output, mask), get_score_patches(output, mask)
                        val_loss.append(loss.item())
                        val_f1.append(f1_score)
                        val_f1_patches.append(f1_patches)
                
                # Logging
                val_loss_to_track = sum(val_loss) / len(val_loss)
                val_f1_to_track = sum(val_f1) / len(val_f1)
                val_f1_patches_to_track = sum(val_f1_patches) / len (val_f1_patches)
                print('Epoch : {} | Loss = {:.4f}, F1 Score = {:.4f}, F1 Patches Score: {:.4f}'.format(
                    epoch, val_loss_to_track, val_f1_to_track, val_f1_patches_to_track))
                save_track(
                    experiment_path,
                    args,
                    val_loss=val_loss_to_track,
                    val_f1=val_f1_to_track,
                    val_f1_patch=val_f1_patches_to_track,
                    )
                scheduler.step(val_loss_to_track)
            else:
                print("Epoch : {} | No validation".format(epoch))

            # Saving the weights
            if args.save_weights and epoch % 10 == 9:
                save_model(model, optimizer, experiment_path, args)

    # Testing
    if args.test:
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
