# CS-433 Project 2: Road Segmentation Using U-net

This is a repository for the second project of the Machine Learning (CS-433) course at EPFL by Nicolas Flammarison and Martin Jaggi, autumn 2021. The objective of this project is to train a classification model to segment the roads in aerial images. The train dataset consists of 100 pairs of 400x400 pixels aerial images from Google Maps and their ground-truth images, in which the road pixels are labeled 1 and background pixels with 0. One of the train images and its ground-truth is presented below:

| ![](images/img_033.png) | ![](images/gt_033.png) |

### Contributors
- Ali Garjani
- Sepehr Mousavi
- Bruno Ploumhans

## Setup
This setup requires a Default Unix Environment with an installed Pyhton 3.7 or Python 3.8. Use the following command to install all the required libraries:
```bash
pip install -r requirements.txt
```

## Regenerating the final submission

The results of the final AICrowd submission ([#168633](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/168633)) is in `./experiments/Final_Submission`. To regenerate the same result, run the following command:

```bash
python main.py --experiment "Final_Submission_Gen" --path ./dataset --model "UNet" --validation_ratio 0.2 --cuda 1 --loss 'dice' --epoch 70 --save_weights True
```
The predictions and the submission file will be saved in this directory: `./experiments/Final_Submission_Gen`.

## Training and testing

For training a new model or getting the predictions of a model on the test dataset, call `main.py`. The following table shows the arguments that control the process.

| Flag                  | Type  | Default   | Description                               | 
| --------------------- | ----- | --------- | ----------------------------------------- |
| path                  | str   | './dataset' | Dataset path.                             |
| model                 | str   | 'UNet'    | The selected model among: 'UNet', 'WNet0404', 'WNet0402'.|
| validation_ratio      | float | None      | The ratio of the validation dataset.      |
| rotate                | bool  | True      | Train over rotated images.                |
| flip                  | bool  | True      | Random flip in training.                  |
| resize                | int   | None      | The resize value for test images.         |
| random_crops          | int   | 0         | Number of random erasings per image.      |
| batch_size            | int   | 8         | The batch size for SGD.                   |
| cuda                  | int   | 1         | Uses GPU if 1, CPU if 0.                  |
| lr                    | float | 1e-03     | The initial optimization learning rate.   |
| weight_path           | str   | None      | Path of the initial weights.              |
| experiment_name       | str   | NotSpec   | The name of the experiment.               |
| train                 | bool  | True      | Trains the model if True.                 |
| test                  | bool  | True      | Tests the model if True.                  |
| epochs                | int   | 100       | Number of epochs.                         |
| save_weights          | bool  | False     | Save weights after every epoch if True.   |
| loss                  | str   | 'dice'    | Loss function among: 'dice', 'cross entropy', 'dice + cross entropy'|
| adversarial_bound     | float | 0         | The bound (epsilon) of the adversarial training. No adversarial attack if zero.|

The results will be saved in a folder with the given experiment name in `./experiments`. If you are using an external dataset, or are storing the dataset in a different directory, the directory should be given by `--path /.../dataset`. If you don't have access to a GPU, set `--cuda 0`.

### Examples

To train a new model based on `0.8` training dataset, validating on `0.2` of the dataset, and then testing it on the testing dataset, without adding the rotations, with `10` random crops per training image, and for `100` epochs, run:
```bash
python main.py --path ./dataset --experiment "Train_And_Test" --validation_ratio 0.2 --rotate False --random_crops 10 --epoch 100
```
- Set `--save_weights True` to save the weights after every epoch.
- To start from initial weights, set `--weights_path /.../weights.pt`.
- Set `--test False` to not perform the test.


To get the test results based on a pre-trained model, run:
```bash
python main.py --path ./dataset --weights_path "/.../weights.pt" --experiment "Test_Pretrained" --train False
```
