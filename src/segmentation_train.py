import os
import shutil
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from torch.utils.data import DataLoader

from dataset_wrapper import DatasetWrapper
from augmentation import *
from utils import get_classes_list, get_data


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(data_file, classes_file):
    CLASSES = get_classes_list(classes_file)
    data_dict = get_data(data_file)

    # create folder for saving weights
    weights_save_dir = Path(data_dict['train_save_weights_dir'])
    if weights_save_dir.exists() and weights_save_dir.is_dir():
        shutil.rmtree(weights_save_dir)
    Path(weights_save_dir).mkdir(parents=True, exist_ok=True)

    # DATA_DIR = data_dict['data_dir']
    x_train_dir = data_dict['train_dir']
    y_train_dir = data_dict['trainannot_dir']
    x_valid_dir = data_dict['val_dir']
    y_valid_dir = data_dict['valannot_dir']

    # model init
    ENCODER = data_dict['encoder']
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = data_dict['activation']  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = data_dict['device']

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = DatasetWrapper(
        x_train_dir,
        y_train_dir,
        all_classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = DatasetWrapper(
        x_valid_dir,
        y_valid_dir,
        all_classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=int(data_dict['batch_size']), shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    loss = smp.utils.losses.JaccardLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.4),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=float(data_dict['learning_rate'])),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    for i in range(0, int(data_dict['epoch'])):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, weights_save_dir.joinpath('best_model.pth'))
            print('Model saved!')

        if i == 20:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default='data.txt', help='default %(default)s')
    parser.add_argument("--classes_file", default='classes.txt', help='default %(default)s')

    args = parser.parse_args()

    train(args.data_file, args.classes_file)

    # python3  src/segmentation_train.py --data_file data.txt --classes_file classes.txt
