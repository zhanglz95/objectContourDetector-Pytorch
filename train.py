import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from torch import optim
from tqdm import tqdm

from eval import eval_net
from model import OCDNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from utils.losses import xxloss
from torch.utils.data import DataLoader, random_split

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              dataPath="./data"):
    dataset = BasicDataset(os.path.join(dataPath, 'imgs'), os.path.join(dataPath, 'edges'), img_scale, mask_suffix='')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_worker=9, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_worker=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'lr_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Start training:
        Epoch:          {epochs}
        Batch size:     {batch_size}
        Learning rate:  {lr}
        Training size:  {n_train}
        Validation size:{n_val}
        Checkpoints:    {save_cp}
        Device:         {device.type}
        Images scaling: {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = weighted_cross_entropy_loss

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            imgs = batch['image']
            true_edges = batch['edge']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_edges = true_edges.to(device=device, dtype=torch.float32)

            edges_pred = net(imgs)
            loss = criterion(edges_pred, true_edges)
            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)

            pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            pbar.update(imgs.shape[0])
            global_step += 1
            if global_step % (n_train // (10 * batch_size)) == 0:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weight/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                val_score = eval_net(net, val_loader, device)
                scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizer.param_group[0]['lr'], global_step)

                logging.info('Validation Dice Coeff: {}'.format(val_score))
                writer.add_scalar('Dice/test', val_score, global_step)

                writer.add_images('images', imgs. global_step)
                writer.add_images('masks/true', true_masks, global_step)
                writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Create checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        filename='log.txt',
                        filemode='a',
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = OCDNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                f'\t{net.n_channels} input channels\n'
                f'\t{net.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=atgs.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)