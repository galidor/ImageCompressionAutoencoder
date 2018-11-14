import torch
import torchvision
from utils import dataset_loader, image_transformations
import models
from tensorboardX import SummaryWriter
import os.path as osp
import argparse


###################################
# TODO:
# parser
# summary_writer
# losses: L1 and Adv
# evaluating function - test data
# baseline experiments: deep baseline on 512 feats, linear model - 512 feats, 200 epochs
####################################


def parse():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('Required Arguments')
    optional = parser.add_argument_group('Optional Arguments')
    required.add_argument('--data_path', type=str, required=True, help='Specifies a path to directory containing'
                                                                       ' CIFAR dataset, or where you wish to download')
    required.add_argument('--experiment_name', type=str, required=True, help='Name of your experiment for future'
                                                                             'reference in TensorBoard')

    optional.add_argument('--batch_size', type=int, default=64, help='Batch size')
    optional.add_argument('--optim_step', type=int, default=20, help='Number of epochs for learning rate decrease')
    optional.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    optional.add_argument('--normalize', type=str, help='user_defined|imagenet')
    optional.add_argument('--download_dataset', action='store_true', help='Use it if you want to download MNIST to'
                                                                          ' DATA_PATH folder')
    optional.add_argument('--model_path', type=str, default='/', help='Path where your network will be'
                                                                      ' stored')
    optional.add_argument('--test', action='store_true', help='Test mode only')
    optional.add_argument('--epochs', type=int, default=50, help='Total number of epochs')
    optional.add_argument('--nf', type=int, default=64)
    optional.add_argument('--ndf', type=int, default=128)
    optional.add_argument('--code_length', type=int, default=256, help='Desired compressed image size in bytes')

    args = parser.parse_args()

    return args


def train(net_ae, optim_ae, crit_ae, epoch, dataloader, args=None, writer=None):
    loss_sum = 0.0
    for i, data in enumerate(dataloader, 0):
        net_ae.zero_grad()
        imgs, _ = data
        imgs = imgs.cuda()
        imgs_rec = net_ae(imgs)
        # feat = imgs.view(16, 3*32**2)
        # feat_rec = imgs_rec.view(16*3, 32**2)
        # loss_ae = crit_ae(torch.mm(feat_rec, feat_rec.t()), torch.mm(feat, feat.t()))
        loss_ae = crit_ae(imgs_rec, imgs)
        loss_sum += loss_ae.item()
        loss_ae.backward()
        optim_ae.step()
        # print(i, loss_ae)
        if i % round(20000.0/16.0) == 0:
            imgs_rec = imgs_rec.clamp_(0, 1)
            im2show = torchvision.utils.make_grid(torch.cat((imgs[0], imgs_rec[0]), dim=0), nrow=2).view(2, 3, 32, 32)
            writer.add_image('Images/epoch{}-{}'.format(epoch+1, i), im2show)
    writer.add_scalar('MSE Loss', loss_sum, epoch+1)

def evaluate(net_ae, dataloader, epoch, args):
    pass


if __name__ == '__main__':

    opt = parse()

    writer = SummaryWriter(osp.join('runs', '{}'.format(opt.experiment_name)))

    transforms = image_transformations.get_transform(normalize=opt.normalize)

    cifar = dataset_loader.CIFAR(data_path=opt.data_path, transforms=transforms)
    cifar_train, cifar_test = cifar.get_CIFAR()
    cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=opt.batch_size, shuffle=True)
    cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=1, shuffle=False)

    autoencoder = models.AutoencoderConv(code_size=256).cuda()
    autoencoder.apply(models.weights_init)
    optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=0.0002, betas=(0.9, 0.999))
    criterion_autoencoder = torch.nn.MSELoss(size_average=True)

    for epoch in range(opt.epochs):
        train(autoencoder, optimizer_autoencoder, criterion_autoencoder, epoch, cifar_train_loader, writer=writer)
        evaluate(autoencoder, cifar_test_loader, epoch, opt)