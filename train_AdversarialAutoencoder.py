import torch
import torchvision
from utils import dataset_loader, image_transformations
import models
from tensorboardX import SummaryWriter
import os.path as osp
import argparse


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
    optional.add_argument('--max_iter', type=int, default=2e5, help='Total number of iterations')
    optional.add_argument('--nf', type=int, default=256)
    optional.add_argument('--ndf', type=int, default=128)
    optional.add_argument('--code_length', type=int, default=256, help='Desired compressed image size in bytes')
    optional.add_argument('--clipping_threshold', type=float, default=0.01)

    args = parser.parse_args()

    return args


def train(max_iter, net_ae, net_disc, optim_ae, optim_disc, crit_ae, crit_disc, dataloader, args=None, writer=None):
    loss_ae_l2_sum = 0.0
    loss_ae_adv_sum = 0.0
    loss_disc_sum = 0.0
    data = iter(dataloader)
    for i in range(1, int(max_iter + 1)):
        # Wasserstein-GAN-like discriminator training
        # fake_imgs is used for images obtained from autoencoder
        for _ in range(5):
            net_disc.zero_grad()
            net_ae.zero_grad()
            try:
                true_imgs, _ = next(data)
            except StopIteration:
                data = iter(dataloader)
                true_imgs, _ = next(data)
            true_imgs = true_imgs.cuda()
            fake_imgs = net_ae(true_imgs)
            pred_true = net_disc(true_imgs)
            pred_fake = net_disc(fake_imgs)
            loss_disc = -(torch.mean(pred_true) - torch.mean(pred_fake))
            loss_disc_sum += loss_disc.item() / 5.0
            loss_disc.backward()
            optim_disc.step()
            for p in net_disc.parameters():
                p.data.clamp_(-args.clipping_threshold, args.clipping_threshold)

        net_ae.zero_grad()
        net_disc.zero_grad()
        try:
            imgs, _ = next(data)
        except StopIteration:
            data = iter(dataloader)
            imgs, _ = next(data)
        imgs = imgs.cuda()
        imgs_rec = net_ae(imgs)
        pred = net_disc(imgs_rec)
        loss_ae_l2 = crit_ae(imgs_rec, imgs)
        loss_ae_adv = -torch.mean(pred)
        loss_ae = loss_ae_l2 + loss_ae_adv
        loss_ae_l2_sum += loss_ae_l2.item()
        loss_ae_adv_sum += loss_ae_adv.item()
        loss_ae.backward()
        optim_ae.step()
        if i % 1000 == 0:
            imgs_rec = imgs_rec.clamp_(0, 1)
            im2show = torchvision.utils.make_grid(torch.cat((imgs[0], imgs_rec[0]), dim=0), nrow=2).view(2, 3, 32, 32)
            writer.add_image('Images/iter{}'.format(i), im2show)
            writer.add_scalar('MSE AE Loss', loss_ae_l2_sum/1000.0, i)
            writer.add_scalar('Adv AE Loss', loss_ae_adv_sum/1000.0, i)
            writer.add_scalar('Disc Loss', loss_disc_sum/1000.0, i)
            loss_ae_l2_sum = 0.0
            loss_ae_adv_sum = 0.0
            loss_disc_sum = 0.0

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
    criterion_autoencoder = torch.nn.MSELoss(reduction='elementwise_mean').cuda()

    discriminator = models.Discriminator().cuda()
    discriminator.apply(models.weights_init)
    optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)
    criterion_discriminator = torch.nn.BCELoss(reduction='elementwise_mean').cuda()

    train(opt.max_iter, autoencoder, discriminator, optimizer_autoencoder, optimizer_discriminator,
          criterion_autoencoder, criterion_discriminator, cifar_train_loader, writer=writer, args=opt)
    evaluate(autoencoder, cifar_test_loader, 1, opt)