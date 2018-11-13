import torch
import torchvision
from utils import dataset_loader
import models
from tensorboardX import SummaryWriter


writer = SummaryWriter('runs/shallow_MSE_baseline')

data_path = 'data/'

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

cifar = dataset_loader.CIFAR(data_path=data_path, transforms=transforms)
cifar_train, cifar_test = cifar.get_CIFAR()
cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=16, shuffle=True)
cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=1, shuffle=False)

autoencoder = models.AutoencoderConv(code_size=256).cuda()
autoencoder.apply(models.weights_init)
optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=0.0002, betas=(0.9, 0.999))
criterion_autoencoder = torch.nn.MSELoss(size_average=True)


def train(net_ae, optim_ae, crit_ae, epoch, dataloader, args=None, writer=None):
    loss_sum = 0.0
    for i, data in enumerate(dataloader, 0):
        net_ae.zero_grad()
        imgs, _ = data
        imgs = imgs.cuda()
        imgs_rec = net_ae(imgs)
        # feat = imgs.view(16*3, 32**2)
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

def evaluate(net_ae, dataloader):
    pass

for epoch in range(200):
    train(autoencoder, optimizer_autoencoder, criterion_autoencoder, epoch, cifar_train_loader, writer=writer)


evaluate(autoencoder, cifar_test_loader)