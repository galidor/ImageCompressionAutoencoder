from torch import nn
import torch
import time


class AutoencoderConv(nn.Module):
    def __init__(self, nf=64, im_depth=3, code_size=100):
        super(AutoencoderConv, self).__init__()
        # self.encoder = nn.Sequential(nn.ReplicationPad2d((1, 1, 1, 1)),
        #                              nn.Conv2d(im_depth, nf, 4, 2, 0, bias=False),
        #                              nn.ReLU(inplace=True),
        #                              nn.ReplicationPad2d((1, 1, 1, 1)),
        #                              nn.Conv2d(nf, nf * 2, 4, 2, 0, bias=False),
        #                              nn.BatchNorm2d(nf*2),
        #                              nn.ReLU(inplace=True),
        #                              nn.ReplicationPad2d((1, 1, 1, 1)),
        #                              nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=False),
        #                              nn.BatchNorm2d(nf * 4),
        #                              nn.ReLU(inplace=True),
        #                              nn.ReplicationPad2d((1, 1, 1, 1)),
        #                              nn.Conv2d(nf * 4, nf * 8, 4, 2, 0, bias=False),
        #                              nn.BatchNorm2d(nf * 8),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(nf * 8, code_size, 2, 1, 0),
        #                              nn.ReLU(inplace=True))
        #
        # self.decoder = nn.Sequential(nn.ConvTranspose2d(code_size, nf * 8, 2, 1, 0),
        #                              nn.BatchNorm2d(nf * 8),
        #                              nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                              nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1),
        #                              nn.BatchNorm2d(nf * 4),
        #                              nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                              nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1),
        #                              nn.BatchNorm2d(nf * 2),
        #                              nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                              nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1),
        #                              nn.BatchNorm2d(nf),
        #                              nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                              nn.ConvTranspose2d(nf, im_depth, 4, 2, 1),
        #                              nn.Tanh())
        self.encoder = nn.Sequential(nn.Conv2d(im_depth, nf, 8, 8, 0),
                                     nn.BatchNorm2d(nf),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(nf, code_size, 4, 4, 0),
                                     nn.BatchNorm2d(code_size),
                                     nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(code_size, nf, 4, 4, 0),
                                     nn.BatchNorm2d(nf),
                                     nn.LeakyReLU(inplace=True),
                                     nn.ConvTranspose2d(nf, im_depth, 8, 8, 0),
                                     nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        # x = RoundNoGrad.apply(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):



def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class RoundNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad):
        return grad