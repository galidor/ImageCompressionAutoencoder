import torchvision


class CIFAR():
    def __init__(self, data_path, transforms, download=False):
        self.cifar_train = torchvision.datasets.CIFAR10(data_path, train=True, transform=transforms,
                                                        target_transform=None, download=download)
        self.cifar_test = torchvision.datasets.CIFAR10(data_path, train=False, transform=transforms,
                                                        target_transform=None, download=download)

    def get_CIFAR(self):
        return self.cifar_train, self.cifar_test


class MNIST():
    def __init__(self, data_path, transforms, download=False):
        self.mnist_train = torchvision.datasets.MNIST(data_path, train=True, transform=transforms,
                                                      target_transform=None, download=download)
        self.mnist_test = torchvision.datasets.MNIST(data_path, train=False, transform=transforms,
                                                     target_transform=None, download=download)

    def get_MNIST(self):
        return self.mnist_train, self.mnist_train