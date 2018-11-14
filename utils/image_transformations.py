from torchvision import transforms


def get_transform(resize=None, normalize=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transforms_list = []
    if resize is not None:
        transforms_list.append(transforms.Resize(resize))
    transforms_list.append(transforms.ToTensor())
    if normalize is not None:
        if normalize == 'imagenet':
            transforms_list.append(transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                                        std=[1/0.229, 1/0.224, 1/0.225]))
        if normalize == 'user_defined':
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
        if normalize == 'denormalize':
            transforms_list.append(denormalize(mean=mean, std=std))
    return transforms.Compose(transforms_list)


def denormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    return transforms.Normalize(mean=[-mean[i]/std[i] for i in range(3)],
                                std=[1.0/std[i] for i in range(3)])
