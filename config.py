import re
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from models.acgan import Generator, Discriminator
from utils.utils import *

root = 'data'
image_size = 64
z_dim = 100
conv_dim = 64
channels = 3
n_classes = 12
g_lr = 0.0001
d_lr = 0.0004
beta1 = 0.5
beta2 = 0.999
batch_size = 128
ctx = 0

g_num = 8
log_path = './logs'
log_step = 10
sample_step = 2000
sample_path = './samples'

logger = SummaryWriter(log_path)

# 加载数据集
def data_iter():
    dst = ImageFolder(root)
    transformer = transforms.Compose([
        # transforms.RandomAutocontrast(p=0.5),
        # transforms.RandomResizedCrop((256, 256)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dst.transform = transformer

    # 加载数据集
    data_loader = DataLoader(
        dst,
        batch_size=batch_size,
        shuffle=True,
    )
    return data_loader


def create_model():
    G = Generator(image_size=image_size, z_dim=z_dim,
              conv_dim=conv_dim, channels=channels, n_classes=n_classes)
    D = Discriminator(image_size=image_size, conv_dim=conv_dim,
                    channels=channels, n_classes=n_classes)
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D


