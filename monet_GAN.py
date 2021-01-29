import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
#############################################
# size of the batches"
batchSize = 1
# number of cpu threads to use during batch generation
n_cpu = 8
# size of image height
size = 128
# number of image channels
channels = 3
# interval between saving model checkpoints
checkpoint_interval = -1
# number of residual blocks in generator
n_residual_blocks = 9
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class ImageDataset(Dataset):
    def __init__(self, img, transforms_=None,):
        self.transform = transforms.Compose(transforms_)
        self.files_A = img

    def __getitem__(self, index):
        item_A = self.transform(self.files_A)
        return  item_A

    def __len__(self):
        return 1

def monet_gan(image):
  input_shape = (channels, size, size)
  device= 'cuda' if torch.cuda.is_available() else 'cpu'
  
  netG_B2A = GeneratorResNet(input_shape, n_residual_blocks)
  netG_B2A = netG_B2A.to(device)
  netG_B2A.load_state_dict(torch.load("/app/G_BA.pth",map_location=torch.device(device)) )
  netG_B2A.eval()

  img = Image.open(image)
  width, height = img.size
  ratio = width / height

  transforms_ = [ transforms.Resize([size,size]),
                    transforms.ToTensor(),
                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
                      
  dataloader = DataLoader(ImageDataset(img, transforms_=transforms_,),
                              batch_size=batchSize, shuffle=False, num_workers=n_cpu)

  for batch in dataloader:
    real_A = batch.to(device)
    fake_A = 0.5*(netG_B2A(real_A).data + 1.0)

  return fake_A, ratio   
