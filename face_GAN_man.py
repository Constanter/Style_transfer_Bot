from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def face_gan_man(img):
    ###### Definition of variables ######
    batchSize=1
    input_nc=3
    output_nc=3
    size= 224
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    n_cpu=8
    generator_B2A='/app/netG_B2A10.pth'
    
    # Load image
    img = Image.open(img)
    width, height = img.size
    ratio = width / height    


    # Network
    netG_B2A = Generator(output_nc, input_nc)
    netG_B2A.to(device)

    # Load state dicts
    netG_B2A.load_state_dict(torch.load(generator_B2A,map_location=torch.device(device)))

    # Set model's test mode
    netG_B2A.eval()

    # Dataset loader
    transforms_ = [ transforms.Resize([size,size]),
                   transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(img, transforms_=transforms_,),
                            batch_size=batchSize, shuffle=False, num_workers=n_cpu)

    for batch in dataloader:
        # Set model input
        real_B = batch.to(device)

        # Generate output
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    return fake_A, ratio

class ImageDataset(Dataset):
    def __init__(self, img, transforms_=None,):
        self.transform = transforms.Compose(transforms_)
        self.files_A = img

    def __getitem__(self, index):
        item_A = self.transform(self.files_A)
        return  item_A

    def __len__(self):
        return 1

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
