import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Encoder(nn.Module):

    def __init__(self):

        super(Encoder , self).__init__()

        self.ConvNet = nn.Sequential(
                        # input: N x channels_img x 64 x 64
                        nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), # batch_size , 16 , 32 , 32
                        nn.LeakyReLU(0.2),
                        # _block(in_channels, out_channels, kernel_size, stride, padding)
                        self.__block(16, 16 * 2, 4, 2, 1),                    # batch_size , 32 , 16 , 16
                        self.__block(16 * 2, 16 * 4, 4, 2, 1),                # batch_size , 64 ,  8 ,  8
                        self.__block(16 * 4, 16 * 8, 4, 2, 1),                # batch_size ,128 ,  4 ,  4
                        nn.Conv2d(16 * 8, 256, kernel_size=4, stride=2, padding=1),  # batch_size , 256 ,  2 ,  2
                        nn.LeakyReLU(0.2),
                        nn.Flatten(start_dim = 1 , end_dim = -1),
                        nn.Linear(2*2*256 , 256),
                        nn.Tanh()
                        )


    def __block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(out_channels, affine=True),
                    nn.LeakyReLU(0.2),
                    )


    def forward(self, x):
        code = self.ConvNet(x)
        mean , log_var_square = code[:,:128] , code[:,128:]
        return mean , log_var_square


class Decoder(nn.Module):

    def __init__(self):

        super(Decoder, self).__init__()

        self.MLPNet = nn.Sequential(
                        nn.Linear(128 , 4*4*256),
                        nn.ReLU()
                        )

        self.ConvNet = nn.Sequential(
                        self.__block(16 * 16, 16 * 8, 4, 2, 1),  # img: 8x8
                        self.__block(16 * 8, 16 * 4, 4, 2, 1),  # img: 16x16
                        self.__block(16 * 4, 16 * 2, 4, 2, 1),  # img: 32x32
                        nn.ConvTranspose2d(16 * 2, 1, kernel_size=4, stride=2, padding=1),
                        nn.Tanh(),
                        )


    def __block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    )


    def forward(self , x):
        mlp = self.MLPNet(x)
        mlp = mlp.view(-1 , 256 ,  4 , 4)
        return self.ConvNet(mlp)