import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, batch_size):
        super(Generator, self).__init__()

        bn = None
        if batch_size == 1:
            bn = False # Instance Normalization
        else:
            bn = True # Batch Normalization

        # layer_1 [1x512x512] -> [64x256x256]
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

        # layer_2 [64*256*256]> [128*128*128]
        conv2 = [nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1)]
        if bn == True:
            conv2 += [nn.BatchNorm2d(128)]
        else:
            conv2 += [nn.InstanceNorm2d(128)]
        self.conv2 = nn.Sequential(*conv2)

        # layer_3 [128*128*128]> [256x64x64]
        conv3 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(128, 256, 4, 2, 1)]
        if bn == True:
            conv3 += [nn.BatchNorm2d(256)]
        else:
            conv3 += [nn.InstanceNorm2d(256)]
        self.conv3 = nn.Sequential(*conv3)

        # layer_4 [256*64*64]> [512x32x32]
        conv4 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(256, 512, 4, 2, 1)]
        if bn == True:
            conv4 += [nn.BatchNorm2d(512)]
        else:
            conv4 += [nn.InstanceNorm2d(512)]
        self.conv4 = nn.Sequential(*conv4)

        # layer_5 [512*32*32]> [1024x16x16]
        conv5 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(512, 1024, 4, 2, 1)]
        if bn == True:
            conv5 += [nn.BatchNorm2d(1024)]
        else:
            conv5 += [nn.InstanceNorm2d(1024)]
        self.conv5 = nn.Sequential(*conv5)

        # layer_6 [1024*16*16]> [2048x8x8]
        conv6 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(1024,2048, 4, 2, 1)]
        if bn == True:
            conv6 += [nn.BatchNorm2d(2048)]
        else:
            conv6 += [nn.InstanceNorm2d(2048)]
        self.conv6 = nn.Sequential(*conv6)

        # lyaer_7 [2048*8*8]> [2048x4x4]
        conv7 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(2048, 2048, 4, 2, 1)]
        if bn == True:
            conv7 += [nn.BatchNorm2d(2048)]
        else:
            conv7 += [nn.InstanceNorm2d(2048)]
        self.conv7 = nn.Sequential(*conv7)

        # layer_8 2048*4*4> [2048x2x2]
        conv8 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(2048,2048, 4, 2, 1)]
        if bn == True:
            conv8 += [nn.BatchNorm2d(2048)]
        else:
            conv8 += [nn.InstanceNorm2d(2048)]
        self.conv8 = nn.Sequential(*conv8)

        # layer_9 2048*2*2> [2048x1x1]
        conv9 = [nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(2048,2048, 4, 2, 1)]
        if bn == True:
            conv9 += [nn.BatchNorm2d(2048)]
        else:
            conv9 += [nn.InstanceNorm2d(2048)]
        self.conv9 = nn.Sequential(*conv9)

        # deconv_layer_9 [2048*1*1]> [2048x2x2]
        deconv9 = [nn.ReLU(),
                   nn.ConvTranspose2d(2048, 2048, 4, 2, 1)]
        if bn == True:
            deconv9 += [nn.BatchNorm2d(2048), nn.Dropout(0.5)]
        else:
            deconv9 += [nn.InstanceNorm2d(2048), nn.Dropout(0.5)]
        self.deconv9 = nn.Sequential(*deconv9)

        # deconv_layer_8 [(2048+2048)*2*2]> [2048x4x4]
        deconv8 = [nn.ReLU(),
                   nn.ConvTranspose2d(2048*2, 2048, 4, 2, 1)]
        if bn == True:
            deconv8 += [nn.BatchNorm2d(2048), nn.Dropout(0.5)]
        else:
            deconv8 += [nn.InstanceNorm2d(2048), nn.Dropout(0.5)]
        self.deconv8 = nn.Sequential(*deconv8)

        # deconv_layer_7 [(2048+2048)x4x4] -> [2048x8x8]
        deconv7 = [nn.ReLU(),
                   nn.ConvTranspose2d(2048 * 2, 2048, 4, 2, 1)]
        if bn == True:
            deconv7 += [nn.BatchNorm2d(2048), nn.Dropout(0.5)]
        else:
            deconv7 += [nn.InstanceNorm2d(2048), nn.Dropout(0.5)]
        self.deconv7 = nn.Sequential(*deconv7)

        # deconv_layer_6 [(2048+2048)x8x8] -> [1024x16x16]
        deconv6 = [nn.ReLU(),
                   nn.ConvTranspose2d(2048 * 2, 1024, 4, 2, 1)]
        if bn == True:
            deconv6 += [nn.BatchNorm2d(1024), nn.Dropout(0.5)]
        else:
            deconv6 += [nn.InstanceNorm2d(1024), nn.Dropout(0.5)]
        self.deconv6 = nn.Sequential(*deconv6)

        # deconv_layer_5 [(1024+1024)x16x16] -> [512x32x32]
        deconv5 = [nn.ReLU(),
                   nn.ConvTranspose2d(1024 * 2, 512, 4, 2, 1)]
        if bn == True:
            deconv5 += [nn.BatchNorm2d(512)]
        else:
            deconv5 += [nn.InstanceNorm2d(512)]
        self.deconv5 = nn.Sequential(*deconv5)

        # deconv-layer_4 [(512+512)x32x32] -> [256x64x64]
        deconv4 = [nn.ReLU(),
                   nn.ConvTranspose2d(512 * 2, 256, 4, 2, 1)]
        if bn == True:
            deconv4 += [nn.BatchNorm2d(256)]
        else:
            deconv4 += [nn.InstanceNorm2d(256)]
        self.deconv4 = nn.Sequential(*deconv4)

        # deconv_layer_3 [(256+256)x64x64] -> [128x128x128]
        deconv3 = [nn.ReLU(),
                   nn.ConvTranspose2d(256 * 2, 128, 4, 2, 1)]
        if bn == True:
            deconv3 += [nn.BatchNorm2d(128)]
        else:
            deconv3 += [nn.InstanceNorm2d(128)]
        self.deconv3 = nn.Sequential(*deconv3)

        # deconv_layer_2 [(128+128)x128x128] -> [64x256x256]
        deconv2 = [nn.ReLU(),
                   nn.ConvTranspose2d(128 * 2, 64, 4, 2, 1)]
        if bn == True:
            deconv2 += [nn.BatchNorm2d(64)]
        else:
            deconv2 += [nn.InstanceNorm2d(64)]
        self.deconv2 = nn.Sequential(*deconv2)

        # deconv_layer_1 [(64+64)x256x256] -> [1x512x512]
        self.deconv1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 2, 3, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, x):

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)
        c9 = self.conv9(c8)

        d8 = self.deconv9(c9)
        d8 = torch.cat((c8, d8),dim=1)
        d7 = self.deconv8(d8)
        d7 = torch.cat((c7, d7), dim=1)
        d6 = self.deconv7(d7)
        d6 = torch.cat((c6, d6), dim=1)
        d5 = self.deconv6(d6)
        d5 = torch.cat((c5, d5), dim=1)
        d4 = self.deconv5(d5)
        d4 = torch.cat((c4, d4), dim=1)
        d3 = self.deconv4(d4)
        d3 = torch.cat((c3, d3), dim=1)
        d2 = self.deconv3(d3)
        d2 = torch.cat((c2, d2), dim=1)
        d1 = self.deconv2(d2)
        d1 = torch.cat((c1, d1), dim=1)
        out = self.deconv1(d1)

        return out

class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super(Discriminator, self).__init__()

        bn = None
        if batch_size == 1:
            bn = False  # Instance Normalization
        else:
            bn = True  # Batch Normalization

        # [(1+1)x512x512] -> [64x256x256] -> [128x128x128]
        main = [nn.Conv2d(3*2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1)]
        if bn == True:
            main += [nn.BatchNorm2d(128)]
        else:
            main += [nn.InstanceNorm2d(128)]

        # [128*128*128]-> [256*64*64]
        main += [nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(128, 256, 4, 2, 1)]
        if bn == True:
            main += [nn.BatchNorm2d(256)]
        else:
            main += [nn.InstanceNorm2d(256)]

        # [256*64*64]-> [512*32*32]
        main += [nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(256, 512, 4, 2, 1)]
        if bn == True:
            main += [nn.BatchNorm2d(512)]
        else:
            main += [nn.InstanceNorm2d(512)]

        # [512*32*32] -> [1024x31x31] (Fully Convolutional)
        main += [nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(512, 1024, 4, 1, 1)]
        if bn == True:
            main += [nn.BatchNorm2d(1024)]
        else:
            main += [nn.InstanceNorm2d(1024)]

        # -> [1x30x30] (Fully Convolutional, PatchGAN)
        main += [nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(1024, 1, 4, 1, 1),
                  nn.Sigmoid()]

        self.main = nn.Sequential(*main)

    def forward(self, x1, x2): # One for Real, One for Fake
        out = torch.cat((x1, x2), dim=1)
        return self.main(out)
