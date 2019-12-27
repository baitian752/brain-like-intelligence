import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from network import *
import pytorch_ssim as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas


parser = argparse.ArgumentParser(description='Easy Implementation of Pix2Pix')

"""
Task                RawSize     Batch Size      Epochs      EpochsInPaper       RandomCrop&Mirroring    #ofSamples      #ofSamples(Validation)
Map-Aerial          600*600     1               100         200                 Yes                     1096            1098
Edges-Bag           256*256     4               15          15                  No                      138567          200
Semantic-Photo      256*256     1               100         200                 Yes                     2975            500
"""

# Task
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders train, val, etc)')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')

# Pre-processing
parser.add_argument('--no_resize_or_crop', action='store_true', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

# Hyper-parameters
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
parser.add_argument('--lambda_A', type=float, default=100.0)

# misc
parser.add_argument('--model_path', type=str, default='./models')  # Model Tmp Save
parser.add_argument('--sample_path', type=str, default='./results')  # Results
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=2)

##### Helper Functions for Data Loading & Pre-processing
class ImageFolder(data.Dataset):
    def __init__(self, opt):
        # os.listdir Function gives all lists of directory
        self.root = opt.dataroot
        self.no_resize_or_crop = opt.no_resize_or_crop
        self.no_flip = opt.no_flip
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])
        self.dir_AB = os.path.join(opt.dataroot, 'train')  # ./maps/train
        self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB)))

    def __getitem__(self, index):
        AB_path = self.image_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        if(not self.no_resize_or_crop):
            AB = AB.resize((542 * 2, 542), Image.BICUBIC)
            AB = self.transform(AB)

            w = int(AB.size(2) / 2)
            h = AB.size(1)
            w_offset = random.randint(0, max(0, w - 512 - 1))
            h_offset = random.randint(0, max(0, h - 512 - 1))

            A = AB[:, h_offset:h_offset + 512, w_offset:w_offset + 512]
            B = AB[:, h_offset:h_offset + 512, w + w_offset:w + w_offset + 512]
        else:
            AB = self.transform(AB)
            w_total = AB.size(2)
            w = int(w_total / 2)

            A = AB[:, :512, :512]
            B = AB[:, :512, w:w + 512]

        if (not self.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.image_paths)

##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

##### Helper Functions for GAN Loss (4D Loss Comparison)
def GAN_Loss(input, target, criterion):
    if target == True:
        tmp_tensor = torch.FloatTensor(input.size()).fill_(1.0)
        labels = Variable(tmp_tensor, requires_grad=False)
    else:
        tmp_tensor = torch.FloatTensor(input.size()).fill_(0.0)
        labels = Variable(tmp_tensor, requires_grad=False)

    if torch.cuda.is_available():
        labels = labels.cuda()

    return criterion(input, labels)

######################### Main Function
def main():
    # Pre-settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    dataset = ImageFolder(args)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batchSize,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Networks
    generator = Generator(args.batchSize)
    discriminator = Discriminator(args.batchSize)

    # Losses
    criterionGAN = nn.BCELoss()
    #criterionL1 = nn.L1Loss()

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), args.lr, [args.beta1, args.beta2])
    d_optimizer = optim.Adam(discriminator.parameters(), args.lr, [args.beta1, args.beta2])

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    """Train generator and discriminator."""
    total_step = len(data_loader) # For Print Log
    # rewrite the lossfunction : Loss_D_real removed, Loss_G_GAN removed.
    # index a list for plt
    lst_epoch = []
    lst_loss_D_real = []
    lst_loss_D_fake =[]
    lst_loss_D = []
    lst_loss_G_GAN = [0,0,0,0,0,0]
    lst_loss_G_ssim = []
    lst_loss_G = []
    for epoch in range(args.num_epochs):
        for i, sample in enumerate(data_loader):

            AtoB = args.which_direction == 'AtoB'
            input_A = sample['A' if AtoB else 'B']
            input_B = sample['B' if AtoB else 'A']

            # ===================== Train D =====================#
            discriminator.zero_grad()

            real_A = to_variable(input_A)
            fake_B = generator(real_A)
            real_B = to_variable(input_B)

            # d_optimizer.zero_grad()

            pred_fake = discriminator(real_A, fake_B)
            loss_D_fake = GAN_Loss(pred_fake, False, criterionGAN)

            pred_real = discriminator(real_A, real_B)
            loss_D_real = GAN_Loss(pred_real, True, criterionGAN)

            # Combined loss
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            #loss_D = loss_d_fake
            loss_D.backward(retain_graph=True)
            d_optimizer.step()

            # ===================== Train G =====================#
            generator.zero_grad()

            pred_fake = discriminator(real_A, fake_B)
            loss_G_GAN = GAN_Loss(pred_fake, True, criterionGAN)

            ssim_loss = ss.SSIM(window_size=11)
            loss_G_ssim = 1 - ssim_loss(fake_B, real_B)

            loss_G = loss_G_GAN + loss_G_ssim * args.lambda_A

            loss_G.backward()
            g_optimizer.step()

            # print the log info
            if (i + 1) % args.log_step == 0:
                print('Epoch [%d/%d], BatchStep[%d/%d], D_Real_loss: %.4f, D_Fake_loss: %.4f, G_loss: %.4f, G_ssim_loss: %.4f' % (epoch + 1, args.num_epochs, i + 1, total_step, loss_D_real.item(), loss_D_fake.item(), loss_G_GAN.item(), loss_G_ssim.item()))
                lst_loss_D_real.append(loss_D_real.item())
                lst_loss_D_fake.append(loss_D_fake.item())
                lst_loss_G_GAN.append(loss_G_GAN.item())
                lst_loss_D.append(loss_D.item())
                lst_loss_G_ssim.append(loss_G_ssim.item())
                lst_loss_G.append(loss_G.item())

            # j = len(lst_loss_G_GAN) -1
           #save the sampled images
            if (((epoch+1)>300)&((epoch+1) % 20 == 0)):
                res = torch.cat((torch.cat((real_A, fake_B), dim=3), real_B), dim=3)
                torchvision.utils.save_image(denorm(res.data), os.path.join(args.sample_path, 'Generated-%d-%d.png' % (epoch + 1, i + 1)))
            # save the model parameters for each epoch
            # if (epoch+1) % 100 == 0:
            #     g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
            #     torch.save(generator.state_dict(), g_path)
            #j = len(lst_loss_G_GAN) -1
        if (((epoch+1)>300)&((epoch+1) % 20 ==0)):
            g_path = os.path.join(args.model_path, 'Generator-%d.pkl'%(epoch+1))
            torch.save(generator.state_dict(),g_path)
    # plt.figure(211)
    # p1, = plt.plot(lst_epoch, lst_loss_D_real, 'red')
    # p2, = plt.plot(lst_epoch, lst_loss_D_fake, 'black')
    # p3, = plt.plot(lst_epoch, lst_loss_G_GAN, 'yellow')
    # p4, = plt.plot(lst_epoch, lst_loss_L1, 'blue')
    # plt.legend(handles=[p1, p2, p3, p4, ], labels=['loss_D_real', 'loss_D_fake', 'loss_G_Gan', 'loss_L1', ],
    #             loc='upper right')
    #
    # plt.fiture(212))
    # p5, = plt.plot(lst_epoch,lst_loss_D,'r-')
    # p6, = plt.plot(lst_epoch,lst_loss_G_l1,'b--')
    # plt.legend(handles=[p5],labels=['loss_G'],loc='upper right')
    # plt.savefig('./')
    # plt.show()

    # save the loss
    loss_D_real_df = pandas.DataFrame(lst_loss_D_real)
    loss_D_fake_df = pandas.DataFrame(lst_loss_D_fake)
    loss_G_GAN_df = pandas.DataFrame(lst_loss_G_GAN)
    loss_G_ssim_df = pandas.DataFrame(lst_loss_G_ssim)


    loss_D_real_df.to_excel('./loss_D_real.xls')
    loss_D_fake_df.to_excel('./loss_D_fake.xls')
    loss_G_GAN_df.to_excel('./loss_G_Gan.xls')
    loss_G_ssim_df.to_excel('./loss_G_ssim.xls')


if __name__ == "__main__":
    main()
