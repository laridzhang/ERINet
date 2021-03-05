# import torch
import torch.nn as nn
# import torch.nn.functional as functional

# from src.network import Conv2d, ConvTranspose2d
from src.unet_parts import DoubleConv, Down, Up, OutConv


class Model(nn.Module):
    def __init__(self, bilinear=True):
        super(Model, self).__init__()

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

    def forward(self, im_data):
        x1 = self.inc(im_data)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        visual_dict = dict()
        visual_dict['logits_map'] = logits

        return logits, visual_dict


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits


class ModelWithLoss(nn.Module):
    def __init__(self, bce_init_weights=None):
        super().__init__()
        self.features = Model()
        self.my_loss = None

        if bce_init_weights is not None:
            self.bce_loss_function = nn.BCEWithLogitsLoss(weight=bce_init_weights.cuda())
        else:
            self.bce_loss_function = nn.BCEWithLogitsLoss()

    @property
    def loss(self):
        return self.my_loss

    def forward(self, im_data, ground_truth=None):
        estimate, visual_dict = self.features(im_data.cuda())

        if self.training:
            self.my_loss, loss_dict = self.build_loss(ground_truth.cuda(), estimate)
        else:
            loss_dict = None

        return estimate, loss_dict, visual_dict

    def build_loss(self, ground_truth, estimate):
        if ground_truth.shape != estimate.shape:
            raise Exception('ground truth shape and estimate shape mismatch')

        this_loss = self.bce_loss_function(estimate, ground_truth)

        loss_dict = dict()
        loss_dict['bce'] = this_loss

        return this_loss, loss_dict