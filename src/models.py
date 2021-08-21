import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.network import Conv2d, ConvTranspose2d
from src.unet_parts import DoubleConv, Down, Up, OutConv
from src.utils import dilate_mask_tensor


class Model(nn.Module):
    def __init__(self, bn=False):
        super(Model, self).__init__()

        # mask
        bilinear = True
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

        self.vgg16 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2),
                                   Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2),
                                   Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2),
                                   Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn))

        self.classifier = nn.Sequential(Conv2d(1024, 256, 1, same_padding=True, bn=bn),
                                        ConvTranspose2d(256, 128, 2, stride=2, padding=0, bn=bn),
                                        Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                        Conv2d(128, 2, 1, same_padding=True, bn=bn))

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        # self.softmax = nn.Softmax2d()

    def forward(self, im_data):
        with torch.no_grad():
            x1 = self.inc(im_data)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            mask = self.outc(x)
            mask = torch.sigmoid(mask)
            mask = (mask > 0.5).to(torch.float)

            dilated_mask = dilate_mask_tensor(mask, 10)
            # dilated_mask = mask

        x_prior = self.vgg16(im_data)

        feature, another_feature = torch.chunk(x_prior, 2, dim=0)
        concat_feature = torch.cat((feature, another_feature), dim=1)

        # classifier
        x_class_map = self.classifier(concat_feature)

        with torch.no_grad():
            dilated_mask = torch.cat(torch.chunk(dilated_mask, 2, dim=0), dim=1)
            dilated_mask = functional.interpolate(dilated_mask, size=(x_class_map.shape[2], x_class_map.shape[3]), mode='bilinear')

        x_class_map = x_class_map * dilated_mask
        x_pool = self.average_pool(x_class_map)
        label = self.softmax(x_pool.reshape(x_pool.shape[0], -1))

        # x_prior = self.softmax(x_prior)

        visual_dict = dict()
        visual_dict['class_map'] = x_class_map

        return label, visual_dict
