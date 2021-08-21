import torch.nn as nn

from src.models import Model


class ERINet(nn.Module):
    def __init__(self, bce_init_weights=None):
        super().__init__()
        self.features = Model()
        self.my_loss = None

        if bce_init_weights is not None:
            self.bce_loss_function = nn.BCELoss(weight=bce_init_weights.cuda())
        else:
            self.bce_loss_function = nn.BCELoss()

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
        loss_dict['label'] = this_loss

        return this_loss, loss_dict
