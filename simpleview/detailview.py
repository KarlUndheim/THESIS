"""
@File           : detailview.py
@Author         : Gefei Kong
@Time:          : 22.11.2023 16:50
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
detailview with a detail view around (1-1.5m height of the tree).
The additional info. of the tree is also added into the net
"""


import torch
import torch.nn as nn

from functools import partial
from torch.nn import functional as F

from torchvision.models import resnet18,resnet34, resnet50, resnext50_32x4d, convnext_tiny

try:
    from simpleview_pytorch.resnet import resnet18_4
except:
    from TLSpecies_SimpleView.simpleview_pytorch.resnet import resnet18_4


backbones = {
    "resnet18_4": resnet18_4,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnext50_32x4d": resnext50_32x4d,
    "convnext_tiny": convnext_tiny,

}


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class DetailView_DfModel(nn.Module):

    def __init__(self, num_views: int, num_classes: int, num_feats: int=1, img_size:int=256,
                 selected_model: str = "resnet18_4", imgnet_weight: bool=False,
                 drop_rate=0.):
        super().__init__()

        self.num_views = num_views
        self.num_feats = num_feats
        self.dropout_ratio = drop_rate

        if "convnext" in selected_model:
            backbone = backbones[selected_model](pretrained=True)
            # backbone.features[0][0].in_channels = 1
            backbone.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
            z_dim = backbone.classifier[2].in_features
            # backbone.classifier = nn.Identity()
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            backbone.classifier = nn.Sequential(
                norm_layer(z_dim), nn.Flatten(1))

            # self.backbone = backbone
            #
            # self.fc = nn.Sequential(
            #     nn.Linear(in_features=z_dim * num_views,
            #               out_features=num_classes)
            # )

        else:
            if selected_model == "resnet18_4": # or (not imgnet_weight):
                backbone = backbones[selected_model]()# resnet18_4()
            else:
                backbone = backbones[selected_model](pretrained=imgnet_weight)
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            z_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

        # extract features for view images. ###########
        self.backbone_whole = backbone # consider the whole tree
        self.backbone_section = backbone # consider the 1-2.5m (a section) of a tree.

        # extract features for other feataures of this tree ###########
        self.feats_way = nn.Sequential(
            nn.Linear(num_feats, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim)
        )

        # fc (classifier) ###########
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=z_dim * (num_views + 1),
                out_features=256
            ),
            nn.ReLU(),
            nn.Linear(256, out_features=num_classes)
        )

        # dropout
        self.drop = nn.Dropout(p=self.dropout_ratio)



    def forward(self, x: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # prepare data
        b, v, c, h, w = x.shape
        x_wholeview = x[:, :6, :, :, :]
        x_secview   = x[:, 6:, :, :, :] if v > 6 else None
        # images (consider all images) part ########################
        v_whole = 6
        x_wholeview = x_wholeview.reshape(b * v_whole, c, h, w)
        z_wholeview = self.backbone_whole(x_wholeview)
        z_wholeview = z_wholeview.reshape(b, v_whole, -1)
        # Concat fuse
        z_wholeview = z_wholeview.reshape(b, -1)
        # print("z_wholeview: ", z_wholeview.size())

        # section_x (consider 1-2.5m of a tree) part ################
        if x_secview is not None:
            v_sec = v - 6
            x_secview = x_secview.reshape(b * v_sec, c, h, w)
            z_secview = self.backbone_section(x_secview)
            z_secview = z_secview.reshape(b, v_sec, -1)
            # Concat fuse
            z_secview = z_secview.reshape(b, -1)
            # print("z_secview: ", z_secview.size())


        if feats is not None:
            # other features part ########################
            feats_z = self.feats_way(feats.view(-1,self.num_feats)) # expect in&out shape: [b, num_feats]
            # print("feats_z: ", feats_z.size())

            # concat for classification ########################
            if x_secview is not None:
                out_all = torch.cat((z_wholeview, z_secview, feats_z), dim=1)
            else:
                out_all = torch.cat((z_wholeview, feats_z), dim=1)
        else:
            if x_secview is not None:
                out_all = torch.cat((z_wholeview, z_secview), dim=1)
            else:
                out_all = z_wholeview
        # print("out_all: ", out_all.size())

        # dropout
        if self.dropout_ratio > 0:
            out_all = self.drop(out_all)

        cls = self.fc(out_all)

        return cls

if __name__=="__main__":
    model = DetailView_DfModel(num_views=7, num_classes=9, num_feats=5,
                 selected_model="convnext_tiny", imgnet_weight=False)

    imgs = torch.randn(size=(2,7,1,256,256))
    feats = torch.randn(size=(2,5))
    model(imgs, feats)
    print(model)
