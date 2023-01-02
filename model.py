from collections import OrderedDict

import timm
import torch
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from torch import nn
from torchsummary import summary
from torchvision import models

from util import L2N, gem

class vit_base_patch8_224_dino(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = timm.create_model('vit_base_patch8_224_dino', pretrained=True)
        self.norm = L2N()

    def forward(self, x):
        x = self.base.forward_features(x)
        x = x[:, 0]
        x = self.norm(x)
        return x

class swin_large_patch4_window7_224_in22k(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=True)
        self.norm = L2N()

    def forward(self, x):
        x = self.base.forward_features(x)
        x = x[:, 0]
        x = self.norm(x)
        return x

class beitv2_large_patch16_224_in22k(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = timm.create_model('beitv2_large_patch16_224_in22k', pretrained=True)
        self.norm = L2N()

    def forward(self, x):
        x = self.base.forward_features(x)
        x = x[:, 0]
        x = self.norm(x)
        return x

# class MobileNet_AVG(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.base = nn.Sequential(OrderedDict(models.mobilenet_v2(pretrained=True).features.named_children()))
#         self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.norm = L2N()

#     def forward(self, x):
#         x = self.base(x)
#         x = self.pool(x).squeeze(-1).squeeze(-1)
#         x = self.norm(x)
#         return x


# class EfficientNet(nn.Module):
#     def __init__(self, arch='tf_efficientnetv2_m_in21ft1k', fc_dim=256, p=3.0, eval_p=4.0):
#         super().__init__()
#         self.backbone = timm.create_model(arch, features_only=True, pretrained=True)
#         self.fc = nn.Linear(self.backbone.feature_info.info[-1]['num_chs'], fc_dim, bias=False)
#         self.bn = nn.BatchNorm1d(fc_dim)
#         self._init_params()
#         self.p = p
#         self.eval_p = eval_p

#     def _init_params(self):
#         nn.init.xavier_normal_(self.fc.weight)
#         nn.init.constant_(self.bn.weight, 1)
#         nn.init.constant_(self.bn.bias, 0)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = self.backbone(x)[-1]
#         p = self.p if self.training else self.eval_p
#         x = gem(x, p).view(batch_size, -1)
#         x = self.fc(x)
#         x = self.bn(x)
#         x = F.normalize(x)
#         return x


if __name__ == '__main__':
    #model = vit_base_patch8_224_dino()
    model = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=True)
    dummy = torch.rand(1, 3, 224, 224)
    output = model(dummy)
    print(model)
    print(output.shape)
    # model = Hybrid_ViT()
    # print(summary(model.cuda(), (3, 224, 224)))
    # macs, params = get_model_complexity_info(model, (3, 224, 224))
    # print(macs, params)
