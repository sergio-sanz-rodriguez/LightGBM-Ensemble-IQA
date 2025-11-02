import torch
from einops import rearrange
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms, utils, models
from utils.convnext import convnext_base as convnext
from utils.convnext import convnext_large as convnext_l


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class _CEncoder(nn.Module):
    def __init__(self):
        super(_CEncoder, self).__init__()
        self.base_model = convnext_l(pretrained=True).eval()
        self.norm = LayerNorm(1536)
        self.model = nn.Sequential(
            nn.Conv2d(1536, 768, kernel_size=1, stride=1, padding=0),
            LayerNorm(768)
        )

    def forward(self, x):
        x4, x3, x2, x1 = self.base_model(x)
        x1 = self.norm(x1)
        x1 = self.model(x1)
        # print(x1.shape)
        return x1, x2, x3, x4


class _DEncoder(nn.Module):
    def __init__(self):
        super(_DEncoder, self).__init__()
        self.base_model = convnext_l(pretrained=True).eval()
        self.norm = LayerNorm(1536)
        self.model = nn.Sequential(
            nn.Conv2d(1536, 768, kernel_size=1, stride=1, padding=0),
            LayerNorm(768)
        )

    def forward(self, x):
        x4, x3, x2, x1 = self.base_model(x)
        x1 = self.norm(x1)
        x1 = self.model(x1)
        return x1, x2, x3, x4


class qhead(nn.Module):
    def __init__(self):
        super(qhead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        embed_dim = 768
        num_outputs = 1
        drop = 0.1
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score


class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError


class AFF(AbstractFusion):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        self.channels = channels
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            LayerNorm(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            LayerNorm(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        batch_size = x.size(0)
        channels = x.size(1)
        xa = torch.cat((x, residual), dim=1)
        xl = self.local_att(xa)
        # xl = self.global_att(xa)
        xg = self.global_att(xa)
        # xg = self.local_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        # x_fusion = 2 * x * wei + 2 * residual * (1 - wei)
        # x_fusion = torch.cat((x * wei[:, :channels, :, :], residual * wei[:, channels:, :, :]), dim=1)
        x_fusion = (x * wei[:, :channels, :, :]) + (residual * wei[:, channels:, :, :])
        # h = x_fusion.size(2)
        # w = x_fusion.size(3)
        # x_fusion = x_fusion.view(batch_size,self.channels,h*w)
        # x_mm = x_fusion.sum(2)

        # x_mm = torch.mul(torch.sign(x_mm),torch.sqrt(torch.abs(x_mm) + 1e-8))
        # x_mm = torch.nn.functional.normalize(x_mm)
        return x_fusion


class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.CEncoder = _CEncoder()
        #for k, v in self.CEncoder.named_parameters():
        #    print(k)
        # pre_model_1 = torch.load(r'D:\PythonProject\xinbo\model\dist_cls_model_bs=12_2e-5(2)Adam(0.1)_f1=0.677113')
        # pre_model_1 = torch.load(r'models_save/convext_large_cls_channle=768_f1=0.746753')
        pre_model_1 = torch.load(r'VCIP_IMQA/VCIP/IMQA/convext_large_cls_channle=768_f1=0.746753')
        predict = {}
        for j in pre_model_1.keys():
            key = j.replace('encoder.', '')
            # print(key)
            predict[key] = pre_model_1[j]
        self.CEncoder.load_state_dict(predict, strict=False)

        self.DEncoder = _DEncoder()
        # pre_model_2 = torch.load(r'models_save/content_aware_channle=768_bestLoss=0.001123')
        pre_model_2 = torch.load(r'VCIP_IMQA/VCIP/IMQA/content_aware_channle=768_bestLoss=0.001123')
        self.DEncoder.load_state_dict(pre_model_2, strict=False)
        self.qhead = qhead()
        self.fusion = AFF(channels=1536)

    def forward(self, x):
        x_c1, x_c2, x_c3, x_c4 = self.CEncoder(x)
        x_d1, x_d2, x_d3, x_d4 = self.DEncoder(x)
        #x_c1 = self.CEncoder(x)
        #x_d1 = self.DEncoder(x)
        #print(x_c1.shape)
        #print(x_d1.shape)

        x = self.fusion(x_c1, x_d1)

        # x = x_c1 + x_d1 #x_c1*x_d1
        # x = x_c1
        score = self.qhead(x)
        return score
