import torch, torch.nn as nn
from math import sqrt
from itertools import product as product
from utils import match, log_sum_exp
import torch.nn.functional as F
from torchsummary import summary
from utils import point_form

class DDB_b(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DDB_b, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=1, stride=1),
            nn.BatchNorm2d(growth_rate)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, groups=growth_rate),
            nn.BatchNorm2d(growth_rate)
        )

    def forward(self, x):
        right = self.block1(x)
        right = self.block2(right)
        return torch.cat([x, right], dim=1)


class transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transition, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
        )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pooling(x)


class transition_w_o_pooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transition_w_o_pooling, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class downsample(nn.Module):
    def __init__(self, in_channels, pad=0):
        super(downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pooling = nn.MaxPool2d(2, 2, padding=pad)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        right = self.conv(x)
        left = self.pooling(x)
        left = self.conv2(left)
        return torch.cat([left, right], dim=1)


class upsample(nn.Module):
    def __init__(self, in_channels, scale):
        super(upsample, self).__init__()
        # self.up = nn.UpsamplingBilinear2d(scale)
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, x):
        # x = self.up(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x


class PriorBox:
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self):
        mean = []
        for k, f in enumerate(self.cfg['feature_maps']):
            for i, j in product(range(f), repeat=2):
                f_k = self.cfg['min_dim'] / self.cfg['steps'][k]
                # unit center x, y
                cx = (j + .5) / f_k
                cy = (i + .5) / f_k

                # aspect_ratio = 1, size = min_size
                s_k = self.cfg['min_sizes'][k] / self.cfg['min_dim']
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio = 1, size = sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.cfg['max_sizes'][k] / self.cfg['min_dim']))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.cfg['aspect_ratios'][k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output


class DCOD300_Body(nn.Module):
    def __init__(self, growth_rate, phase='train'):
        super(DCOD300_Body, self).__init__()
        self.growth_rate = growth_rate
        self.phase = phase

        self.cfg = {
            'num_classes': 21,
            'lr_steps': (80000, 100000, 120000),
            'max_iter': 120000,
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'min_dim': 300,
            'steps': [8, 16, 32, 64, 100, 300],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'name': 'VOC'
        }

        # Stem Block
        self.stem_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # stage 0
        in_channels = 128
        self.stage_0 = nn.ModuleList()
        for _ in range(4):
            self.stage_0.append(DDB_b(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        self.transition0 = transition(in_channels, in_channels // 2)
        in_channels //= 2

        # stage 1
        self.growth_rate += 16
        self.stage_1 = nn.ModuleList()
        for _ in range(6):
            self.stage_1.append(DDB_b(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        self.transition1 = transition_w_o_pooling(in_channels, 128)
        in_channels = 128

        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # stage 2
        self.growth_rate += 16
        self.stage_2 = nn.ModuleList()
        for _ in range(6):
            self.stage_2.append(DDB_b(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        self.transition2 = transition_w_o_pooling(in_channels, in_channels // 2)
        in_channels //= 2

        # stage 3
        self.growth_rate += 16
        self.stage_3 = nn.ModuleList()
        for _ in range(6):
            self.stage_3.append(DDB_b(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        self.transition3 = transition_w_o_pooling(in_channels, 64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # downsample
        odd = [1, 0, 1, 1]
        self.downsample = nn.ModuleList()
        for i in range(4):
            self.downsample.append(downsample(128, odd[i]))

        # reverse
        scale = [38/19, 19/10, 10/5, 5/3, 3/2]
        self.reverse_upsample = nn.ModuleList()
        self.reverse_relu = nn.ModuleList()
        for i in range(5):
            self.reverse_upsample.append(upsample(128, scale[i]))
            self.reverse_relu.append(nn.ReLU(inplace=True))

        # prediction layer
        self.conv_layer = nn.ModuleList()
        self.loc_layer = nn.ModuleList()
        for i in range(6):
            min_size = [self.cfg['min_sizes'][i]]
            aspect_ratio = self.cfg['aspect_ratios'][i]
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)

            num_loc = 4 * num_priors_per_location
            num_conf = self.cfg['num_classes'] * num_priors_per_location

            # confidence prediction layer
            self.conv_layer.append(nn.Sequential(
                nn.Conv2d(128, num_conf, 1),
                nn.Conv2d(num_conf, num_conf, kernel_size=3, padding=1, groups=num_conf),
                nn.BatchNorm2d(num_conf)
            ))

            self.loc_layer.append(nn.Sequential(
                nn.Conv2d(128, num_loc, 1),
                nn.Conv2d(num_loc, num_loc, kernel_size=3, padding=1, groups=num_loc),
                nn.BatchNorm2d(num_loc)
            ))

        self.priors = PriorBox(self.cfg).forward().cuda()

    def forward(self, x):
        outputs = []
        feature = []
        total_conf = []
        total_loc = []

        x = self.stem_block(x)

        for module in self.stage_0:
            x = module(x)
        x = self.transition0(x)

        for module in self.stage_1:
            x = module(x)
        x = self.transition1(x)
        feature.append(x)

        x = self.maxpool0(x)

        for module in self.stage_2:
            x = module(x)
        x = self.transition2(x)

        for module in self.stage_3:
            x = module(x)
        x = self.transition3(x)

        concat_layer = self.maxpool1(feature[-1])
        concat_layer = self.conv_bn_relu(concat_layer)
        x = torch.cat([concat_layer, x], dim=1)
        feature.append(x)

        for module in self.downsample:
            x = module(x)
            feature.append(x)

        # Reverse structure
        outputs.append(feature[-1])
        for i in range(4, -1, -1):
            outputs.append(self.reverse_relu[i](self.reverse_upsample[i](outputs[-1]) + feature[i]))

        # prediction layer
        for i in range(6):
            conf = self.conv_layer[i](outputs[i])
            conf = conf.permute(0, 2, 3, 1).contiguous()
            total_conf.append(conf.view(conf.size(0), -1))

            loc = self.loc_layer[i](outputs[i])
            loc = loc.permute(0, 2, 3, 1).contiguous()
            total_loc.append(loc.view(loc.size(0), -1))

        total_loc = torch.cat(total_loc, dim=1)
        total_conf = torch.cat(total_conf, dim=1)

        # if self.phase == 'train':
        output = (
            total_loc.view(total_loc.size(0), -1, 4),
            total_conf.view(total_conf.size(0), -1, self.cfg['num_classes']),
            self.priors
        ) if self.phase == 'train' else (
            None
        )
        return output


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh,
                 neg_mining, neg_pos, neg_overlap):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, 1:].data.sum(dim=1) > 0
            labels = targets[idx][:, 0][truths]
            truths = targets[idx][:, 1:][truths]
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        pos = conf_t > 0
        # num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        # return loss_l, loss_c
        return loss_l, loss_c


if __name__ == '__main__':
    model = DCOD300_Body(32)
    # model = DDB_b(128, 32)
    cnt = 0
    for name, item in model.named_parameters():
        layer_cnt = 1
        for i in item.size():
            layer_cnt *= i
        cnt += layer_cnt
        # print(name, layer_cnt)
    # print(cnt)
    summary(model, (3, 300, 300), device='cpu')
    # prior = PriorBox(cfg)
    # out = prior.forward()
    # print(out)