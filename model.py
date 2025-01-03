import torch
import torch.nn as nn
import torch.nn.functional as F


class DRA(nn.Module):
    def __init__(self, inplanes, stride=(1, 1)):
        super(DRA, self).__init__()
        planes = inplanes // 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(planes, affine=True),
            nn.ELU()
        )
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, inplanes // 3, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(inplanes // 3, affine=True),
            nn.ELU()
        )
        self.elu = nn.ELU()
        self.squeeze = inplanes // 6
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inplanes, self.squeeze, (1, 1), (1, 1), (0, 0)),
            # nn.BatchNorm2d(self.squeeze),
            nn.ELU(),
            nn.Conv2d(self.squeeze, 4, (1, 1), (1, 1), (0, 0)),
        )
        self.sf = nn.Softmax(dim=1)
        self.conv_s1 = nn.Conv2d(planes, planes, (1, 1), stride=stride, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(planes, planes, (1, 1), stride=stride, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(planes, planes, (1, 1), stride=stride, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(planes, planes, (1, 1), stride=stride, padding=0, bias=False)

    def forward(self, x):
        # x = torch.cat([x1, x2, x3], dim=1)  # 32,48,16,512
        b, c, h, w = x.size()
        y = self.fc(self.avg_pool(x).view(b, c, 1, 1)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        out = self.conv1(x)

        dyres = self.conv_s1(out) * y[:, 0] + self.conv_s2(out) * y[:, 1] + self.conv_s3(out) * y[:, 2] + self.conv_s4(out) * y[:, 3]

        out = self.bn2(dyres)
        out = self.elu(out)

        out = self.conv3(out)

        return out


class AttentionBasicBlock(nn.Module):
    def __init__(self, planes, pad, length):
        super(AttentionBasicBlock, self).__init__()
        # ch attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # t = int(abs((math.log(planes * 1, 2) + 1)) / 2)  # 12:555777; 1:333555
        # k = t if t % 2 else t + 1
        # self.squeeze = min(32, planes//6)
        self.ch_att = nn.Sequential(
            nn.Conv2d(planes, planes//16, (1, 1), (1, 1), (0, 0)),
            nn.ELU(),
            nn.Conv2d(planes//16, planes, (1, 1), (1, 1), (0, 0)),
        )
        self.sigmoid = nn.Sigmoid()

        self.squeeze_dim = 12
        # self.squeeze_hw = 2  # if length >= 128 else 1
        self.squeeze_hw = 4 if length >= 128 else 2
        # self.att_drop = nn.Dropout(0.25)

        self.query_conv = nn.Conv2d(in_channels=planes, out_channels=planes // self.squeeze_dim, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=planes, out_channels=planes // self.squeeze_dim, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(1, 1))

        self.anchor = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes // self.squeeze_dim, kernel_size=(1, 1)),
            nn.AvgPool2d((1, self.squeeze_hw))
        )

        self.softmax = nn.Softmax(dim=-1)

        self.DWC = nn.Sequential(
            nn.ZeroPad2d((pad*2, pad*2 - 1, 0, 0)),
            nn.Conv2d(planes, planes, kernel_size=(1, pad * 4), stride=(1, 1), padding=(0, 0), bias=False,
                      groups=planes),
            nn.BatchNorm2d(planes),
            nn.ELU(),
        )

        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        # x = torch.cat([x1, x2, x3], dim=1)  # 32,48,16,512
        m_batchsize, C, height, width = x.size()
        fea_s = self.avg_pool(x)
        attention_vectors = self.ch_att(fea_s)
        attention_vectors = self.sigmoid(attention_vectors)  # 32,48,1,1
        att_x = attention_vectors * x

        att_x_res = att_x + x
        anchor = self.anchor(att_x_res)
        # anchor = anchor.view(m_batchsize, -1, width * height // (self.squeeze_hw * self.squeeze_hw))  # C, WH//S
        anchor = anchor.view(m_batchsize, -1, width * height // self.squeeze_hw)  # C, WH//S

        proj_key = self.key_conv(att_x_res)
        proj_key = proj_key.view(m_batchsize, -1, width * height)  # HW, C
        proj_key_e = F.normalize(proj_key, dim=-1).transpose(-2, -1) @ F.normalize(anchor, dim=-1)  # HW, HW//S
        proj_key_s = self.softmax(proj_key_e)
        # proj_key_s = self.att_drop(proj_key_s)

        proj_value = self.value_conv(att_x_res)
        proj_value_s = proj_value.view(m_batchsize, -1, width * height)
        # proj_value_s = self.down_sample_v(proj_value).view(m_batchsize, -1, width * height)

        z = proj_value_s @ proj_key_s  # D, HW//S

        proj_query = self.query_conv(att_x_res)
        proj_query = proj_query.view(m_batchsize, -1, width * height)  # C, HW
        proj_query_e = F.normalize(anchor, dim=-1).transpose(-2, -1) @ F.normalize(proj_query, dim=-1)  # WH//S, WH
        proj_query_s = self.softmax(proj_query_e)
        # proj_query_s = self.att_drop(proj_query_s)

        out = z @ proj_query_s

        out = out.view(m_batchsize, C, height, width)

        out = out + self.DWC(proj_value) + att_x

        out = self.bn(out)

        return out


class AttentionBlock(nn.Module):
    def __init__(self, planes, pad, length):
        super(AttentionBlock, self).__init__()
        # ch attention
        self.head = 1
        self.head_list = nn.ModuleList()
        for h in range(self.head):
            self.att_basic = AttentionBasicBlock(planes, pad, length)
            self.head_list.append(self.att_basic)

    def forward(self, x):
        # x = torch.cat([x1, x2, x3], dim=1)  # 32,48,16,512
        # b, d, c, t = x.size()
        for h in range(self.head):
            if h == 0:
                out = self.head_list[h](x)
            else:
                out = self.head_list[h](out)

        fea_v = torch.chunk(out, chunks=3, dim=1)
        fea_v = tuple(fea_v)
        fea_v = torch.stack(fea_v, dim=1)
        fea_v = fea_v.sum(dim=1)

        return fea_v


class BN_Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels, affine=True),
        )

    def forward(self, x):
        return F.elu(self.seq(x))


class Stem_A_res(nn.Module):

    def __init__(self, in_channels, b1_n2, b2_n2, b3_n2, n1_linear, pad1, pad2, pad3, length):
        super(Stem_A_res, self).__init__()
        self.branch1 = nn.Sequential(
            nn.ZeroPad2d((pad1, pad1 - 1, 0, 0)),
            BN_Conv2d(in_channels, b1_n2, (1, 2 * pad1), (1, 1), (0, 0), bias=False),
            nn.Dropout(0.25)
        )
        self.branch2 = nn.Sequential(
            nn.ZeroPad2d((pad2, pad2 - 1, 0, 0)),
            BN_Conv2d(in_channels, b2_n2, (1, 2 * pad2), (1, 1), (0, 0), bias=False),
            nn.Dropout(0.25)
        )
        self.branch3 = nn.Sequential(
            nn.ZeroPad2d((pad3, pad3 - 1, 0, 0)),
            BN_Conv2d(in_channels, b3_n2, (1, 2 * pad3), (1, 1), (0, 0), bias=False),
            nn.Dropout(0.25)
        )

        self.dar = DRA(n1_linear * 3)

        self.att = AttentionBlock(n1_linear * 3, pad2, length)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        U = torch.cat([out1, out2, out3], dim=1)  # 32,48,16,512
        dar = self.dar(U)
        att = self.att(U)
        out = dar + att
        # print(att[0, 0, 0, :])
        return out


class Inception_A_res(nn.Module):

    def __init__(self, in_channels, b1_n1, b1_n2, b2_n1, b2_n2, b3_n1, b3_n2, n1_linear, pad1, pad2, pad3, length):
        super(Inception_A_res, self).__init__()
        self.branch1 = nn.Sequential(
            nn.ZeroPad2d((pad1, pad1 - 1, 0, 0)),
            BN_Conv2d(in_channels, b1_n1, (1, 1), (1, 1), (0, 0), bias=False),
            BN_Conv2d(b1_n1, b1_n2, (1, 2 * pad1), (1, 1), (0, 0), bias=False),
            nn.Dropout(0.25)
        )
        self.branch2 = nn.Sequential(
            nn.ZeroPad2d((pad2, pad2 - 1, 0, 0)),
            BN_Conv2d(in_channels, b2_n1, (1, 1), (1, 1), (0, 0), bias=False),
            BN_Conv2d(b2_n1, b2_n2, (1, 2 * pad2), (1, 1), (0, 0), bias=False),
            nn.Dropout(0.25)
        )
        self.branch3 = nn.Sequential(
            nn.ZeroPad2d((pad3, pad3 - 1, 0, 0)),
            BN_Conv2d(in_channels, b3_n1, (1, 1), (1, 1), (0, 0), bias=False),
            BN_Conv2d(b3_n1, b3_n2, (1, 2 * pad3), (1, 1), (0, 0), bias=False),
            nn.Dropout(0.25)
        )

        self.dar = DRA(in_channels * 3)

        self.att = AttentionBlock(n1_linear * 3, pad2, length)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        U = torch.cat([out1, out2, out3], dim=1)  # 32,48,16,512
        dar = self.dar(U)
        att = self.att(U)
        out = dar + att
        # print(att[0, 0, 0, :])
        return out


class Spatial_Conv_res(nn.Module):
    
    def __init__(self, in_channels, b1):
        super(Spatial_Conv_res, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 8, 7)),
            BN_Conv2d(in_channels, b1, (16, 1), (1, 1), (0, 0), bias=False, groups=in_channels),
            nn.Dropout(0.25)
        )

        self.short_cut = nn.Sequential()
        if in_channels != b1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, b1, (1, 1), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(b1, affine=True),
                nn.ELU()
            )

    def forward(self, x):
        out = self.seq(x)
        out = out + self.short_cut(x)
        return out


class Reduction_B(nn.Module):

    def __init__(self, in_channels, b2_n1, b2_n2, b3_n1, b3_n2, b3_n3, n1_linear, pad3, length):
        super(Reduction_B, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d((1, 8), (1, 4), (0, 2)),
            nn.Dropout(0.25)
        )
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, (1, 1), (1, 1), (0, 0), bias=False),
            BN_Conv2d(b2_n1, b2_n2, (1, 8), (1, 4), (0, 2), bias=False),
            nn.Dropout(0.25)
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, (1, 1), (1, 1), (0, 0), bias=False),
            BN_Conv2d(b3_n1, b3_n2, (1, 8), (1, 1), (0, 2), bias=False),
            BN_Conv2d(b3_n2, b3_n3, (1, 8), (1, 4), (0, 4), bias=False),
            nn.Dropout(0.25)
        )

        self.dar = DRA(in_channels * 3)

        self.att = AttentionBlock(in_channels * 3, pad3, length)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        U = torch.cat([out1, out2, out3], dim=1)  # 32,48,16,512
        dar = self.dar(U)
        att = self.att(U)
        out = dar + att
        # print(att[0, 0, 0, :])
        return out


class ConfuseNet(nn.Module):

    def __init__(self, num_classes=2):
        super(ConfuseNet, self).__init__()

        self.inception = self.make_inception()

        self.spatial_merge = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(16, 1), groups=128, bias=False),
            nn.Conv2d(128, 256, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ELU(),
            nn.Dropout(0.25)
        )

        self.temporal_merge_1 = nn.Sequential(
            nn.Conv2d(256, 256, (1, 3), (1, 1), (0, 1), groups=256, bias=False),
            nn.Conv2d(256, 512, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)

        )

        self.fc0 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Dropout(0.25)
        )

        self.fc = nn.Linear(128, num_classes)

    def make_inception(self):
        layers = [Stem_A_res(1, 16, 16, 16, 16, 8, 16, 32, 512),
                  Spatial_Conv_res(16, 32),
                  Reduction_B(32, 16, 32, 16, 16, 32, 32, 16, 128),

                  Inception_A_res(32, 16, 32, 16, 32, 16, 32, 32, 4, 8, 16, 128),
                  Spatial_Conv_res(32, 64),
                  Reduction_B(64, 32, 64, 32, 32, 64, 64, 8, 32),

                  Inception_A_res(64, 32, 64, 32, 64, 32, 64, 64, 2, 4, 8, 32),
                  Spatial_Conv_res(64, 128),
                  Reduction_B(128, 64, 128, 64, 64, 128, 128, 4, 8),
                  ]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.inception(x) 
        out = self.spatial_merge(out)
        out = self.temporal_merge_1(out)
        out = out.view(out.size(0), -1)
        out_0 = self.fc0(out)
        out = self.fc(out_0)
        return out, out_0

