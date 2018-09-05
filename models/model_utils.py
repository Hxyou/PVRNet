import torch
import config
import torch.nn as nn


ALEXNET = "ALEXNET"
DENSENET121 = "DENSENET121"
VGG13 = "VGG13"
VGG13BN = "VGG13BN"
VGG11BN = 'VGG11BN'
RESNET50 = "RESNET50"
RESNET101 = "RESNET101"
INCEPTION_V3 = 'INVEPTION_V3'

# MVCNN functions


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(conv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(fc_layer, self).__init__()
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.fc(x)
        return x



class transform_net(nn.Module):
    def __init__(self, in_ch, K=3, pc_input_num=1024):
        super(transform_net, self).__init__()
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(128, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(pc_input_num, 1))
        self.fc1 = fc_layer(1024, 512, bn=True)
        self.fc2 = fc_layer(512, 256, bn=True)
        self.fc3 = nn.Linear(256, K*K)


    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x, _ = torch.max(x, dim=-1, keepdim=True)
        x = self.conv2d3(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(3).view(1,9).repeat(x.size(0),1)
        iden = iden.to(device=config.device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


def pairwise_distance(x):
    batch_size = x.size(0)
    point_cloud = torch.squeeze(x)
    if batch_size == 1:
        point_cloud = torch.unsqueeze(point_cloud, 0)
    point_cloud_transpose = torch.transpose(point_cloud, dim0=1, dim1=2)
    point_cloud_inner = torch.matmul(point_cloud_transpose, point_cloud)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(point_cloud ** 2, dim=1, keepdim=True)
    point_cloud_square_transpose = torch.transpose(point_cloud_square, dim0=1, dim1=2)
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def gather_neighbor(x, nn_idx, n_neighbor):
    x = torch.squeeze(x)
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x.unsqueeze(2).expand(batch_size, num_dim, num_point, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, num_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n

def get_neighbor_feature(x, n_point, n_neighbor):
    if len(x.size()) == 3:
        x = x.unsqueeze()
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    nn_idx = nn_idx[:, :n_point, :]
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x[:, :, :n_point, :].expand(-1, -1, -1, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, n_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n


def get_edge_feature(x, n_neighbor):
    if len(x.size()) == 3:
        x = x.unsqueeze(3)
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    point_cloud_neighbors = gather_neighbor(x, nn_idx, n_neighbor)
    point_cloud_center = x.expand(-1, -1, -1, n_neighbor)
    edge_feature = torch.cat((point_cloud_center, point_cloud_neighbors-point_cloud_center), dim=1)
    return edge_feature


class residual_attn_block(nn.Module):
    def __init__(self, c_in=64, c_mv_ft=1024, K=20, c_out=64, c_attn=256):
        """
        :param c_in:
        :param c_mv_ft: view feature channel
        :param k: k neighbors
        :param c_out: output channel
        :param c_attn:
        """
        super(residual_attn_block, self).__init__()
        self.c_in = c_in
        self.K = K
        self.conv2d_pc = conv_2d(c_in*2, c_out, [1, 1])
        self.conv2d_pc_mv = conv_2d(c_in+c_mv_ft, c_out, [1, 1])
        self.sig = nn.Sigmoid()

    def forward(self, pc, mv_ft):
        """
        :param pc: (N, C_in, 1, P)
        :param mv_ft:  (N, C(1024))
        :return: (N, C_out, 1, P)
        """
        # pc = torch.squeeze(pc)
        num_points = pc.size(2)
        edge_ft = get_edge_feature(pc, self.K)
        pc_out = self.conv2d_pc(edge_ft)
        pc_out, _ = torch.max(pc_out, dim=-1, keepdim=True)

        pc_mv_ft = mv_ft.unsqueeze(2).unsqueeze(2).expand(-1, -1, num_points, -1)
        pc_mv_ft = torch.cat((pc_mv_ft, pc), dim=1)
        pc_mv_mask = self.conv2d_pc_mv(pc_mv_ft)
        pc_mv_mask = self.sig(torch.log(pc_mv_mask))

        pc_out_atten = torch.mul(pc_mv_mask, pc_out)
        pc_out = torch.add(pc_out_atten, pc_out)
        return pc_out


class residual_attn_block_no_mv(nn.Module):
    def __init__(self, c_in=64, K=20, c_out=64, c_attn=256):
        """
        :param c_in:
        :param c_mv_ft: view feature channel
        :param k: k neighbors
        :param c_out: output channel
        :param c_attn:
        """
        super(residual_attn_block_no_mv, self).__init__()
        self.c_in = c_in
        self.K = K
        self.conv2d_pc = conv_2d(c_in*2, c_out, [1, 1])
        self.conv2d_pc_at= conv_2d(c_in, c_out, [1, 1])
        self.sig = nn.Sigmoid()

    def forward(self, pc):
        """
        :param pc: (N, C_in, 1, P)
        :param mv_ft:  (N, C(1024))
        :return: (N, C_out, 1, P)
        """
        # pc = torch.squeeze(pc)
        num_points = pc.size(2)
        edge_ft = get_edge_feature(pc, self.K)
        pc_out = self.conv2d_pc(edge_ft)
        pc_out, _ = torch.max(pc_out, dim=-1, keepdim=True)

        pc_mask = self.conv2d_pc_at(pc)
        pc_mask = self.sig(torch.log(pc_mask))

        pc_out_atten = torch.mul(pc_mask, pc_out)
        pc_out = torch.add(pc_out_atten, pc_out)
        return pc_out

