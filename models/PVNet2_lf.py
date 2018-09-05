from models import *
import config

class PVNet2_lf (nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_lf, self).__init__()
        self.n_neighbor = config.pv_net.n_neighbor

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)
        self.fusion_trans_ft = fc_layer(self.mvcnn.feature_len, 1024, True)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)


        self.conv2d5 = conv_2d(320, 1024, 1)
        self.fusion_mlp1 = nn.Sequential(
            fc_layer(2048, 512, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp22 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        if init_weights:
            self.init_mvcnn()
            self.init_dgcnn()

    def init_mvcnn(self):
        print(f'init parameter from mvcnn {config.base_model_name}')
        mvcnn_state_dict = torch.load(config.view_net.ckpt_file)['model']
        pvnet_state_dict = self.state_dict()

        mvcnn_state_dict = {k.replace('features', 'mvcnn', 1): v for k, v in mvcnn_state_dict.items()}
        mvcnn_state_dict = {k: v for k, v in mvcnn_state_dict.items() if k in pvnet_state_dict.keys()}
        pvnet_state_dict.update(mvcnn_state_dict)
        self.load_state_dict(pvnet_state_dict)

    def init_dgcnn(self):
        print(f'init parameter from dgcnn')
        dgcnn_state_dict = torch.load(config.pc_net.ckpt_file)['model']
        pvnet_state_dict = self.state_dict()

        dgcnn_state_dict = {k: v for k, v in dgcnn_state_dict.items() if k in pvnet_state_dict.keys()}

        pvnet_state_dict.update(dgcnn_state_dict)
        self.load_state_dict(pvnet_state_dict)

    def forward(self, pc, mv):
        mv_ft, _ = self.mvcnn(mv)
        mv_ft_trans = self.fusion_trans_ft(mv_ft)

        x_edge = get_edge_feature(pc, self.n_neighbor)
        x_trans = self.trans_net(x_edge)
        x = pc.squeeze().transpose(2, 1)
        x = torch.bmm(x, x_trans)
        x = x.transpose(2, 1)

        x1 = get_edge_feature(x, self.n_neighbor)
        x1 = self.conv2d1(x1)
        x1, _ = torch.max(x1, dim=-1, keepdim=True)

        x2 = get_edge_feature(x1, self.n_neighbor)
        x2 = self.conv2d2(x2)
        x2, _ = torch.max(x2, dim=-1, keepdim=True)

        x3 = get_edge_feature(x2, self.n_neighbor)
        x3 = self.conv2d3(x3)
        x3, _ = torch.max(x3, dim=-1, keepdim=True)

        x4 = get_edge_feature(x3, self.n_neighbor)
        x4 = self.conv2d4(x4)
        x4, _ = torch.max(x4, dim=-1, keepdim=True)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv2d5(x5)
        x5, _ = torch.max(x5, dim=-2, keepdim=True)


        net = x5.view(x5.size(0), -1)
        net = torch.cat((net, mv_ft_trans), dim=1)
        net = self.fusion_mlp1(net)
        net = self.fusion_mlp22(net)
        net = self.fusion_mlp3(net)
        return net