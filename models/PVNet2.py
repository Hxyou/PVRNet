from models import *
import config
import numpy as np

class PVNet2 (nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2, self).__init__()
        self.n_neighbor = config.pv_net.n_neighbor

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)


        self.conv2d5 = conv_2d(320, 1024, 1)

        # # self.fusion_trans_mv = fc_layer(self.mvcnn.feature_len, 1024, True)
        # # self.fusion_trans_pc = fc_layer(1024, 1024, True)
        # self.fusion_theta = conv_2d(256, 128, 1)
        # self.fusion_phi = conv_2d(1024, 128, 1)
        # self.fusion_g_mv = conv_2d(256, 512, 1)
        # self.fusion_g_pc = conv_2d(1024, 512, 1)
        #
        # self.fusion_mv_mlp1 = nn.Sequential(
        #     fc_layer(1024*6*6, 1024, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )
        # self.fusion_mlp1 = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )
        # self.fusion_mlp2 = nn.Sequential(
        #     fc_layer(1024, 512, True),
        #     nn.Dropout(p=0.5)
        # )
        self.fusion_mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        # self.fusion_mlp3 = fc_layer(256, n_classes, False)
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        pc_ft = x5.squeeze(3)
        pc_ft = pc_ft.transpose(1, 2)
        pc_ft = pc_ft.expand(-1, view_num, -1)
        mv_view = mv_view.view(batch_size*view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view = mv_view.view(batch_size, view_num, -1)
        fusion_ft = torch.cat((pc_ft, mv_view), 2)
        fusion_ft = fusion_ft.view(batch_size*view_num, fusion_ft.size(2))
        fusion_ft = self.fusion_fc(fusion_ft)
        fusion_ft = fusion_ft.view(batch_size, view_num, fusion_ft.size(1))
        # fusion_ft = torch.sum(fusion_ft, 1)
        fusion_ft, _ = torch.max(fusion_ft, 1)




        # pc_fp = self.conv2d5(x5)
        # # x5, _ = torch.max(pc_fp, dim=-2, keepdim=True)
        #
        # pc_uni = self.fusion_g_pc(pc_fp)
        # mv_uni = self.fusion_g_mv(mv_fp)
        # # net = x5.view(x5.size(0), -1)
        #
        # # (B*12, 256, 6, 6)=>(B, 256, 12*6*6, 1)=>(B, 128, 12*6*6, 1)
        # mv_fp_ = mv_fp.view(batch_size, mv_fp.size(1), -1)
        # mv_fp_ = mv_fp_.unsqueeze(3)
        # # (B, 1024, 1024, 1)=>(B, 128, 1024, 1)
        # # g_mv = self.fusion_g(mv_fp)
        # theta_mv = self.fusion_theta(mv_fp_)
        # phi_pc = self.fusion_phi(pc_fp)
        #
        # theta_mv = theta_mv.squeeze()
        # phi_pc = phi_pc.squeeze()
        # pc_fp = pc_fp.squeeze()
        # pc_uni = pc_uni.squeeze()
        # # mv_uni = mv_uni.squeeze()
        #
        # # mask => (B, 432, 1024)
        # # mask_mv => softmax(B, 432, 1024) =>(B, 1024, 432)
        # # gather_mv => (B, dims_pc, 1024) X (B, 1024, 432)=>(B, dims_pc, 432)=>(B*12, dims_pc, 6, 6)
        # # =>(B*12, dims_pc+256, 6, 6)=>(B*12, (dims_pc+256)*6*6)=>(B*12, 4096)=>(B, 4096)
        # theta_mv = theta_mv.permute(0, 2, 1)
        # mask = torch.matmul(theta_mv, phi_pc)
        # mask_mv = torch.nn.functional.softmax(mask, dim=2)
        # mask_mv = mask_mv.permute(0, 2, 1)
        # ft_mv = torch.matmul(pc_uni, mask_mv)
        # ft_mv = ft_mv.view(batch_size*view_num, ft_mv.size(1), 6, 6)
        # gather_mv = torch.cat((mv_uni, ft_mv), dim=1)
        # gather_mv_flatten = gather_mv.view(batch_size*view_num, -1)
        # gather_mv_flatten = self.fusion_mv_mlp1(gather_mv_flatten)
        # x = gather_mv_flatten.view(-1, view_num, gather_mv_flatten.size(1))
        # gather_mv_pooling, _ = torch.max(x, 1)
        # mv_ft_trans = gather_mv_pooling
        #
        # mv_uni = mv_uni.view(batch_size, mv_uni.size(1), -1)
        # mask_pc = torch.nn.functional.softmax(mask, dim=1)
        # ft_pc = torch.matmul(mv_uni.squeeze(), mask_pc)
        # gather_pc = torch.cat((pc_uni, ft_pc), dim=1)
        # gather_pc_pooling, _ = torch.max(gather_pc, dim=2)
        # pc_ft_trans = gather_pc_pooling
        #
        # net = torch.cat((pc_ft_trans, mv_ft_trans), dim=1)

        net_fea = self.fusion_mlp2(fusion_ft)
        net = self.fusion_mlp3(net_fea)
        if get_fea:
            return net, net_fea
        else:
            return net


class PVNet2_v4(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v4, self).__init__()
        self.n_centerpoint = 128
        self.n_cp_neighbor = 40
        self.n_scale = 4
        self.n_neighbor = config.pv_net.n_neighbor

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_conv1 = conv_2d(2048, 1, 1)
        self.sig = nn.Sigmoid()

        self.fusion_conv2 = conv_2d(2048, 512, 1)


        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = fc_layer(256, n_classes, False)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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
        # x5, _ = torch.max(x5, dim=-2, keepdim=True)

        pc_ft = get_neighbor_feature(x5, self.n_centerpoint, self.n_cp_neighbor)
        pc_ft, _ = torch.max(pc_ft, dim=-1)
        pc_ft = pc_ft.unsqueeze(1).expand(-1, view_num, -1, -1)
        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view = mv_view.view(batch_size, view_num, -1)
        mv_view_expand1 = mv_view.unsqueeze(3).expand(-1, -1, -1, self.n_centerpoint)
        fusion_ft = torch.cat((pc_ft, mv_view_expand1), 2)
        fusion_ft = fusion_ft.view(batch_size * view_num, fusion_ft.size(2), fusion_ft.size(3))
        fusion_ft = fusion_ft.unsqueeze(3)
        fusion_mask = self.fusion_conv1(fusion_ft)

        fusion_mask = fusion_mask.squeeze()
        # fusion_mask = self.sig(torch.log(fusion_mask))
        fusion_mask = self.sig(torch.log(fusion_mask))
        fusion_mask_val, fusion_mask_idx = torch.sort(fusion_mask, dim=-1, descending=True)
        fusion_mask_idx = fusion_mask_idx.view(batch_size, view_num, -1).unsqueeze(2).expand(-1, -1, pc_ft.size(2), -1)
        pc_rank = torch.gather(pc_ft, -1, fusion_mask_idx)
        pc_local_ft = []
        for i in range(self.n_scale):
            pc_local_ft.append(pc_rank[:, :, :, int(self.n_centerpoint/self.n_scale)*i:int(self.n_centerpoint/self.n_scale)*(i+1)].max(dim=-1, keepdim=True)[0])
        pc_global_ft, _ = torch.max(x5, dim=-2, keepdim=True)
        pc_global_ft = pc_global_ft.squeeze(-1).unsqueeze(1).expand(-1, view_num, -1, -1)
        pc_multiscale_ft = torch.cat((pc_local_ft[0], pc_local_ft[1], pc_local_ft[2], pc_local_ft[3], pc_global_ft), dim=-1)
        mv_view_expand2 = mv_view.unsqueeze(3).expand(-1, -1, -1, self.n_scale+1)
        fusion_ft1 = torch.cat((pc_multiscale_ft, mv_view_expand2), dim=2)
        fusion_ft1 = fusion_ft1.view(batch_size*view_num, fusion_ft1.size(2), fusion_ft1.size(3)).unsqueeze(3)
        fusion_ft1 = self.fusion_conv2(fusion_ft1)
        # pc_local_mask = []
        # for i in range(self.n_scale):
        #     pc_local_mask.append(fusion_mask_val[:, int(self.n_centerpoint/self.n_scale)*i:int(self.n_centerpoint/self.n_scale)*(i+1)].max(dim=-1, keepdim=True)[0])
        # pc_multiscale_mask = torch.cat((pc_local_mask[0], pc_local_mask[1], pc_local_mask[2], pc_local_mask[3]), dim=-1)
        # pc_multiscale_mask = pc_multiscale_mask.unsqueeze(1)
        # pc_multiscale_mask = pc_multiscale_mask.unsqueeze(3).expand(-1, 512, -1, -1)
        # fusion_ft1_local = fusion_ft1[:, :, :self.n_scale, :]*pc_multiscale_mask
        # fusion_ft1_local = fusion_ft1_local.sum(dim=-2, keepdim=True)

        fusion_ft1_local = fusion_ft1[:, :, :self.n_scale, :]
        fusion_ft1_local = fusion_ft1_local.max(dim=-2, keepdim=True)[0]
        fusion_ft1_global = fusion_ft1[:, :, -1, :]
        fusion_ft1_global = fusion_ft1_global.squeeze()
        fusion_ft1_local = fusion_ft1_local.squeeze()
        fusion_ft1_all = torch.cat((fusion_ft1_local, fusion_ft1_global), dim=1)
        fusion_ft1_all = fusion_ft1_all.view(batch_size, view_num, -1)
        fusion_ft, _ = torch.max(fusion_ft1_all, 1)

        net = self.fusion_mlp2(fusion_ft)
        net = self.fusion_mlp3(net)

        return net


class PVNet2_v5(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v5, self).__init__()
        self.n_centerpoint = 128
        self.n_cp_neighbor = 40
        self.n_scale = 4
        self.n_neighbor = config.pv_net.n_neighbor

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.sig = nn.Sigmoid()

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
            nn.Dropout(p=0.5)
        )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = fc_layer(256, n_classes, False)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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
        # x5, _ = torch.max(x5, dim=-2, keepdim=True)

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_expand = mv_view.unsqueeze(2).expand(-1, -1, 1024)
        mv_expand = mv_expand.unsqueeze(3)
        pc_expand = x5.unsqueeze(1).expand(-1, view_num, -1, -1, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, pc_expand.size(2), pc_expand.size(3), pc_expand.size(4))
        fusion_mask = torch.cat((pc_expand, mv_expand), 1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = self.sig(torch.log(fusion_mask))
        pc_expand_ft = torch.mul(pc_expand, fusion_mask)
        pc_expand_ft = torch.add(pc_expand_ft, pc_expand)
        pc_expand_ft, _ = torch.max(pc_expand_ft, dim=2)
        pc_expand_ft = pc_expand_ft.squeeze()
        fusion_ft = torch.cat((mv_view, pc_expand_ft), dim=1)
        fusion_ft = self.fusion_fc(fusion_ft)
        fusion_ft = fusion_ft.view(batch_size, view_num, -1)
        fusion_ft, _ = torch.max(fusion_ft, dim=1)

        net = self.fusion_mlp2(fusion_ft)
        net = self.fusion_mlp3(net)

        return net


class PVNet2_v6(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v6, self).__init__()

        self.n_scale = [6, 8, 12]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )
        self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)

        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)
        fusion_mask = torch.nn.functional.softmax(fusion_mask, dim=1)
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))
        mv_view = mv_view.view(batch_size, view_num, -1)
        mv_view_scale=[]
        mv_pc_scale=[]
        for i in range(len(self.n_scale)):
            mv_view_scale.append(torch.gather(mv_view, 1, mask_idx[:, 0:self.n_scale[i], :]))
            mv_pc_scale.append(torch.cat((pc, mv_view_scale[i].max(1)[0]), dim=1).unsqueeze(2))

        mv_pc_multi = torch.cat(mv_pc_scale, 2).unsqueeze(3)
        mv_pc_multi = self.fusion_conv2(mv_pc_multi)
        fusion_ft,_ = torch.max(mv_pc_multi, dim=2)
        fusion_ft = fusion_ft.squeeze()


        net = self.fusion_mlp2(fusion_ft)
        net = self.fusion_mlp3(net)

        return net



class PVNet2_v7(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v7, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        self.n_scale = [2, 3, 5]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)

        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        fusion_global = self.fusion_fc(fusion_mask)
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)
        fusion_mask = torch.nn.functional.softmax(fusion_mask, dim=1)
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))
        mv_view_expand = mv_view.view(batch_size, view_num, -1)
        scale_out = []

        for i in range(len(self.n_scale)):
            random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_expand, 1, mask_idx[:, random_idx, :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).mean(2)
        scale_out = torch.cat(scale_out, dim=2).max(2)[0]
        final_out = torch.cat((scale_out, fusion_global),1)

        net = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net)

        return net



class PVNet2_v8(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v8, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        self.n_scale = [2, 3, 5]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        fusion_global = self.fusion_fc(fusion_mask)
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)
        fusion_mask = self.sig(fusion_mask)
        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))
        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        scale_out = torch.cat(scale_out, dim=2).mean(2)
        final_out = torch.cat((scale_out, fusion_global),1)

        net = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net)

        return net


class PVNet2_v9(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v9, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 5]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        fusion_mask = self.sig(fusion_mask)
        # fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        scale_out = torch.cat(scale_out, dim=2).mean(2)
        final_out = torch.cat((scale_out, fusion_global),1)

        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            # return net, final_out
            return net, net_fea
        else:
            return net


class PVNet2_v10(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v10, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
        if init_weights:
            self.init_mvcnn()
            self.init_dgcnn()

    def init_mvcnn(self):
        print(f'init parameter from mvcnn {config.base_model_name}')
        mvcnn_state_dict = torch.load(config.view_net.ckpt_file)['model']
        pvnet_state_dict = self.state_dict()

        # mvcnn_state_dict = {k.replace('features', 'mvcnn', 1): v for k, v in mvcnn_state_dict.items()}
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        # fusion_mask = self.sig(fusion_mask)
        fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        scale_out = torch.cat(scale_out, dim=2).mean(2)
        final_out = torch.cat((scale_out, fusion_global),1)

        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            return net, net_fea
        else:
            return net

class PVNet2_v11(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v11, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 5]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(2048, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        fusion_mask = self.sig(fusion_mask)
        # fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale)
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        # scale_out = torch.cat(scale_out, dim=2).mean(2)
        scale_out = torch.cat(scale_out, dim=1)
        final_out = torch.cat((scale_out, fusion_global),1)

        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            return net, net_fea
        else:
            return net



class PVNet2_v12(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v12, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 5]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        fusion_mask = self.sig(fusion_mask)
        # fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        # scale_out = torch.cat(scale_out, dim=2).mean(2)
        scale_out = torch.cat(scale_out, dim=2)
        final_out = torch.cat((scale_out, fusion_global.unsqueeze(2)),2).mean(2)

        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            return net, net_fea
        else:
            return net


class PVNet2_v13(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v13, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 5]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.6)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        fusion_mask = self.sig(fusion_mask)
        # fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        # scale_out = torch.cat(scale_out, dim=2).mean(2)
        scale_out = torch.cat(scale_out, dim=2)
        final_out = torch.cat((scale_out, fusion_global.unsqueeze(2)),2).max(2)[0]

        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            return net, net_fea
        else:
            return net


class PVNet2_v14(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v14, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 5]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2, 3]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        fusion_mask = self.sig(fusion_mask)
        # fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        scale_out = torch.cat(scale_out, dim=2).mean(2)
        final_out = torch.cat((scale_out, fusion_global),1)

        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            # return net, final_out
            return net, net_fea
        else:
            return net



class PVNet2_v15(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_v15, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 5]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        fusion_mask = self.sig(fusion_mask)
        # fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        scale_out = torch.cat(scale_out, dim=2).mean(2)
        final_out = torch.cat((scale_out, fusion_global),1)

        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            # return net, final_out
            return net, net_fea
        else:
            return net



class PVNet2_vv0(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_vv0, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 5]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )

        self.fusion_mlp1 = nn.Sequential(
            fc_layer(1024, 512, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        # fusion_mask = self.sig(fusion_mask)
        fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        # mv_view_enhance = torch.mul(mv_view_expand, fusion_mask)
        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        scale_out = torch.cat(scale_out, dim=2).mean(2)
        final_out = torch.cat((scale_out, fusion_global),1)

        net = self.fusion_mlp1(final_out)
        net_fea = self.fusion_mlp2(net)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            # return net, final_out
            return net, net_fea
        else:
            return net


class PVNet2_vv1(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVNet2_vv1, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        # self.n_scale = [3, 5, 8]
        # self.n_scale = [2, 3, 5]
        # self.n_scale = [2, 3, 6]
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pv_net.n_neighbor

        # self.lstm = torch.nn.LSTM()

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)

        self.conv2d5 = conv_2d(320, 1024, 1)


        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        # self.fusion_fc_pc = nn.Sequential(
        #     fc_layer(1024, 1024, True),
        # )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        # self.fusion_conv1 = conv_2d(2048, 1024, 1)
        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]
        # self.fusion_conv2 = conv_2d(2048, 512, 1)
        self.sig = nn.Sigmoid()

        # self.fusion_fc = nn.Sequential(
        #     fc_layer(2048, 512, True),
        #     nn.Dropout(p=0.5)
        # )
        self.fusion_mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        # self.fusion_mlp2 = nn.Sequential(
        #     fc_layer(1024, 256, True),
        #     nn.Dropout(p=0.5)
        # )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        # self.fusion_mlp2 = fc_layer(512, n_classes, False)
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

    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)


        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)
        # pc_expand = self.fusion_fc_pc(pc_expand)
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        # fusion_global = self.fusion_fc(fusion_mask)
        # fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)

        fusion_mask = self.sig(fusion_mask)
        # fusion_mask = self.sig(torch.log(torch.abs(fusion_mask)))
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand
        # mv_view_enhance = mv_view_expand
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)
        # fusion_global = torch.mean(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        scale_out = []

        for i in range(len(self.n_scale)):
            # random_idx = np.random.choice(int(view_num/self.n_scale[i]), self.n_scale[i], replace=True)
            # random_idx = [v + k * int(view_num/self.n_scale[i]) for k, v in enumerate(list(random_idx))]
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        # scale_out = torch.cat(scale_out, dim=2).max(2) v7
        # scale_out = torch.cat(scale_out, dim=2).mean(2)
        # final_out = torch.cat((scale_out, fusion_global),1)

        scale_out = torch.cat(scale_out, dim=2)
        final_out = torch.cat((scale_out, fusion_global.unsqueeze(2)),2).mean(2)


        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            # return net, final_out
            return net, net_fea
        else:
            return net