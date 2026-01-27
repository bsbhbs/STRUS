import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x

        return x

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        self.enable_padding = enable_padding
        if enable_padding:
            self.padding = (int((kernel_size[0] - 1) // 2 * dilation[0]), 0) 
        else:
            self.padding = (0, 0)
        super(CausalConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding,
            dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, input):
        return super(CausalConv2d, self).forward(input)


class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=2 * c_out,
                kernel_size=(Kt, 1),
                enable_padding=True, 
                dilation=1
            )
        else:
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(Kt, 1),
                enable_padding=True,  
                dilation=1
            )
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x):
        x_in = self.align(x)  
        x_causal_conv = self.causal_conv(x)  

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :] 
            x_q = x_causal_conv[:, -self.c_out:, :, :] 
            if self.act_func == 'glu':
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
            else:
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))
        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)
        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        return x

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, bias, Kt): 
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.Kt = Kt 
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        x = torch.permute(x, (0, 2, 3, 1)) 
        batch_size, time_step, n_vertex, c_in = x.shape

        Kt = self.Kt  
        truncated_t = time_step - (Kt - 1)
        if truncated_t <= 0:
            raise ValueError(f"时间维度不足，输入时间步 {time_step}，Kt={Kt}")

        if gso.dim() == 2:  
            gso = gso.unsqueeze(0).unsqueeze(0) 
            gso = gso.repeat(batch_size, truncated_t, 1, 1) 
        elif gso.dim() == 3:  
            gso = gso.unsqueeze(0).repeat(batch_size, 1, 1, 1)[:, :truncated_t, :, :]  
        gso = gso.to(x.device).float()

        Kt = self.Kt 
        truncated_t = time_step

        if truncated_t <= 0:
            raise ValueError(f"时间维度不足，输入时间步 {time_step}，Kt={Kt}")

        if self.Ks - 1 < 0:
            raise ValueError(f'图卷积核 Ks 必须为正整数，当前为 {self.Ks}')
        elif self.Ks - 1 == 0:
            x_0 = x[:, :truncated_t, :, :] 
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x[:, :truncated_t, :, :] 
            x_1 = torch.einsum('btnn,btnf->btnf', gso, x[:, :truncated_t, :, :])
            x_list = [x_0, x_1]
        else:  # Ks ≥ 2
            x_0 = x[:, :truncated_t, :, :] 
            x_1 = torch.einsum('btnn,btnf->btnf', gso, x[:, :truncated_t, :, :])
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_k = torch.einsum('btnn,btnf->btnf', 2 * gso, x_list[k - 1]) - x_list[k - 2]
                x_list.append(x_k)


        x = torch.stack(x_list, dim=2)  # [b, t, Ks, n, c_in]

        cheb_graph_conv = torch.einsum('btkni,kij->btnj', x, self.weight) 
        if self.bias is not None:
            cheb_graph_conv = cheb_graph_conv + self.bias

        return cheb_graph_conv.permute(0, 3, 1, 2)

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))

        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)


class TFLMGraphConv(nn.Module):
    """

    """
    def __init__(self, c_in, c_out, Ks=3, Kt=3, bias=True, d_k=15, sym=True, topk=4):

        super(TFLMGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks  
        self.Kt = Kt  
        self.bias = bias
        self.d_k = d_k
        self.sym = sym
        self.topk = topk

        self.pos_channels = 3
        self.feat_channels = c_in - self.pos_channels
        assert self.feat_channels > 0, f"输入通道数c_in={c_in}需>3（前3为位置）"


        self.beta = nn.Parameter(torch.tensor(0.9), requires_grad=True)

        self.pos_embed = nn.Linear(self.pos_channels, 8)  
        self.pos_conv = nn.Conv1d(8, d_k, kernel_size=1)
        self.pos_act = h_swish()


        self.Wq = nn.Linear(self.feat_channels + 8, d_k, bias=False)
        self.Wk = nn.Linear(self.feat_channels + 8, d_k, bias=False)
        self.Wv = nn.Linear(self.feat_channels, d_k, bias=False)
        self.scale = math.sqrt(d_k)


        self.coord_att = nn.Sequential(
            nn.Conv2d(self.pos_channels, d_k//2, kernel_size=1),
            nn.BatchNorm2d(d_k//2),
            h_swish(),
            nn.Conv2d(d_k//2, 1, kernel_size=1),
            h_sigmoid()
        )


        self.weight = nn.Parameter(torch.FloatTensor(d_k, c_out))
        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_param, -bound, bound)

    def _generate_adj(self, X_feat, pos):
        """"""
        B, N, _ = X_feat.shape

        pos_embed = self.pos_embed(pos)  # [B, N, 8]
        pos_feat = self.pos_act(self.pos_conv(pos_embed.transpose(1, 2))).transpose(1, 2)


        X_pos = torch.cat([X_feat, pos_embed], dim=-1)
        Q = self.Wq(X_pos)
        K = self.Wk(X_pos)
        A_feat = torch.matmul(Q, K.transpose(1, 2)) / self.scale


        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        pos_diff = pos_diff.permute(0, 3, 1, 2)  # [B, 3, N, N]
        pos_att = self.coord_att(pos_diff).squeeze(1)  # [B, N, N]


        A = A_feat * pos_att
        A = F.softmax(A, dim=-1)
        if self.sym:
            A = 0.5 * (A + A.transpose(1, 2))
        if self.topk is not None and self.topk < N:
            topk_val, topk_idx = torch.topk(A, self.topk, dim=-1)
            mask = torch.zeros_like(A).scatter_(-1, topk_idx, 1.0)
            A = A * mask
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        return A

    def forward(self, x):
        """"""
        B, C_in, T, N = x.shape


        pos = x[:, :self.pos_channels, :, :]  # [B, 3, T, N]
        x_feat = x[:, self.pos_channels:, :, :]  # [B, feat_channels, T, N]


        x_feat = x_feat.permute(0, 2, 3, 1)  # [B, T, N, feat_channels]
        pos = pos.permute(0, 2, 3, 1)  # [B, T, N, 3]

        out = torch.zeros(B, T, N, self.c_out, device=x.device)
        A_prev_local = None


        for t in range(T):
            X_t_feat = x_feat[:, t, :, :]  # [B, N, feat_channels]
            pos_t = pos[:, t, :, :]  # [B, N, 3]


            A_t = self._generate_adj(X_t_feat, pos_t)


            if A_prev_local is not None:
                A_t = self.beta * A_prev_local + (1 - self.beta) * A_t
            A_prev_local = A_t.detach()


            V = self.Wv(X_t_feat)
            H_t = torch.matmul(A_t, V)
            H_t = torch.matmul(H_t, self.weight)
            if self.bias_param is not None:
                H_t = H_t + self.bias_param
            out[:, t, :, :] = H_t


        return out.permute(0, 3, 1, 2)

class GraphConvLayer(nn.Module):

    def __init__(self, graph_conv_type, c_in, c_out, Ks, Kt, bias):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.align = Align(c_in, c_out)

        if graph_conv_type == 'cheb_graph_conv':
            self.graph_conv = ChebGraphConv(c_out, c_out, Ks, bias, Kt)
        elif graph_conv_type == 'graph_conv':
            self.graph_conv = GraphConv(c_out, c_out, bias)
        elif graph_conv_type == 'tflm_graph_conv':
            self.graph_conv = TFLMGraphConv(c_out, c_out, Ks, Kt, bias)
        else:
            raise NotImplementedError(f"Unknown graph_conv_type: {graph_conv_type}")

    def forward(self, x):
        x_gc_in = self.align(x)
        x_gc = self.graph_conv(x_gc_in)  
        return x_gc + x_gc_in  

class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, bias, droprate):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, Kt, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        #  [B, C_in, T, N]
        x = self.tmp_conv1(x)  
        x = self.graph_conv(x)  
        x = self.relu(x)
        x = self.tmp_conv2(x) 
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x

class FeatureMining(nn.Module):
    def __init__(self, hidden_size=256, layer=3, dropout=0, uav_num=5, feature_num=15):  
        super(FeatureMining, self).__init__()

        self.uav_num = uav_num
        self.feature_num = feature_num  
        self.hidden_size = hidden_size


        Kt = 3 
        Ks = 3  


        self.st_conv_block1 = STConvBlock(
            Kt=Kt,
            Ks=Ks,
            n_vertex=self.uav_num,
            last_block_channel=self.feature_num, 
            channels=[32, 16, 32],  
            act_func='glu',
            graph_conv_type='tflm_graph_conv',
            bias=True,
            droprate=dropout
        )

        self.st_conv_block2 = STConvBlock(
            Kt=Kt,
            Ks=Ks,
            n_vertex=self.uav_num,
            last_block_channel=32,  
            channels=[64, 32, 64],
            act_func='glu',
            graph_conv_type='tflm_graph_conv',
            bias=True,
            droprate=dropout
        )

        self.output_projection = nn.Linear(64 * uav_num, 1089)

    def forward(self, x):
        device = x.device
        batch, seq_len, n_vertex, feature_num = x.shape  # [B, T, N, F=15]

        
        x_st = x.permute(0, 3, 1, 2)  # [B, 15, T, N]


        x1 = self.st_conv_block1(x_st)  
        x2 = self.st_conv_block2(x1)     

        B, C, T, N = x2.shape
        x2 = x2.permute(0, 2, 1, 3).reshape(B, T, C * N)
        x2 = self.output_projection(x2)

        return x2


    def __init__(self):
        super(Vgg16,self).__init__()
        self.input_channel = 3
        self.conv_kernel_size = 3
        self.pool_kernel_size = 2
        self.layer_kernel_num = [64,128,256,512,512]
        self.layer_conv_num = [2,2,3,3,3]
        self.block0 = nn.Sequential(
            nn.Conv2d(self.input_channel,self.layer_kernel_num[0],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[0],self.layer_kernel_num[0],
                      kernel_size=self.conv_kernel_size,padding=1),
            # nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=1),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(self.layer_kernel_num[0],self.layer_kernel_num[1],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[1],self.layer_kernel_num[1],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.layer_kernel_num[1],self.layer_kernel_num[2],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[2],self.layer_kernel_num[2],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[2],self.layer_kernel_num[2],
                      kernel_size=1),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=1),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(self.layer_kernel_num[2],self.layer_kernel_num[3],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[3],self.layer_kernel_num[3],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[3],self.layer_kernel_num[3],
                      kernel_size=1),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=(1)),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(self.layer_kernel_num[3],self.layer_kernel_num[4],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[4],self.layer_kernel_num[4],
                      kernel_size=self.conv_kernel_size,padding=1),
            nn.Conv2d(self.layer_kernel_num[4],self.layer_kernel_num[4],
                      kernel_size=1),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size,stride=(1,2)),
        )

    def forward(self,x):
        batch = x.shape[0]
        block0_y = self.block0(x)
        block1_y = self.block1(block0_y)
        block2_y = self.block2(block1_y)
        block3_y = self.block3(block2_y)
        block4_y = self.block4(block3_y)
        y = block4_y.transpose(1,3)
        y = y.reshape(batch,3,-1)
        return y

class inception_2D(nn.Module):
    def __init__(self, input_channel, cp1, cp2, cp3, cp4):
        super(inception_2D, self).__init__()

        self.p1 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp1[0],
                kernel_size=8,
                stride=16,
            ),
        )

        self.p2 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp2[0],
                kernel_size=4,
                stride=4,
            ),
            nn.Conv1d(
                in_channels=cp2[0],
                out_channels=cp2[1],
                kernel_size=6,
                stride=4,
                dilation=1,
                padding=1,
            ),
        )

        self.p3 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp3[0],
                kernel_size=8,
                stride=4,
            ),
            nn.Conv1d(
                in_channels=cp3[0],
                out_channels=cp3[1],
                kernel_size=4,
                stride=4,
                padding=1,
            ),
        )

        self.p4 = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=3,
                stride=4,
            ),
            nn.Conv1d(
                input_channel,
                out_channels=cp4[0],
                kernel_size=10,
                stride=4,
                padding=3,
            ),
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        # len(shape) == 3 表示此时第一维是batch
        if len(p1.shape) == 3:
            return F.relu(torch.concat([p1, p2, p3, p4], dim=1))
        # len(shape) == 2 表示第一维是channel，且不存在batch维
        elif len(p1.shape) == 2:
            return F.relu(torch.concat([p1, p2, p3, p4], dim=0))
        return None

class inception_1D(nn.Module):
    def __init__(self, input_channel, cp1, cp2, cp3, cp4):
        super(inception_1D, self).__init__()

        self.p1 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp1[0],
                kernel_size=3,
                stride=2,
            ),
        )

        self.p2 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp2[0],
                kernel_size=1,
                stride=1,
            ),
            nn.Conv1d(
                in_channels=cp2[0],
                out_channels=cp2[1],
                kernel_size=3,
                stride=2,
                dilation=2,
                padding=1,
            ),
        )

        self.p3 = nn.Sequential(
            nn.Conv1d(
                input_channel,
                out_channels=cp3[0],
                kernel_size=1,
                stride=1,
            ),
            nn.Conv1d(
                in_channels=cp3[0],
                out_channels=cp3[1],
                kernel_size=5,
                stride=2,
                padding=1,
            ),
        )

        self.p4 = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=2,
                stride=1,
            ),
            nn.Conv1d(
                input_channel,
                out_channels=cp4[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        if len(p1.shape) == 3:
            return F.relu(torch.concat([p1, p2, p3, p4], dim=1))
        elif len(p1.shape) == 2:
            return F.relu(torch.concat([p1, p2, p3, p4], dim=0))
        return None


class InceptionDWConv1d(nn.Module):
    """

    """

    def __init__(self, in_channels, small_kernel_size=3, mid_kernel_size=7, large_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        assert in_channels - 3 * gc > 0, f"in_channels ({in_channels}) too small for branch_ratio {branch_ratio}"


        self.dwconv_s = nn.Conv1d(gc, gc, kernel_size=small_kernel_size, padding=small_kernel_size // 2, groups=gc)
        self.dwconv_m = nn.Conv1d(gc, gc, kernel_size=mid_kernel_size, padding=mid_kernel_size // 2, groups=gc)
        self.dwconv_l = nn.Conv1d(gc, gc, kernel_size=large_kernel_size, padding=large_kernel_size // 2, groups=gc)


        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)


        self.mix_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_id, x_s, x_m, x_l = torch.split(x, self.split_indexes, dim=1)
        y_s = self.dwconv_s(x_s)
        y_m = self.dwconv_m(x_m)
        y_l = self.dwconv_l(x_l)
        out = torch.cat([x_id, y_s, y_m, y_l], dim=1)
        out = self.mix_conv(out)
        out = self.bn(out)
        return self.act(out)

class MultiTask(nn.Module):
    """

    """

    def __init__(self, input_channel, uav_num, dropout=(0, 0, 0, 0)):
        super(MultiTask, self).__init__()
        self.uav_num = uav_num
        self.dropout = dropout

        # ===== Block 1 =====
        self.block1 = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            InceptionDWConv1d(8),
        )
        self.output1_pool = nn.AvgPool1d(kernel_size=3, stride=1)
        self.output1_linear = nn.Linear(8680, uav_num)

        # ===== Block 2 =====
        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            InceptionDWConv1d(8),
        )
        self.output2_pool = nn.AvgPool1d(kernel_size=3, stride=1)
        self.output2_linear = nn.Linear(8664, uav_num)

        # ===== Block 3 =====
        self.block3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            InceptionDWConv1d(8),
        )
        self.output3_pool = nn.AvgPool1d(kernel_size=3, stride=1)
        self.output3_linear = nn.Linear(8664, uav_num)

        # ===== Block 4 =====
        self.block4 = nn.Sequential(
            InceptionDWConv1d(8),
            nn.ReLU(),
        )
        self.output4_flat = nn.Flatten()
        self.output4_layers = nn.ModuleList([
            nn.Linear(8680, 2) for _ in range(uav_num)
        ])


        self._initialize_weights()

        self.freeze_linear_layers = True

    def _initialize_weights(self):

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
         
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def set_linear_requires_grad(self, requires_grad):
        self.output1_linear.requires_grad_(requires_grad)
        self.output2_linear.requires_grad_(requires_grad)
        self.output3_linear.requires_grad_(requires_grad)
        for layer in self.output4_layers:
            layer.requires_grad_(requires_grad)

    def unfreeze_linear_layers(self):
        self.freeze_linear_layers = False
        self.set_linear_requires_grad(True)

    def forward(self, x):
        if self.training:
            self.set_linear_requires_grad(not self.freeze_linear_layers)

        # ===== Block 1 =====
        x_b1 = self.block1(x)
        x1 = self.output1_pool(x_b1)
        x1 = torch.flatten(x1, 1)
        out1 = torch.sigmoid(self.output1_linear(x1))

        # ===== Block 2 =====
        x_b2 = self.block2(x_b1)
        x2 = self.output2_pool(x_b2)
        x2 = torch.flatten(x2, 1)
        out2 = torch.sigmoid(self.output2_linear(x2))

        # ===== Block 3 =====
        x_b3 = self.block3(x_b1)
        x3 = self.output3_pool(x_b3)
        x3 = torch.flatten(x3, 1)
        out3 = torch.sigmoid(self.output3_linear(x3))

        # ===== Block 4 =====
        x_b4 = self.block4(x_b3)
        out4_flat = self.output4_flat(x_b4)
        control_outputs = [layer(out4_flat) for layer in self.output4_layers]
        out4 = torch.stack(control_outputs, dim=1)

        return out1, out2, out3, out4

