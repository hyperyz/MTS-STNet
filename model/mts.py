import torch.nn.functional as F
import torch.nn as nn
import torch

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shape: [B, T, N, F]  -> [Batch, Time, Node, Feature]
        # node_embeddings shape: [N, D] -> [Node, Embedding dimension]

        B, T, N, _ = x.shape  # Extract batch size, time steps, number of nodes, feature size
        node_num = node_embeddings.shape[0]

        # Compute support matrix: [N, N] (Adjacency matrix)
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1).contiguous())), dim=1)  # [N, N]
        support_set = [torch.eye(node_num).to(supports.device), supports]

        # Default cheb_k = 3, computing higher-order Chebyshev polynomials
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        supports = torch.stack(support_set, dim=0)  # [cheb_k, N, N]

        # Generate weights and bias: [cheb_k, dim_in, dim_out]
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # [N, cheb_k, dim_in, dim_out]
        bias = torch.matmul(node_embeddings, self.bias_pool)  # [N, dim_out]

        x_g_list = []
        for t in range(T):
            x_t = x[:, t, :, :]  # Shape: [B, N, F] for the t-th time step

            # Chebyshev polynomial expansion: x_g shape: [B, cheb_k, N, dim_in]
            x_g = torch.einsum("knm,bmc->bknc", supports, x_t)  # [B, cheb_k, N, F]
            x_g = x_g.permute(0, 2, 1, 3).contiguous() # [B, N, cheb_k, F]

            # Apply graph convolution
            x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # [B, N, dim_out]
            x_g_list.append(x_gconv.unsqueeze(1))  # Add the output for this time step

        # Concatenate results across time steps: [B, T, N, dim_out]
        output = torch.cat(x_g_list, dim=1)

        return output


class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=d_series, out_channels=d_series, kernel_size=(1, 1),
                               bias=True)  # [batch_size, time_steps, num_nodes, d_series]
        self.conv2 = nn.Conv2d(in_channels=d_series, out_channels=d_core, kernel_size=(1, 1),
                               bias=True)  # [batch_size, time_steps, num_nodes, d_core]

        self.conv3 = nn.Conv2d(in_channels=d_series+int(d_core/2), out_channels=d_series, kernel_size=(1, 1),
                               bias=True)  # [batch_size, time_steps, num_nodes, d_series]
        self.conv4 = nn.Conv2d(in_channels=d_series, out_channels=d_series, kernel_size=(1, 1),
                               bias=True)  # [batch_size, time_steps, num_nodes, d_series]

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        for i in range(2):
            new_dilation = 1
            self.filter_convs.append(nn.Conv2d(in_channels=12,  # 输入的通道数是d_series
                                               out_channels=12,
                                               kernel_size=(int(d_core/2)+1, 1), dilation=new_dilation))
            self.gate_convs.append(nn.Conv2d(in_channels=12,
                                             out_channels=12,
                                             kernel_size=(int(d_core/2)+1, 1), dilation=new_dilation))
            new_dilation *= 2

    def forward(self, input):
        # 输入形状 [batch_size, time_steps, num_nodes, channels]
        combined_mean = F.gelu(self.conv1(input))  # [batch_size, d_series, time_steps, num_nodes]
        combined_mean = self.conv2(combined_mean)  # [64, 128, 12, 170]

        residual = combined_mean.permute(0, 2, 1, 3).contiguous()
        # print(residual.shape)  # [64, 12, 128, 170]
        for i in range(2):
            filter = self.filter_convs[i](residual)  # [batch_size, d_series, time_steps, num_nodes]
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)  # [batch_size, d_series, time_steps, num_nodes]
            gate = torch.sigmoid(gate)
            combined_mean = filter * gate  # element-wise multiplication

        combined_mean = combined_mean.permute(0, 2, 1, 3).contiguous()

        # print(input.shape, combined_mean.shape)  # [64, 64, 12, 170] [64, 64, 12, 170]

        combined_mean_cat = torch.cat([input, combined_mean],
                                      dim=1)  # [batch_size, d_series+d_core, time_steps, num_nodes]
        combined_mean_cat = F.gelu(self.conv3(combined_mean_cat))  # [batch_size, d_series, time_steps, num_nodes]
        combined_mean_cat = self.conv4(combined_mean_cat)  # [batch_size, d_series, time_steps, num_nodes]

        return combined_mean_cat


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class MTS(nn.Module):
    def __init__(self, num_node, input_dim, output_dim, horizon, ad_layers, agc_layers, embed_dim, hidden_dim,
                 time_embed, core_dim, cheb_k,predict_step):
        super(MTS, self).__init__()
        self.num_node = num_node
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.agc_layers = agc_layers
        self.ad_layers = ad_layers
        self.num_layer = 3
        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True
        self.predict_step = predict_step

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(288, time_embed))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, time_embed))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.node_embeddings = nn.Parameter(torch.randn(num_node, embed_dim))  # [170, 2]
        self.conv = AVWGCN(input_dim, hidden_dim, cheb_k, embed_dim)

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AVWGCN(input_dim, hidden_dim, cheb_k, embed_dim))
        for _ in range(1, agc_layers):
            self.dcrnn_cells.append(AVWGCN(hidden_dim, hidden_dim, cheb_k, embed_dim))

        self.stars = nn.ModuleList()
        for _ in range(ad_layers):
            self.stars.append(STAR(d_series=hidden_dim, d_core=core_dim))

        self.cc = nn.Conv2d(in_channels=self.input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=(1, 1), bias=True)
        self.time_series_emb_layer2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=(1, 1), bias=True)

        self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        ddd = time_embed * 2 + hidden_dim * 2
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(ddd, ddd) for _ in range(self.num_layer)])
        self.regression_layer = nn.Conv2d(
            in_channels=ddd, out_channels=predict_step, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        # x: B, T_1, N, D
        # target: B, T_2, N, D

        if self.if_time_in_day:
            t_i_d_data = x[..., self.input_dim]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]  # [64, 170, 32]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = x[..., self.input_dim + 1]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * 7).type(torch.LongTensor)]  # [64, 170, 32]
        else:
            day_in_week_emb = None

        # 时间嵌入
        tem_emb = []  # 2*[64, 32, 170, 1]
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1).contiguous())
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1).contiguous())

        input_data = x[..., range(self.input_dim)]  # [64, 12, 170, 3]

        # 图卷积
        output = self.dcrnn_cells[0](input_data, self.node_embeddings)  # [64, 12, 170, 64]
        for i in range(1, self.agc_layers):
            output = self.dcrnn_cells[i](output, self.node_embeddings)  # [64, 12, 170, 64]

        input_data = input_data.permute(0, 3, 1, 2).contiguous()
        # STAR 模块
        output2 = self.cc(input_data)
        for star in self.stars:
            output2 = star(output2)  # [64, 64, 12, 170]

        output2 = output2.permute(0, 1, 3, 2).contiguous() # [64, 64, 170, 12]
        output2 = output2[:, :, :, -1:]  # [64, 64, 170, 1]

        output = output.permute(0, 3, 2, 1).contiguous()
        output = output[:, :, :, -1:]

        hidden = torch.cat([output] + [output2] + tem_emb, dim=1)

        hidden = self.encoder(hidden)  # [64, 128, 170, 1]
        # 预测
        output = self.regression_layer(hidden)
        return output


def mts(num_node, input_dim, ad_layers=1, agc_layers=1, node_embed_dim=5, hidden_dim=64, core_dim=128, predict_step=12):
    model = MTS(num_node=num_node, input_dim=input_dim, output_dim=1, horizon=12, ad_layers=ad_layers, agc_layers=agc_layers,
                embed_dim=node_embed_dim, time_embed=32,
                hidden_dim=hidden_dim, core_dim=core_dim,
                cheb_k=2,predict_step=predict_step)
    return model
