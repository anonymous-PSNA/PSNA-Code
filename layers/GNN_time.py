import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from layers.Transformer_encoder import TransformerEncoder
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

class MultiLayerGCN_time(nn.Module):
            # self.GNN_encoder_time = MultiLayerGCN_time(
            #     configs.e_layers, self.d_model // 2, configs.dropout,
            #     #图卷积神经网络的层数，输入到图的维度，dropout概率
            #     configs.n_heads, configs.d_ff, self.k, configs.activation
            #     #注意力头的数量，前馈神经网络层的维度，k的个数，激活函数
            # ).cuda()
    def __init__(self, num_layers, d_model, dropout, n_heads, d_ff,
                 k, activation,hidden_channels,
                 enc_weight,gcn_weight,broad_cast):
        super(MultiLayerGCN_time, self).__init__()
        self.layers = nn.ModuleList()
        self.gcn_model = GCN(
            in_channels=d_model,
            #hidden_channels=d_model,
            hidden_channels=hidden_channels,
            out_channels=d_model,
            dropout=dropout,
            activation=activation,
            n_heads=n_heads,
            d_ff=d_ff,
            broad_cast=broad_cast
        )
        self.d_model = d_model
        self.k = k
        self.enc_weight = enc_weight
        self.gcn_weight = gcn_weight


    def person_correlation(self, x):
        # x: [B, L, patch_num, d_model//2]
        B, L, patch_num, d_feat = x.size()

        # 1. 重塑数据：将时间和特征维度展平，作为用于计算相关性的“序列长度”
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B, patch_num, L * d_feat)
            # [B, patch_num, L * d_feat]
            # 这一步将每个变量的所有时间步和特征维度展平为一个长序列，目的是计算跨越整个序列的变量间相关性。

        batch_size, num_vars, input_len = x_reshaped.shape

        # 计算均值
        mean = x_reshaped.mean(dim=2, keepdim=True)
            # [B, patch_num, 1]
        # 中心化数据
        centered_data = x_reshaped - mean
            # [batch_size , num_vars , input_len]


        # 计算协方差矩阵，使用批量矩阵乘法 (协方差矩阵用来衡量两个变量共同变化的趋势）
        cov_matrix = torch.bmm(centered_data, centered_data.transpose(1, 2)) / (input_len - 1 + 1e-5)
            # [B,num_vars, num_vars]
        # 计算标准差 （协方差矩阵没有上限，用于归一化协方差矩阵）
        std_dev = torch.sqrt(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))
            #  [B,num_vars]
        # 防止标准差为0的情况，将0替换为1
        std_dev[std_dev == 0] = 1e-5 # 使用小量防止除零

        # 计算相关系数矩阵 (协方差除以两个标准差变量乘积）
        std_dev_matrix = std_dev.unsqueeze(2) * std_dev.unsqueeze(1)
        correlation_matrix = cov_matrix / std_dev_matrix
            # [B ,num_vars , num_vars]
        return correlation_matrix

    def edge_index(self, x):
        # x: [B, L, patch_num, d_model//2]

        # 皮尔逊相关系数
        similarity_matrix = self.person_correlation(x)
                #[B, patch_num, patch_num]

        # 选择k个最近邻居
        k = self.k
        neighbors = torch.argsort(similarity_matrix, dim=-1)[:, :, 1:k + 1]
                # [batch, patch_num, k]

        # 生成行索引
        batch_size, num_nodes = similarity_matrix.shape[:2]
        row_indices = torch.arange(num_nodes, device=x.device).repeat(k).reshape(1, -1).repeat(batch_size, 1)
                # [batch, patch_num * k]

        # 生成列索引
        col_indices = neighbors.reshape(batch_size, -1)
                # [batch, patch_num * k]

        # 堆叠生成edge_index
        edge_index = torch.stack((row_indices, col_indices), dim=1)
                # [batch, 2, patch_num * k]

        # 确保edge_index在CUDA上
        edge_index = edge_index.long().to(x.device)
            # [batch, 2, patch_num * k]
            # 2 是指源节点和目标节点
        return edge_index

    def get_batch_edge_index(self, x):
        """
        x: [B, L, patch_num, d_model] 用于计算相关性
        """
        # 1. 计算相关性矩阵 [B, N, N]
        adj_matrix = self.person_correlation(x)
            # B,N,N
        B, N, _ = adj_matrix.shape
        device = x.device

        topk_indices = torch.topk(adj_matrix, k=self.k, dim=-1).indices
            # B, N, K
            # 挑前k个最相关的节点

        # 3. 构建 edge_index (需要加上 Batch 的偏移量)
        # 目标节点 (Row): [B, N, K]
        target_nodes = torch.arange(N, device=device).view(1, N, 1).expand(B, N, self.k)

        # 源节点 (Col): [B, N, K]
        source_nodes = topk_indices

        # 4. 计算 Batch 偏移量: [0, N, 2N, ..., (B-1)N]
        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * N

        # 5. 加上偏移量并展平
        target_nodes_flat = (target_nodes + batch_offset).flatten()
        source_nodes_flat = (source_nodes + batch_offset).flatten()

        # [2, B*N*K] -> 符合 PyG 的 edge_index 格式
        edge_index = torch.stack([source_nodes_flat, target_nodes_flat], dim=0)
            # 第一行是起点，第二行是终点
        return edge_index

    def forward(self, enc_out_vari_embeding, x_enc, enc_in):
        """
        enc_out_vari_embeding: [B, patch_num, d_model] (静态节点特征)
        enc_in: [B, L, patch_num, d_model] (全序列特征)
        """
        B, L, N, D = enc_in.shape

        edge_index = self.get_batch_edge_index(enc_in)
            # [2, B*N*K] 得到图的关系

            # 2. 准备 GCN 输入
        # GCN 需要输入 [ B,patch_num,d_model ] -> [B*N, D]
        x_gcn_input = enc_out_vari_embeding.reshape(B * N, D)

        # 3. GCN 前向传播 (纯空间交互)
        x_gcn_out = self.gcn_model(x_gcn_input, edge_index)

        # 4. 还原维度 [B*N, D] -> [B, N, D]
        x_gcn_out = x_gcn_out.reshape(B, N, D)

        # 5. 关键步骤：广播回时间维度
        # x_gcn_out: [B, N, D] -> unsqueeze -> [B, 1, N, D]
        # enc_in:    [B, L, N, D]
        # 两者相加，不仅保留了原始时间信息，还融入了变量间的空间依赖
        # output = enc_in + x_gcn_out.unsqueeze(1)
        output = (
                self.enc_weight * enc_in
                + self.gcn_weight * x_gcn_out.unsqueeze(1)
        )


        return output


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout, activation,n_heads,d_ff,num_layers=1,broad_cast=0):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.Transformer_encoder_time = TransformerEncoder(in_channels, n_heads, num_layers, d_ff, dropout).cuda()
        self.broadcast = broad_cast

        self.norm = nn.LayerNorm(out_channels)

        if activation == 'sigmoid':
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()

    def forward(self, x, edge_index):
        # x: [B * N, D]
        # edge_index: [2, E] # [2, B*N*K]

        # 保存输入用于残差
        x_in = x

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.activate(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.activate(x)
        x = self.dropout(x)

        # # Residual + Norm
        # x = self.norm(x + x_in)
        if self.broadcast:
            x = x+x_in
            BN,D = x.shape
            N = edge_index.max().item()+1
            B=BN//N

            x_node=x.view(B,N,D)
            x_node = self.Transformer_encoder_time(
                query=x_node,
                value=x_node,
                mask=None
            )
            x = x_node.reshape(BN,D)
        else:
            x = self.norm(x + x_in)


        return x

    def visual_jiedian(self, edge_index, num_nodes, filename='structure.png'):
        """
        可视化图结构。
        注意：因为现在的 edge_index 包含整个 Batch 的所有边，
        这里只截取【第一个样本】的边进行可视化，否则图会太乱。

        参数:
            edge_index: [2, B*N*K] 当前batch的所有边
            num_nodes: int, 变量个数 (patch_num)
        """
        # 1. 转移到 CPU 并转为 numpy/list
        edge_index_cpu = edge_index.detach().cpu()

        # 2. 筛选属于第一个样本的边
        # 原理：第一个样本的节点索引范围是 [0, num_nodes-1]
        # 只要源节点和目标节点都小于 num_nodes，就属于第一个样本
        mask = (edge_index_cpu[0] < num_nodes) & (edge_index_cpu[1] < num_nodes)
        sample_edges = edge_index_cpu[:, mask]

        # 3. NetworkX 建图
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes)) # 添加所有变量节点

        # 转为 list: [(u1, v1), (u2, v2), ...]
        edges_list = sample_edges.t().tolist()
        G.add_edges_from(edges_list)

        # 4. 绘图
        plt.figure(figsize=(8, 6)) # 设置画布大小
        plt.clf()

        # 布局算法：spring_layout 适合看聚类，circular_layout 适合看连接关系
        pos = nx.spring_layout(G, seed=42)

        nx.draw(G, pos,
                with_labels=True,
                node_color="skyblue",
                node_size=600,
                edge_color="gray",
                width=1.5,
                font_size=12,
                font_weight='bold')

        plt.title(f"Variable Spatial Dependency (First Sample)", fontsize=15)
        plt.savefig(filename)
        # plt.show() # 训练过程中建议注释掉 show，只保存图片
        plt.close() # 关闭画布释放内存
        # print(f'Graph structure saved to {filename}')
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        # 定义可学习的权重矩阵 W
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x, edge_index):
        # Step 1: 添加自环
        num_nodes = x.size(0)
        edge_index_with_self_loops = self.add_self_loops(edge_index, num_nodes)

        # Step 2: 计算度矩阵 D
        row, col = edge_index_with_self_loops
        deg = self.compute_degree(row, num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 避免分母为0的情况

        # Step 3: 计算 A_hat = D^{-1/2} * A * D^{-1/2}，并构建稀疏矩阵
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index_with_self_loops, norm, (num_nodes, num_nodes))

        # Step 4: 执行图卷积 H^{l+1} = A_hat * X * W
        out = torch.sparse.mm(adj, x)  # 使用稀疏矩阵乘法加速消息传递
        out = torch.matmul(out, self.weight)  # 线性变换
        return out

    def add_self_loops(self, edge_index, num_nodes):
        """为每个节点添加自环"""
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index_with_self_loops = torch.cat([edge_index, loop_index], dim=1)
        return edge_index_with_self_loops

    def compute_degree(self, row, num_nodes):
        """计算度矩阵 D"""
        deg = torch.zeros(num_nodes, dtype=torch.float, device=row.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=row.device))
        return deg
