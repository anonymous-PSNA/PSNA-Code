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
            #     configs.n_heads, configs.d_ff, self.k, configs.activation
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

        x_reshaped = x.permute(0, 2, 1, 3).reshape(B, patch_num, L * d_feat)
            # [B, patch_num, L * d_feat]

        batch_size, num_vars, input_len = x_reshaped.shape

        mean = x_reshaped.mean(dim=2, keepdim=True)
            # [B, patch_num, 1]
        centered_data = x_reshaped - mean
            # [batch_size , num_vars , input_len]


        cov_matrix = torch.bmm(centered_data, centered_data.transpose(1, 2)) / (input_len - 1 + 1e-5)
            # [B,num_vars, num_vars]
        std_dev = torch.sqrt(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))
            #  [B,num_vars]
        std_dev[std_dev == 0] = 1e-5

        std_dev_matrix = std_dev.unsqueeze(2) * std_dev.unsqueeze(1)
        correlation_matrix = cov_matrix / std_dev_matrix
            # [B ,num_vars , num_vars]
        return correlation_matrix

    def edge_index(self, x):
        # x: [B, L, patch_num, d_model//2]

        similarity_matrix = self.person_correlation(x)
                #[B, patch_num, patch_num]

        k = self.k
        neighbors = torch.argsort(similarity_matrix, dim=-1)[:, :, 1:k + 1]
                # [batch, patch_num, k]

        batch_size, num_nodes = similarity_matrix.shape[:2]
        row_indices = torch.arange(num_nodes, device=x.device).repeat(k).reshape(1, -1).repeat(batch_size, 1)
                # [batch, patch_num * k]

        col_indices = neighbors.reshape(batch_size, -1)
                # [batch, patch_num * k]

        edge_index = torch.stack((row_indices, col_indices), dim=1)
                # [batch, 2, patch_num * k]

        edge_index = edge_index.long().to(x.device)
            # [batch, 2, patch_num * k]
        return edge_index

    def get_batch_edge_index(self, x):
        adj_matrix = self.person_correlation(x)
            # B,N,N
        B, N, _ = adj_matrix.shape
        device = x.device

        topk_indices = torch.topk(adj_matrix, k=self.k, dim=-1).indices
            # B, N, K

        target_nodes = torch.arange(N, device=device).view(1, N, 1).expand(B, N, self.k)

        source_nodes = topk_indices

        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * N

        target_nodes_flat = (target_nodes + batch_offset).flatten()
        source_nodes_flat = (source_nodes + batch_offset).flatten()

        edge_index = torch.stack([source_nodes_flat, target_nodes_flat], dim=0)
        return edge_index

    def forward(self, enc_out_vari_embeding, x_enc, enc_in):
        B, L, N, D = enc_in.shape

        edge_index = self.get_batch_edge_index(enc_in)

        x_gcn_input = enc_out_vari_embeding.reshape(B * N, D)

        x_gcn_out = self.gcn_model(x_gcn_input, edge_index)

        x_gcn_out = x_gcn_out.reshape(B, N, D)

        # x_gcn_out: [B, N, D] -> unsqueeze -> [B, 1, N, D]
        # enc_in:    [B, L, N, D]
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
        edge_index_cpu = edge_index.detach().cpu()

        mask = (edge_index_cpu[0] < num_nodes) & (edge_index_cpu[1] < num_nodes)
        sample_edges = edge_index_cpu[:, mask]

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        edges_list = sample_edges.t().tolist()
        G.add_edges_from(edges_list)

        plt.figure(figsize=(8, 6))
        plt.clf()

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
        plt.close()
        # print(f'Graph structure saved to {filename}')
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        edge_index_with_self_loops = self.add_self_loops(edge_index, num_nodes)

        row, col = edge_index_with_self_loops
        deg = self.compute_degree(row, num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index_with_self_loops, norm, (num_nodes, num_nodes))

        out = torch.sparse.mm(adj, x)
        out = torch.matmul(out, self.weight)
        return out

    def add_self_loops(self, edge_index, num_nodes):
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index_with_self_loops = torch.cat([edge_index, loop_index], dim=1)
        return edge_index_with_self_loops

    def compute_degree(self, row, num_nodes):
        deg = torch.zeros(num_nodes, dtype=torch.float, device=row.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=row.device))
        return deg
