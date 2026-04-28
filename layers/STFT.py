
from layers.DeepMemoryModule import PhaseAwareUNet2D
import torch
import torch.nn as nn
class STFTBranch(nn.Module):
    def __init__(self, input_len, pred_len, num_vars, nfft_min=32, n_stft_branch=2, hidden_dim=1,dropout=0.3):
        super(STFTBranch, self).__init__()
        self.pred_len = pred_len
        self.num_vars = num_vars
        self.n_stft_branch = n_stft_branch
        self.hidden_dim = hidden_dim

        self.branch_modules = nn.ModuleList()

        for n_branch in range(n_stft_branch):
            curr_nfft = min(nfft_min * (2 ** n_branch), input_len)
            if curr_nfft % 2 != 0: curr_nfft -= 1
            self.branch_modules.append(
                BranchModule(input_len, pred_len, curr_nfft)
            )


        self.flatten_dim = input_len * hidden_dim * n_stft_branch
        self.projection = nn.Linear(self.flatten_dim, pred_len)
        self.trend_projection = nn.Linear(input_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        BN, input_len, C = x.shape

        outputs = []

        for module in self.branch_modules:
            outputs.append(module(x))

        stack_tensor = torch.stack(outputs, dim=1)
        feat_flatten = stack_tensor.view(BN, -1)
        season_prediction = self.projection(self.dropout(feat_flatten))


        x_flat = x.squeeze(-1)
        trend_prediction = self.trend_projection(x_flat)

        return season_prediction + trend_prediction

class BranchModule(nn.Module):
    def __init__(self, input_len, pred_len, feat_dim_long=32):
        super(BranchModule, self).__init__()
        self.feat_dim_long = feat_dim_long
        self.PhaseAwareU = PhaseAwareUNet2D(base_dim=32)

    def forward(self, x):
        BN, L, C = x.shape
        x_flat = x.squeeze(-1)

        window = torch.hann_window(self.feat_dim_long).to(x.device)

        stft_res = torch.stft(x_flat, n_fft=self.feat_dim_long, hop_length=self.feat_dim_long//4,
                              window=window, return_complex=True, center=True)

        stft_res = torch.view_as_real(stft_res)

        mem_out = self.PhaseAwareU(stft_res.permute(0, 3, 1, 2))
        istft_input = mem_out.permute(0, 2, 3, 1)
        istft_complex = torch.complex(istft_input[..., 0], istft_input[..., 1])
        rec = torch.istft(istft_complex, n_fft=self.feat_dim_long, hop_length=self.feat_dim_long//4,
                          window=window, length=L, center=True)

        return rec.unsqueeze(-1)