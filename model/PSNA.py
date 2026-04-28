import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.RevIN import RevIN
from layers.GNN_variate import MultiLayerGCN_variate
from layers.GNN_time import MultiLayerGCN_time
from layers.STFT import STFTBranch

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        #基础参数的设置
        self.seq_len = configs.seq_len
        self.input_len = configs.seq_len
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.num_vars = configs.enc_in
        self.use_norm = configs.use_norm
        self.batch = configs.batch_size
        self.use_norm = configs.use_norm
        self.k = configs.k
        if configs.activation == 'sigmoid':
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()

        #################
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.embedding_patch = nn.Linear(self.patch_len, configs.d_model // 2)
        self.padding_patch = configs.padding_patch

        context_window = configs.enc_in
        self.patch_num = int((context_window - self.patch_len)/self.stride + 1)

        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1
        self.seq_pred = nn.Linear(self.patch_num * (configs.d_model // 2), configs.enc_in,bias=True)
        self.seq_len = self.patch_num

        self.GNN_encoder_time = MultiLayerGCN_time(
            configs.e_layers, self.d_model // 2, configs.dropout,
            configs.n_heads, configs.d_ff, self.k, configs.activation,configs.hidden_dims,
            configs.enc_weight, configs.gcn_weight,
            configs.broadcast
        ).cuda()
        self.use_spatial_refinement = False
        self.project=nn.Linear(self.patch_num*(configs.d_model//2), configs.d_model)
        self.patch_fuse = nn.Linear(self.patch_len, 1)


        revin = True
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.num_vars, affine=True, subtract_last=True)

        self.GNN_encoder = MultiLayerGCN_variate(
            configs.e_layers, self.d_model, configs.dropout,
            configs.n_heads, configs.d_ff, self.k, configs.activation
        ).cuda()
        self.flatten = nn.Flatten(start_dim=-2)


        self.FC = nn.Linear(configs.d_model * 1, configs.pred_len)
        self.FC2 = nn.Linear(configs.d_model * 1, configs.enc_in)
        self.FC3 = nn.Linear(configs.pred_len * 2, configs.pred_len)
        self.FC4 = nn.Linear(configs.seq_len * 1, configs.pred_len)


    #时间维度分支
        self.nfft_min = configs.nfft_min
        self.hidden_dims = configs.hidden_dims
        self.n_stft_branch=configs.n_stft_branch

        self.branch_a = STFTBranch(self.input_len, self.pred_len, self.num_vars,self.nfft_min,self.n_stft_branch,1,configs.dropout)

        self.fusion_layer = nn.Linear(2, 1)
        self.embedding = nn.Linear(1, self.hidden_dims)



    def Embed_stft(self, x):
        B, N, L = x.shape
        return x.reshape(B * N, L, 1)

    def Embedding_patch(self, seasonal_init):
        if self.padding_patch == 'end':
            x_enc_patch_row = self.padding_patch_layer(seasonal_init)

        x_enc_patch_row = x_enc_patch_row.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        x_enc_patch = self.embedding_patch(x_enc_patch_row)  # batch multivariate patch_num 128


        return x_enc_patch,x_enc_patch_row

    def Channel_independence(self, enc_in, x_enc, B, x_enc_patch_row):

        enc_out_in = torch.mean(enc_in, dim=1, keepdim=False)
        dec_out_time = self.GNN_encoder_time(enc_out_in, enc_out_in, enc_in)
        enc_out_in = self.flatten(dec_out_time)
        dec_out_time = self.seq_pred(enc_out_in)

        return dec_out_time
        dec_out_time = self.GNN_encoder_time(enc_out_in, enc_out_in, enc_in)

        enc_out_in = self.flatten(dec_out_time)
        dec_out_time = self.seq_pred(enc_out_in)
        return dec_out_time
    def forecast(self, x_enc,x_mark_enc= None):
        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')
        B, _, N = x_enc.shape
        enc_out=self.Embed_stft(x_enc.permute(0, 2, 1))
        out_a = self.branch_a(enc_out)
        out_a=out_a.reshape(B, N, -1)
        enc_out_in,x_enc_patch_row = self.Embedding_patch(x_enc)
        dec_out_vari = self.Channel_independence(enc_out_in, x_enc, B, x_enc_patch_row)
        dec_out_vari=dec_out_vari.permute(0, 2, 1)
        if self.pred_len != self.seq_len:
            dec_out_vari = self.FC4(dec_out_vari)
        enc_out_concat = torch.cat((out_a, dec_out_vari), dim=-1)
        dec_out = self.FC3(enc_out_concat)
        if self.use_norm:
            dec_out_all = self.revin_layer(dec_out.transpose(2, 1), 'denorm')
        else:
            dec_out_all = dec_out.transpose(2, 1)

        return dec_out_all


    def forward(self, x_enc,  x_mark_enc, dec_in=None, y_mark=None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]