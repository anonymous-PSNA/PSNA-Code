#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 12; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/  \
          --data_path PEMS08.npz \
          --model_id PEMS08_96_96_$p \
          --model $model_name \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $p \
          --e_layers 2 \
          --enc_in 170 \
          --num_vars 170 \
          --des 'Exp' \
          --d_model 64 \
          --n_heads 4 \
          --d_ff 64 \
          --batch_size 8 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 20 \
          --k 2 \
          --activation sigmoid \
          --patch_len 5 \
          --stride 2 \
          --padding_patch end \
          --nfft_min 8 \
          --hidden_dims 64 \
          --n_stft_branch 2 \
          --patience 3 \
          --broadcast 1 \
          --enc_weight 0.0 \
          --gcn_weight 1.0
done
#DFGCN 0.079
#mse:0.08426817506551743, mae:0.18877963721752167, rse:0.2661271393299103
#改static_adj_path之前
#mse:0.08062729239463806, mae:0.18118688464164734, rse:0.26031455397605896
export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 24; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/  \
          --data_path PEMS08.npz \
          --model_id PEMS08_96_96_$p \
          --model $model_name \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $p \
          --e_layers 2 \
          --enc_in 170 \
          --num_vars 170 \
          --des 'Exp' \
          --d_model 64 \
          --n_heads 4 \
          --d_ff 64 \
          --batch_size 8 \
          --dropout 0.1 \
          --learning_rate 0.003 \
          --train_epochs 20 \
          --k 2 \
          --activation sigmoid \
          --patch_len 20 \
          --stride 5 \
          --padding_patch end \
          --nfft_min 16 \
          --hidden_dims 64 \
          --n_stft_branch 2 \
          --patience 3 \
          --broadcast 1 \
          --enc_weight 0.0 \
          --gcn_weight 1.0
done
#mse:0.11772744357585907, mae:0.21862094104290009, rse:0.31444138288497925
export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 48; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/  \
          --data_path PEMS08.npz \
          --model_id PEMS08_96_96_$p \
          --model $model_name \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $p \
          --e_layers 2 \
          --enc_in 170 \
          --num_vars 170 \
          --des 'Exp' \
          --d_model 64 \
          --n_heads 4 \
          --d_ff 64 \
          --batch_size 8 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 20 \
          --k 2 \
          --activation sigmoid \
          --patch_len 20 \
          --stride 5 \
          --padding_patch end \
          --nfft_min 16 \
          --hidden_dims 64 \
          --n_stft_branch 2 \
          --patience 3 \
          --broadcast 1 \
          --enc_weight 0.0 \
          --gcn_weight 1.0
done
#mse:0.2144014686346054, mae:0.30337396264076233, rse:0.42402565479278564
export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 96; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/  \
          --data_path PEMS08.npz \
          --model_id PEMS08_96_96_$p \
          --model $model_name \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $p \
          --e_layers 2 \
          --enc_in 170 \
          --num_vars 170 \
          --des 'Exp' \
          --d_model 64 \
          --n_heads 4 \
          --d_ff 64 \
          --batch_size 8 \
          --dropout 0.1 \
          --learning_rate 0.003 \
          --train_epochs 20 \
          --k 2 \
          --activation sigmoid \
          --patch_len 20 \
          --stride 5 \
          --padding_patch end \
          --nfft_min 16 \
          --hidden_dims 64 \
          --n_stft_branch 2 \
          --patience 3 \
          --broadcast 1 \
          --enc_weight 0.0 \
          --gcn_weight 1.0
done
#mse:0.3458040952682495, mae:0.3860457241535187, rse:0.5387234687805176
