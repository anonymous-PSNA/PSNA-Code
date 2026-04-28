#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 96 192 336; do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_96_96_$p \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $p \
      --e_layers 1 \
      --enc_in 8 \
      --num_vars 8 \
      --des 'Exp' \
      --d_model 64 \
      --n_heads 4 \
      --d_ff 64 \
      --batch_size 32 \
      --dropout 0.3 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --k 2 \
      --activation sigmoid \
      --patch_len 2 \
      --stride 1 \
      --padding_patch end \
      --nfft_min 16 \
      --hidden_dims 64 \
      --patience 3 \
      --n_stft_branch 2 \
      --broadcast 1 \
      --enc_weight 0.0 \
      --gcn_weight 1.0
done
export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 720; do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_96_96_$p \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $p \
      --e_layers 1 \
      --enc_in 8 \
      --num_vars 8 \
      --des 'Exp' \
      --d_model 64 \
      --n_heads 4 \
      --d_ff 64 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --k 2 \
      --activation sigmoid \
      --patch_len 2 \
      --stride 1 \
      --padding_patch end \
      --nfft_min 16 \
      --hidden_dims 64 \
      --patience 3 \
      --n_stft_branch 2 \
      --broadcast 1 \
      --enc_weight 1.0 \
      --gcn_weight 0.1
done
