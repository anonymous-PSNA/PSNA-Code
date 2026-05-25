#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 96; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/electricity/  \
          --data_path electricity.csv \
          --model_id ECL_96_96_$p \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $p \
          --e_layers 1 \
          --enc_in 321 \
          --num_vars 321 \
          --des 'Exp' \
          --d_model 128 \
          --n_heads 4 \
          --d_ff 64 \
          --batch_size 8 \
          --dropout 0.1 \
          --learning_rate 0.002 \
          --train_epochs 10 \
          --k 2 \
          --activation relu \
          --patch_len 2 \
          --stride 2 \
          --padding_patch end \
          --nfft_min 8 \
          --hidden_dims 128 \
          --patience 3 \
          --n_stft_branch 4 \
          --broadcast 1 \
          --enc_weight 0.0 \
          --gcn_weight 1.0
done
for p in 192; do
    echo "Running for pred_len: $p"
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/electricity/ \
          --data_path electricity.csv \
          --model_id ECL_96_96_$p \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $p \
          --e_layers 1 \
          --enc_in 321 \
          --num_vars 321 \
          --des 'Exp' \
          --d_model 128 \
          --n_heads 4 \
          --d_ff 64 \
          --batch_size 32 \
          --dropout 0.1 \
          --learning_rate 0.0005 \
          --train_epochs 10 \
          --k 2 \
          --activation relu \
          --patch_len 2 \
          --stride 1 \
          --padding_patch end \
          --nfft_min 8 \
          --hidden_dims 128 \
          --patience 3 \
          --n_stft_branch 3 \
          --broadcast 1 \
          --enc_weight 0.0 \
          --gcn_weight 1.0
done
export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 336; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/electricity/  \
          --data_path electricity.csv \
          --model_id ECL_96_96_$p \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $p \
          --e_layers 2 \
          --enc_in 321 \
          --num_vars 321 \
          --des 'Exp' \
          --d_model 128 \
          --n_heads 4 \
          --d_ff 64 \
          --batch_size 8 \
          --dropout 0.1 \
          --learning_rate 0.002 \
          --train_epochs 10 \
          --k 2 \
          --activation relu \
          --patch_len 2 \
          --stride 2 \
          --padding_patch end \
          --nfft_min 8 \
          --hidden_dims 128 \
          --patience 3 \
          --n_stft_branch 4 \
          --broadcast 1 \
          --enc_weight 0.0 \
          --gcn_weight 1.0
done
export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 720; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/electricity/  \
          --data_path electricity.csv \
          --model_id ECL_96_96_$p \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $p \
          --e_layers 2 \
          --enc_in 321 \
          --num_vars 321 \
          --des 'Exp' \
          --d_model 128 \
          --n_heads 4 \
          --d_ff 64 \
          --batch_size 8 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 10 \
          --k 2 \
          --activation relu \
          --patch_len 2 \
          --stride 2 \
          --padding_patch end \
          --nfft_min 8 \
          --hidden_dims 128 \
          --patience 3 \
          --n_stft_branch 4 \
          --broadcast 1 \
          --enc_weight 0.0 \
          --gcn_weight 1.0
done
