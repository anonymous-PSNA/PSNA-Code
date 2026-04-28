#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 96; do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/Flight/ \
      --data_path Flight.csv \
      --model_id Flight_96_96_$p \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $p \
      --e_layers 1 \
      --d_layers 1 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 32 \
      --n_heads 4 \
      --d_ff 64 \
      --factor 3 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --k 2 \
      --class_strategy projection \
      --embed timeF \
      --activation sigmoid \
      --num_workers 8 \
      --patch_len 2 \
      --stride 1 \
      --padding_patch end \
      --num_vars 137 \
      --nfft_min 8 \
      --hidden_dims 32 \
      --n_stft_branch 3 \
      --broadcast 1 \
      --patience 3 \
      --enc_weight 0.0 \
      --gcn_weight 1.0
done
export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 192 336; do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/Flight/ \
      --data_path Flight.csv \
      --model_id Flight_96_96_$p \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $p \
      --e_layers 1 \
      --d_layers 1 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 64 \
      --n_heads 4 \
      --d_ff 64 \
      --factor 3 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --k 2 \
      --class_strategy projection \
      --embed timeF \
      --activation sigmoid \
      --num_workers 8 \
      --patch_len 2 \
      --stride 1 \
      --padding_patch end \
      --num_vars 137 \
      --nfft_min 16 \
      --hidden_dims 64 \
      --n_stft_branch 2 \
      --broadcast 1 \
      --patience 3 \
      --enc_weight 0.0 \
      --gcn_weight 1.0
done
export CUDA_VISIBLE_DEVICES=0
model_name=PSNA
for p in 720; do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/Flight/ \
      --data_path Flight.csv \
      --model_id Flight_96_96_$p \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $p \
      --e_layers 1 \
      --d_layers 1 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 64 \
      --n_heads 4 \
      --d_ff 64 \
      --factor 3 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --k 2 \
      --class_strategy projection \
      --embed timeF \
      --activation sigmoid \
      --num_workers 8 \
      --patch_len 2 \
      --stride 1 \
      --padding_patch end \
      --num_vars 137 \
      --nfft_min 8 \
      --hidden_dims 64 \
      --n_stft_branch 3 \
      --broadcast 1 \
      --patience 3 \
      --enc_weight 0.0 \
      --gcn_weight 1.0
done
