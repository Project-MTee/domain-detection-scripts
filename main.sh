#!/bin/bash

#$1 = parallel data folder
#$2 = monolingual data folder
#$3 = directory for tokenized data
#$4 = directory for model checkpoints
#$5 = experiment name

python 02_cut_and_tokenize.py \
    --parallel_data_dir $1 \
    --mono_data_dir $2 \
    --out_dir $3 \
    --tokenizer xlm-roberta-base \
    --limit 202733 \
    --custom_limit crisis-et=570495 military-et=321490 military-en=321490

train_data_path="${3}/train.tok"
valid_data_path="${3}/valid.tok"

shuffled_train_path="${3}/train.shuf.tok"

#shuffle train data
shuf $train_data_path > $shuffled_train_path

#train model
#remove wandb argument if you don't use it
python 03_train.py \
    --model xlm-roberta-base \
    --tokenizer xlm-roberta-base \
    --train_file $shuffled_train_path \
    --valid_file $valid_data_path \
    --out_path $4 \
    --run_name $5 \
    --wandb true \
    train


