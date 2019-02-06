#!/bin/bash

# parameters
DATA='~/.datasets/txt'
NAMEDATASET='paranmt'
DATASET='commandpairsextsmall.txt' #commandpairsext.txt; para-nmt-50m/para-nmt-50m.txt; para-nmt-50m-demo/para-nmt-50m-small.txt
VOCFILE='para-nmt-50m-demo/ngram-word-concat-40.pickle'
PROJECT='../out/netruns'
EPOCHS=1000
NBATCH=500 #20, 1000
BATCHSIZE=128 #10, 128
LEARNING_RATE=0.0001
MOMENTUM=0.9
PRINT_FREQ=30
RESUME='model_best.pth.tar' #'chk000042.pth.tar', model_best.pth.tar
GPU=0
ARCH='nmt' #
LOSS='maskll'
OPT='adam'
SCHEDULER='step'
SNAPSHOT=50
EXP_NAME='nlp_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_006'


#NET CONFIGURATE
ATTNMODEL='dot' #dot, general, concat
HIDDENSIZE=500 
ENCODER_N_LAYERS=4 
DECODER_N_LAYERS=4
DROPOUT=0.1
TEACHER_FORCING_RATIO=1.0
MAX_LENGTH=10


rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT
mkdir $PROJECT/$EXP_NAME 


## execute
python ../train.py \
$DATA \
--namedataset=$NAMEDATASET \
--project=$PROJECT \
--vocabulary=$VOCFILE \
--dataset=$DATASET \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--nbatch=$NBATCH \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--print-freq=$PRINT_FREQ \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--name-dataset=$NAMEDATASET \
--finetuning \
--attn_model=$ATTNMODEL \
--hidden_size=$HIDDENSIZE \
--encoder_n_layers=$ENCODER_N_LAYERS \
--decoder_n_layers=$DECODER_N_LAYERS \
--dropout=$DROPOUT \
--teacher_forcing_ratio=$TEACHER_FORCING_RATIO \
--max_length=$MAX_LENGTH \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \




#--parallel \
