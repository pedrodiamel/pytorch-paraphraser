#!/bin/bash

# parameters
DATA='~/.datasets/txt'
DATASET='para-nmt-50m/para-nmt-50m.txt' #para-nmt-50m/para-nmt-50m.txt; para-nmt-50m-demo/para-nmt-50m-small.txt
VOCFILE='para-nmt-50m-demo/ngram-word-concat-40.pickle'
NAMEDATASET='txt'
PROJECT='../out/netruns'
EPOCHS=50
NBATCH=1000
BATCHSIZE=128 #128
LEARNING_RATE=0.0001
MOMENTUM=0.9
PRINT_FREQ=30
RESUME='model_best.pth.tar' #'chk000042.pth.tar', model_best.pth.tar
GPU=1
ARCH='nmt' #
LOSS='maskll'
OPT='adam'
SCHEDULER='step'
SNAPSHOT=50
EXP_NAME='nlp_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_001'


# rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
# rm -rf $PROJECT/$EXP_NAME/
# mkdir $PROJECT
# mkdir $PROJECT/$EXP_NAME 


## execute
python ../train.py \
$DATA \
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
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \


#--parallel \
