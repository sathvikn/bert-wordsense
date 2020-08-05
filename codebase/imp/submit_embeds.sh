#!/bin/bash
dataset="bert"  #dataset name
labelratio=0.4 #fraction of  labeled data
numunlabel=5 #number of unlabeled examples
nclassestrain=5 #number of classes to sample from 
nclassesepisode=5 #n-way of episode
accumulationsteps=1 #number of gradient accumulation steps before propagating loss (refer to paper)
nclasseseval=5 #n-way of test episodes
nshot=1 #shot of episodes
model="imp" #model name
seed=0 #seed
results="bert_results"
word_type="table_n"

#data-root should be the JSON for a dictionary of embeddings for a particular type
#Change the dataset parameter to something like "BERT"

python2 run_eval.py --data-root=/Users/sathvik/Desktop/Berkeley/Research/thesis/codebase/data/pipeline_results/semcor/ --dataset=$dataset --label-ratio $labelratio --num-unlabel-test=$numunlabel --num-unlabel=$numunlabel --nclasses-train=$nclassestrain --nclasses-episode=$nclassesepisode --nclasses-eval=$nclasseseval --model $model --results $results"/"$dataset"/"$nshot"_"$nclassesepisode"/" --nshot=$nshot --seed=$seed --type=$word_type
