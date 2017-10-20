#!/bin/bash
source activate ds

today=`date +%m-%d.%H:%M`
echo $s
logs=./logs/model_predict
mkdir -p $logs

fname=$logs/console_${today}.log
echo $fname;
python -u main_model_predict.py 2>&1 | tee ${fname};