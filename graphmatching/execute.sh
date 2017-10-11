#!/bin/bash
source activate ds
s=0.91
fname=91
alg=uid_repeat_seed
today=`date +%m-%d.%H:%M`
echo $s
logs=./logs/${alg}/${fname}
mkdir -p $logs
for var in 1 2
do
	fname=$logs/seed_$(printf "%02d" $var)_thr_$(printf "%.2f" $s)_${today}.log
	echo $fname;
	python -u main.py $var $s 2>&1 | tee ${fname};
	sleep 1;
done
