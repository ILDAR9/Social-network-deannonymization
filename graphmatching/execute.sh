#!/bin/bash
source activate ds
s=0.95
fname=95
alg=alg3
today=`date +%Y-%m-%d.%H:%M`
echo $s
logs=./logs/${alg}/${fname}
mkdir -p $logs
for var in 1 2 3 4 5 7 10 20
do
	echo $logs/${alg}_s_${var}_thr_${s}_${today}.log;
	python -u main.py ${alg} $var $s 2>&1 | tee $logs/${alg}_s_${var}_ths_$s_${today}.log;
	sleep 1;
done
