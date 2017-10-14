#!/bin/bash
source activate ds
s=0.30
fname=30
is_repeat=1
is_model=1
if [ $is_repeat -eq 1 ]
then
alg=uid_repeat
else
alg=uid
fi

if [ $is_model -eq 1 ]
then
alg=with_model_$alg
fi

today=`date +%m-%d.%H:%M`
echo $s
logs=./logs/${alg}/${fname}
mkdir -p $logs
for var in 0
do
	fname=$logs/seed_$(printf "%02d" $var)_thr_$(printf "%.2f" $s)_${today}.log
	echo $fname;
	python -u main.py $var $s ${is_repeat} ${is_model} 2>&1 | tee ${fname};
	sleep 1;
done
