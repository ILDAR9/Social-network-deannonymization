#!/bin/bash
source activate ds
s_threshold_or_step=99
is_repeat=0
is_model=1
algo_type=0

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

if [ $algo_type -eq 1 ]
then
alg=step_${s_threshold_or_step}_$alg
fi

today=`date +%m-%d.%H:%M`
echo $s
logs=./logs/alg_type${algo_type}_${alg}/${s_threshold_or_step}
mkdir -p $logs
for var in 0
do
	fname=$logs/seed_$(printf "%02d" $var)_thr_$(printf "%.2f" $s)_${today}.log
	echo $fname;
	python -u main.py $algo_type $var ${s_threshold_or_step} ${is_repeat} ${is_model} 2>&1 | tee ${fname};
	sleep 1;
done
