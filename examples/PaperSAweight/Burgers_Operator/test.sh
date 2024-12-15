#!/bin/sh

ksi_Vec=(0 1) #(0.01, 0.001, 0.0001)
SAweight_Vec=(1) #(None, 'BRDR')

for SAweight in ${SAweight_Vec[@]}; do
for ksi in ${ksi_Vec[@]}; do
	for repeatID in $(seq $1 $2);do
                str="repeatID=${repeatID}_ksi=${ksi}_SAweight=${SAweight}"           
		sbatch -o "${str}.out"  -p $3 run.sh $repeatID $ksi $SAweight
	done
done
done


