#!/bin/sh



nResi_Vec=(10000)
vis_Vec=(0) #(0.01, 0.00314, 0.001, 0.000314)
SAweight_Vec=(3) #(None, 'minMax', 'RBA', 'BRDR')

for nResi in ${nResi_Vec[@]}; do
for SAweight in ${SAweight_Vec[@]}; do
for vis in ${vis_Vec[@]}; do
	for repeatID in $(seq $1 $2);do
                str="repeatID=${repeatID}_nResi=${nResi}_ksi=${ksi}_SAweight=${SAweight}"           
		sbatch -o "${str}.out"  -p $3 run.sh $repeatID $nResi   $vis $SAweight
	done
done
done
done


