#!/bin/sh

nResi_Vec=(10201)
SAweight_Vec=(2) #(None, 'minMax', 'RBA', 'BRDR')

for nResi in ${nResi_Vec[@]}; do
for SAweight in ${SAweight_Vec[@]}; do
	for repeatID in $(seq $1 $2);do
                str="repeatID=${repeatID}_nResi=${nResi}_SAweight=${SAweight}"           
		sbatch -o "${str}.out"  -p $3 run.sh $repeatID $nResi  $SAweight
	done
    done
done


