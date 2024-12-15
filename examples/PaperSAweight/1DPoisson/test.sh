#!/bin/sh

nResi_Vec=(1000)
ksi_Vec=(3) #(1,2,4,8)
useSAweight_Vec=(0) #(True, False)

for nResi in ${nResi_Vec[@]}; do
for useSAweight in ${useSAweight_Vec[@]}; do
for ksi in ${ksi_Vec[@]}; do
	for repeatID in $(seq $1 $2);do
                str="repeatID=${repeatID}_nResi=${nResi}_ksi=${ksi}_useSAweight=${useSAweight}"           
		sbatch -o "${str}.out"  -p $3 run.sh $repeatID $nResi   $ksi   $useSAweight
	done
    done
done
done


