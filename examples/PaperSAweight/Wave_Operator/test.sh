#!/bin/sh

ksi_Vec=(1) #(1, 2, 4,8)
useSAweight_Vec=(1) #(None, 'BRDR')

for useSAweight in ${useSAweight_Vec[@]}; do
for ksi in ${ksi_Vec[@]}; do
	for repeatID in $(seq $1 $2);do
                str="repeatID=${repeatID}_ksi=${ksi}_useSAweight=${useSAweight}"           
		sbatch -o "${str}.out"  -p $3 run.sh $repeatID $ksi $useSAweight
	done
done
done


