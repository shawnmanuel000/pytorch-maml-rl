#!/bin/bash

env_name=$1
latent=$2
model=$3
laser=$4
script=scripts/train_sc.sh
envs=([0]="Particle-R1" [1]="Arm-R1" [2]="PathFollow-3D" [3]="DoorOpen")

n="17"
mem="8GB"

for i in `seq 0 3`
do
	env_name=${envs[$i]}
	export env_name="$env_name"
	echo "sbatch -n $n --mem=$mem --job-name="$env_name" $script"
	sbatch -n $n --mem=$mem --job-name="$jobname" $script
done

