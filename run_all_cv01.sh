#!/bin/bash

for model in dense cnn
do
	for optimizer in sgd adam
	do
		for dp in 0 0.1 0.3 0.5
		do
			for lr in 0.1 0.01 0.001 0.0001 0.00001
			do
				for iteration in 1 2 3 4 5
				do
					qsub -v model="$model",optimizer="$optimizer",dp="$dp",lr="$lr" plan_cv01.sh
				done
			done
		done
	done
done