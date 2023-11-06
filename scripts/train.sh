#!/bin/bash
for((seed=1;seed<=5;seed++));
do
	accelerate launch --config_file config/parallel_config.yaml confit/train.py \
    --config config/training_config.yaml \
    --dataset GB1_Olson2014_ddg \
    --sample_seed 0 \
    --model_seed $seed
done
python confit/inference.py --dataset GB1_Olson2014_ddg --shot 48