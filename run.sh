#!/bin/sh
python3 train.py --lr 0.001 --momentum 0.02 --num_hidden 3 --sizes 100,100,100 --activation sigmoid --loss ce --opt adam --batch_size 100 --anneal true --save_dir pa1/ --expt_dir pa1/exp1/ --train train --test train
