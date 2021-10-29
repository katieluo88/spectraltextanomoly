#!/bin/bash

data_str=data_raw/snli_1.0/snli_1.0_test.jsonl
data_out=data/snli_test
model=bert-base-uncased-snli
attack=a2t
num_examples=1000

python generate_csv.py -i $data_str -o $data_out0.csv
textattack attack --recipe $attack --model $model --num-examples $num_examples
