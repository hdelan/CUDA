#!/bin/bash

nval=20000

[[ -e results.txt ]] && rm results.txt 

for block_size in 1 2 4 8 16 32 64 128 256 512 1024
do
echo "Running for block size $block_size"
./expint -n $nval -m $nval -s -v -k $block_size >> results.txt
done  
