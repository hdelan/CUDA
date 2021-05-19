#!/bin/bash

[[ -e results.txt ]] && rm results.txt 

for block_size in 2 4 8 16 32 64 128 256 512 1024
do
echo "Running for block size $block_size"
./radiator -c -a -b $block_size -t >> results.txt
done  
