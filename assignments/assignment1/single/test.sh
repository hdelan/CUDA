#!/bin/bash

for block_size in 4 8 16 32 64 128 256 512 1024
do

[[ -e output/$block_size.txt ]] && rm output/$block_size.txt 

echo "Running for block size $block_size"
for n in 1000 5000 10000 25000
do
echo "/*                N=M=$n              */" >> output/$block_size.txt
./matrix -n $n -m 1000 -b $block_size -t >> output/$block_size.txt
done  
done  
