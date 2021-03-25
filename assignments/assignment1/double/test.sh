#!/bin/bash

block_size=16

echo "Running for block size $block_size"
[[ -e output/$block_size.txt ]] && rm output/$block_size.txt

for n in 1000 5000 10000 25000
do
echo "/*                N=M=$n              */" >> output/$block_size.txt
./matrix -n $n -m 1000 -b $block_size -t >> output/$block_size.txt
done  
