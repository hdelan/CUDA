for block_size in 4 8 16 32 64 128 256 512 1024
do
echo "Running for block size $block_size"
echo "/*                N=M=1000               */" >> $block_size.txt
./matrix -n 1000 -m 1000 -b $block_size -t >> $block_size.txt
echo "/*                N=M=5000               */" >> $block_size.txt
./matrix -n 5000 -m 5000 -b $block_size -t >> $block_size.txt
echo "/*                N=M=10000               */" >> $block_size.txt
./matrix -n 10000 -m 10000 -b $block_size -t >> $block_size.txt
echo "/*                N=M=25000               */" >> $block_size.txt
./matrix -n 25000 -m 25000 -b $block_size -t >> $block_size.txt
done  
