hold off;
clear; clc;

blocks = [1 2 4 8 16 32 64 128 256 512 1024];
s5000 = [80.82 84.745 85.3 85.8 86.17 85.8893 86.551 86.1809 86.519 86.136 86.451];
s8192 = [65.92 80.991 82.65 83.0853 84.0185 84.22 84.20 84.046 84.091 83.3746 80.5576];
s16384 = [.2991, .281 .2789 .2761 .274 .2729 .2726 .2701 .2699 .2707 .27274];
s16384 = 21.75 ./ s16384 ;
s20000 = [.421 .41 .415 .405 .403 .4008 .403 .402 .402 .403 .403];
s20000 = 31.52 ./ s20000;


semilogx(blocks, s5000, '-o');
hold on;
semilogx(blocks, s8192, '-o');
semilogx(blocks, s16384, '-o');
semilogx(blocks, s20000, '-o');

xlabel("Block size (xdim) (ydim = 1024/xdim)");
ylabel("Speedup vs serial");

legend("n=m=5000", "n=m=8192", "n=m=16384", "n=m=20000", "Location", "southeast");

title("Single precision speedups")