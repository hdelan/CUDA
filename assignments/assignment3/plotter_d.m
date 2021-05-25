hold off;
clear; clc;

blocks = [1 2 4 8 16 32 64 128 256 512 1024];
s5000 = [56.772 58.29 58.55 58.901 58.653 58.519 58.61 59.446 58.532 58.248 58.474];
s8192 = [54.8526 55.7918 55.6371 56.0098 55.7227 55.616 55.9131 55.381 55.6301 55.8941 52.4637];
s16384 = [53.2693 54.0193 54.313 54.3123 53.8551 56.5138 51.9409 56.4319 54.1498 53.7111 54.4597];
s20000 = [53.243 53.9148 54.8041 54.443 54.2813 53.8324 54.8324 54.5019 54.6268 54.8098 54.2196];


semilogx(blocks, s5000, '-o');
hold on;
semilogx(blocks, s8192, '-o');
semilogx(blocks, s16384, '-o');
semilogx(blocks, s20000, '-o');

xlabel("Block size (xdim) (ydim = 1024/xdim)");
ylabel("Speedup vs serial");

legend("n=m=5000", "n=m=8192", "n=m=16384", "n=m=20000", "Location", "southeast");

title("Double precision speedups")