#fsl bitcoin Mining
--------
using cuda.
#Test Cases
---
https://en.bitcoin.it/wiki/Test_Cases

#Tutorial
---
http://www.righto.com/2014/02/bitcoin-mining-hard-way-algorithms.html

http://pastebin.com/EXDsRbYH

https://github.com/ckolivas/cgminer/blob/master/sha2.c#L93

http://www.gladman.me.uk

can be compiled with the following command

gcc -o bitcoin bitcoinIterative.c -O3 -lssl -lcrypto

mpicc -o bitcoin bitCoinMpi.c -O3 -lssl -lcrypto
