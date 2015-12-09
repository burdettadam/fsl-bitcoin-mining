#ifdef fail
        #!/bin/bash
        # NOTE you can chmod 0755 this file and then execute it to compile (or just copy and paste)
        gcc -o hashblock hashblock.c -lssl
        exit 0
#endif
 
//#include <openssl/sha.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "sha256.cu"
 
#define SHA256_DIGEST_SIZE 32
#define NUM_BLOCKS 1024

// this is the block header, it is 80 bytes long (steal this code)
typedef struct block_header {
        unsigned int    version;
        // dont let the "char" fool you, this is binary data not the human readable version
        unsigned char   prev_block[32];
        unsigned char   merkle_root[32];
        unsigned int    timestamp;
        unsigned int    bits;
        unsigned int    nonce;
} block_header;
 
double When()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

// we need a helper function to convert hex to binary, this function is unsafe and slow, but very readable (write something better)
void hex2bin(unsigned char* dest, const char* src)
{
        int c, pos;
        char buf[3];
 
        pos=0;
        c=0;
        buf[2] = 0;
        while(c < strlen(src))
        {
                // read in 2 characaters at a time
                buf[0] = src[c++];
                buf[1] = src[c++];
                // convert them to a interger and recast to a char (uint8)
                dest[pos++] = (unsigned char)strtol(buf, NULL, 16);
        }
       
}
 
// this function is mostly useless in a real implementation, were only using it for demonstration purposes
__device__ void print_hash(unsigned char hash[])
{
   int idx;
   for (idx=0; idx < 32; idx++)
      printf("%02x",hash[idx]);
   printf("\n");
}
 
// this function swaps the byte ordering of binary data, this code is slow and bloated (write your own)
__device__ __host__ void byte_swap(unsigned char* data) {
        int c;
        unsigned char tmp[SHA256_DIGEST_SIZE];
       
        c=0;
        while(c<SHA256_DIGEST_SIZE)
        {
                tmp[c] = data[SHA256_DIGEST_SIZE-(c+1)];
                c++;
        }
       
        c=0;
        while(c<SHA256_DIGEST_SIZE)
        {
                data[c] = tmp[c];
                c++;
        }
}

__global__ void doCalc(unsigned char *dev_prev_block, unsigned char *dev_merkle_root, int seed) {
    int repeats = 1;
    int i;   
    block_header header;
    header.version =        2;
    header.timestamp =      1392872245;
    header.bits =           419520339;
    // we are going to supply the block header with the values from the generation block 0
    for(i=0;i<32;i++) {
        header.prev_block[i] = dev_prev_block[i];
        header.merkle_root[i] = dev_merkle_root[i];
        //if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u",header.prev_block[i]);
    }
    //if(threadIdx.x==0 && blockIdx.x == 0) printf("\n");

    //we need a place to store the checksums
    unsigned char hash1[32];
    unsigned char hash2[32];
   
    // you should be able to reuse these, but openssl sha256 is slow, so your probbally not going to implement this anyway
    SHA256_CTX sha256_pass1, sha256_pass2;

    header.nonce = (seed *  blockDim.x * NUM_BLOCKS) + blockIdx.x * blockDim.x + threadIdx.x*repeats;
    //if(threadIdx.x == 0) printf("nonce: %d\n", header.nonce);
    // Use SSL's sha256 functions, it needs to be initialized


    for(i=0;i<repeats;i++) {
        sha256_init(&sha256_pass1);
        // then you 'can' feed data to it in chuncks, but here were just making one pass cause the data is so small
        sha256_update(&sha256_pass1, (unsigned char*)&header, sizeof(block_header));
        // this ends the sha256 session and writes the checksum to hash1
        sha256_final(&sha256_pass1,hash1);
           
            // to display this, we want to swap the byte order to big endian
     //       byte_swap(hash1, SHA256_DIGEST_LENGTH); // this is for printing 
     //       printf("Useless First Pass Checksum: ");
     //       hexdump(hash1, SHA256_DIGEST_LENGTH);
     
            // but to calculate the checksum again, we need it in little endian, so swap it back
     //       byte_swap(hash1, SHA256_DIGEST_LENGTH);
           
        //same as above
        sha256_init(&sha256_pass2);
        sha256_update(&sha256_pass2, hash1, SHA256_DIGEST_SIZE);
        sha256_final(&sha256_pass2, hash2);
        // if ( header.nonce == 0 || header.nonce == 3 || header.nonce == 856192328 ) {
        //     //hexdump((unsigned char*)&header, sizeof(block_header));
        //     printf("%u:\n", header.nonce);
        //     byte_swap(hash2);
        //     printf("Target Second Pass Checksum: \n");
        //     print_hash(hash2);
        // }
        header.nonce++;
    }
}

int main() {
    int i = 0;
    int blocksize = 1024;
    int threads = 1024;

    long long hashes = 0;

    int counter = 0;

    unsigned char *dev_merkle_root, *dev_prev_block;

    unsigned char prev_block[32], merkle_root[32];

    hex2bin(prev_block,              "000000000000000117c80378b8da0e33559b5997f2ad55e2f7d18ec1975b9717");
    hex2bin(merkle_root,             "871714dcbae6c8193a2bb9b2a69fe1c0440399f38d94b3a0f1b447275a29978a");


    byte_swap(prev_block);
    byte_swap(merkle_root);

    // for(i=0; i<32;i++) {
    //     printf("%u",prev_block[i]);
    // }
    // printf("\n");

    cudaMalloc((void**)&dev_prev_block, 32*sizeof(unsigned char));
    cudaMemcpy(dev_prev_block, &(prev_block), 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_merkle_root, 32*sizeof(unsigned char));
    cudaMemcpy(dev_merkle_root, &(merkle_root), 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    double start = When();
    double timer = When() - start;
    while ( timer < 60.0){
        //printf("before kernel call\n");
        doCalc<<< blocksize, threads >>>(dev_prev_block, dev_merkle_root, counter);
        hashes += blocksize*threads;
        counter++;
        timer = When() - start;
        //printf("%d iterations\n",counter);
        cudaDeviceSynchronize();
    }
    // printf("hashes: %lld\n", hashes);
    // printf("seconds: %f\n", When() - start);
    printf("number of hashs per second = %lld\n",(long long) (hashes / (When() - start)) );

 
    return 0;
}