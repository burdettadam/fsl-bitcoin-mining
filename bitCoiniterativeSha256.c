/*
#Copyright (c) 2011, Joseph Matheney
#All rights reserved.
 
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
#    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifdef fail
        #!/bin/bash
        # NOTE you can chmod 0755 this file and then execute it to compile (or just copy and paste)
        gcc -o hashblock hashblock.c -lssl
        exit 0
#endif
 
//#include <openssl/sha.h>
#include "sha2.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
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
void hex2bin(unsigned char* dest, unsigned char* src)
{
        unsigned char bin;
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
void hexdump(unsigned char* data, int len)
{
        int c;
       
        c=0;
        while(c < len)
        {
                printf("%.2x", data[c++]);
        }
        printf("\n");
}
 
// this function swaps the byte ordering of binary data, this code is slow and bloated (write your own)
void byte_swap(unsigned char* data, int len) {
        int c;
        unsigned char tmp[len];
       
        c=0;
        while(c<len)
        {
                tmp[c] = data[len-(c+1)];
                c++;
        }
       
        c=0;
        while(c<len)
        {
                data[c] = tmp[c];
                c++;
        }
}
static inline void swap256(void *dest_p, const void *src_p) {
        uint32_t *dest = dest_p;
        const uint32_t *src = src_p;

        dest[0] = src[7];
        dest[1] = src[6];
        dest[2] = src[5];
        dest[3] = src[4];
        dest[4] = src[3];
        dest[5] = src[2];
        dest[6] = src[1];
        dest[7] = src[0];
    }
int main() {
    // start with a block header struct
    block_header header;
   
    // we need a place to store the checksums
    unsigned char hash1[SHA256_DIGEST_SIZE];
    unsigned char hash2[SHA256_DIGEST_SIZE];
   
    // you should be able to reuse these, but openssl sha256 is slow, so your probbally not going to implement this anyway
    sha256_ctx sha256_pass1, sha256_pass2;


    // we are going to supply the block header with the values from the generation block 0
    header.version =        2;
    hex2bin(header.prev_block,              "000000000000000117c80378b8da0e33559b5997f2ad55e2f7d18ec1975b9717");
    hex2bin(header.merkle_root,             "871714dcbae6c8193a2bb9b2a69fe1c0440399f38d94b3a0f1b447275a29978a");
    header.timestamp =      1392872245;
    header.bits =           419520339;
    header.nonce =          0;
   
    // the endianess of the checksums needs to be little, this swaps them form the big endian format you normally see in block explorer
    byte_swap(header.prev_block, 32); // we need this to set up the header
    byte_swap(header.merkle_root, 32); // we need this to set up the header
   
    // dump out some debug data to the terminal
    printf("sizeof(block_header) = %d\n", sizeof(block_header));
    printf("Block header (in human readable hexadecimal representation): ");
    hexdump((unsigned char*)&header, sizeof(block_header));
    double start = When();
    double timer = When() - start;
        while ( timer < 60.0){

            // Use SSL's sha256 functions, it needs to be initialized
            sha256_init(&sha256_pass1);
            // then you 'can' feed data to it in chuncks, but here were just making one pass cause the data is so small
            sha256_update(&sha256_pass1, (unsigned char*)&header, sizeof(block_header));
            // this ends the sha256 session and writes the checksum to hash1
            sha256_final(hash1, &sha256_pass1);
               
                // to display this, we want to swap the byte order to big endian
         //       swap256(hash1, SHA256_DIGEST_LENGTH); // this is for printing 
         //       printf("Useless First Pass Checksum: ");
         //       hexdump(hash1, SHA256_DIGEST_LENGTH);
         
                // but to calculate the checksum again, we need it in little endian, so swap it back
         //       swap256(hash1, SHA256_DIGEST_LENGTH);
               
            //same as above
            sha256_init(&sha256_pass2);
            sha256_update(&sha256_pass2, hash1, SHA256_DIGEST_SIZE);
            sha256_final(hash2, &sha256_pass2);
            if ( header.nonce == 0 || header.nonce == 3 || header.nonce == 856192328 ) {
                swap256(hash2, SHA256_DIGEST_SIZE);
                printf("Target Second Pass Checksum: ");
                hexdump(hash2, SHA256_DIGEST_SIZE);

            }

            header.nonce ++;
            if ( header.nonce % 800000 == 0){
                timer = (When() - start);
            }
        }
       printf("number of hashs per second = %f\n",header.nonce / 60.0 );

 
        return 0;
}
