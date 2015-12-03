#include <mpi.h>
#include <stdio.h>
#include <openssl/sha.h>
#include <string.h>
#include <stdlib.h>
#define SIZE     16384
struct package{
  int nance;
  int range;
};
int main(int argc, char** argv) {
    package mypack; 
    int root = 0, range;
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank,clientRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // caculate number of hashes(range) to run
    range = SIZE / world_size;    
    // start with a block header struct
    block_header header;
   
    // we need a place to store the checksums
    unsigned char hash1[SHA256_DIGEST_LENGTH];
    unsigned char hash2[SHA256_DIGEST_LENGTH];
   
    // you should be able to reuse these, but openssl sha256 is slow, so your probbally not going to implement this anyway
    SHA256_CTX sha256_pass1, sha256_pass2;


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
    if (rank == 0){
        while ( timer < 60.0){
                    //http://stackoverflow.com/questions/4348900/mpi-recv-from-an-unknown-source
        int buf[32];
        MPI_Status status;
        // receive message from any source  //recive all rank 
        MPI_recv(&clientRank, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        

        //sends start nance and range to 
        // send reply back to sender of the message received above
        MPI_send(mypack, 2, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);

          
        }


    }
    else {
        //do first range based off of caclulation

        //while
            //send rank to root
            MPI_Send(rank, 1, MPI_INT, root, 0, MPI_COMM_WORLD ); 
            //recive nance and range from root.

            MPI_Recv(&mypack, 2, MPI_INT, root, 0, MPI_COMM_WORLD, &status);
              // Use SSL's sha256 functions, it needs to be initialized
            SHA256_Init(&sha256_pass1);
            // then you 'can' feed data to it in chuncks, but here were just making one pass cause the data is so small
            SHA256_Update(&sha256_pass1, (unsigned char*)&header, sizeof(block_header));
            // this ends the sha256 session and writes the checksum to hash1
            SHA256_Final(hash1, &sha256_pass1);
               
                // to display this, we want to swap the byte order to big endian
         //       byte_swap(hash1, SHA256_DIGEST_LENGTH); // this is for printing 
         //       printf("Useless First Pass Checksum: ");
         //       hexdump(hash1, SHA256_DIGEST_LENGTH);
         
                // but to calculate the checksum again, we need it in little endian, so swap it back
         //       byte_swap(hash1, SHA256_DIGEST_LENGTH);
               
            //same as above
            SHA256_Init(&sha256_pass2);
            SHA256_Update(&sha256_pass2, hash1, SHA256_DIGEST_LENGTH);
            SHA256_Final(hash2, &sha256_pass2);
            if ( header.nonce == 0 || header.nonce == 3 || header.nonce == 856192328 ) {
                byte_swap(hash2, SHA256_DIGEST_LENGTH);
                printf("Target Second Pass Checksum: ");
                hexdump(hash2, SHA256_DIGEST_LENGTH);

            }

            header.nonce ++;
            if ( header.nonce % 800000 == 0){
                timer = (When() - start);
            }
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}