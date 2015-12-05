#include <mpi.h> //http://www.hpc.cam.ac.uk/using-clusters/compiling-and-development/parallel-programming-mpi-example
#include <openssl/sha.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define WORKTAG     1
#define SIZE     16384
#define DIETAG     2
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
void master(int ntasks,int  range )
{
	int	 rank, nonce = 0 ;
	double       result=0;
	MPI_Status     status;
/*
* Seed the slaves.
*/
	for (rank = 1; rank < ntasks; ++rank) {
		nonce += range; /* get_next_nonce_request */;
		MPI_Send(&nonce,         /* message buffer */
		1,              /* one data item */
		MPI_INT,        /* data item is an integer */
		rank,           /* destination process rank */
		WORKTAG,        /* user chosen message tag */
		MPI_COMM_WORLD);/* always use this */
	}

/*
* Receive a result from any slave and dispatch a new nonce
* request nonce requests have been exhausted.
*/
 	nonce += range; /* get_next_nonce_request */
	double start = When();
	double timer = When() - start;
	while (timer < .1/* valid new nonce request */) {
		MPI_Recv(&result,       /* message buffer */
		1,              /* one data item */
		MPI_DOUBLE,     /* of type double real */
		MPI_ANY_SOURCE, /* receive from any sender */
		MPI_ANY_TAG,    /* any type of message */
		MPI_COMM_WORLD, /* always use this */
		&status);       /* received message info */
		MPI_Send(&nonce, 1, MPI_INT, status.MPI_SOURCE,
		WORKTAG, MPI_COMM_WORLD);
		nonce += range; /* get_next_nonce_request */
		if ( nonce % 800000 == 0){
            timer = (When() - start);
        }
	}
/*
* Receive results for outstanding nonce requests.
*/
	for (rank = 1; rank < ntasks; ++rank) {
		MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
		MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	}
/*
* Tell all the slaves to exit.
*/
	for (rank = 1; rank < ntasks; ++rank) {
		MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
	}
	return;
}

void slave(int range,SHA256_CTX sha256_pass1,block_header header,unsigned char hash1,SHA256_CTX sha256_pass2,unsigned char hash2 )
{
	double              result=1;
	MPI_Status          status;
	for (;;) {
		MPI_Recv(&header.nonce, 1, MPI_INT, 0, MPI_ANY_TAG,
		MPI_COMM_WORLD, &status);
/*
* Check the tag of the received message.
*/		
		if (status.MPI_TAG == DIETAG) {
			return;
		}
        //printf("slave received nonce: %d", nonce);
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

		result = 1 /* do the nonce */;
		MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
}
int main(argc, argv)
int argc;
char *argv[];
{

	int myrank;
	MPI_Init(&argc, &argv);   /* initialize MPI */
	MPI_Comm_rank(MPI_COMM_WORLD , &myrank);      /* process rank, 0 thru N-1 */
	int	ntasks, range;
	MPI_Comm_size(
	MPI_COMM_WORLD, &ntasks);          /* #processes in application */
	range = SIZE / ntasks;    
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

	if (myrank == 0) {
		master(ntasks,range);
	} else {
		slave(range,sha256_pass1,header,hash1,sha256_pass2,hash2 );
	}
	MPI_Finalize();       /* cleanup MPI */
}
