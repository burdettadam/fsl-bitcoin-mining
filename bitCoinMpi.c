#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // caculate number of hashes(range) to run

    if (rank == 0){
    //while // 60 secs
        //recive all rank 
        //sends start nance and range to 
    }
    else {
        //do first range based off of caclulation
        //while
            //send rank to root
            //recive nance and range from root.
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}