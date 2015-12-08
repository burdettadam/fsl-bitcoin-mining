#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCKSIZE 1024
#define MAXIT 359
#define TOTROWS		(BLOCKSIZE*8)
#define TOTCOLS		(BLOCKSIZE*8)
#define NOTSETLOC       -1 // for cells that are not fixed
#define SETLOC 		1 // for cells that are fixed
#define EPSILON		0.1

#define QMAX(x,y) (((x) > (y))? (x): (y))


int *lkeepgoing;
float *iplate;
float *oplate;
float *fixed;
float *tmp;
int ncols, nrows;

double When();
void Compute();


int main(int argc, char *argv[])
{
	double t0, tottime;
	ncols = TOTCOLS;
	nrows = TOTROWS;
	int i=0;

	cudaMalloc((void **) &lkeepgoing, nrows * ncols * sizeof(int));
	cudaMalloc((void **) &iplate, nrows * ncols * sizeof(float));
	cudaMalloc((void **) &oplate, nrows * ncols * sizeof(float));
	cudaMalloc((void **) &fixed,  nrows * ncols * sizeof(float));
	fprintf(stderr,"Memory allocated\n");
	t0 = When();
	/* Now proceed with the Jacobi algorithm */
	for(i=0;i<10;i++) {
		Compute();
	}

	tottime = (When() - t0)/10;
	printf("Total Time is: %lf sec.\n", tottime);

	return 0;
}

__global__ void InitArrays(float *ip, float *op, float *fp, int *kp, int ncols)
{
	int i;
	float *fppos, *oppos, *ippos;
        int *kppos;
        int blockOffset;
        int rowStartPos;
        int colsPerThread;
	
        // Each block gets a row, each thread will fill part of a row

	// Calculate the offset of the row
        blockOffset = blockIdx.x * ncols;
        // Calculate our offset into the row
	rowStartPos = threadIdx.x * (ncols/blockDim.x);
        // The number of cols per thread
        colsPerThread = ncols/blockDim.x;

	ippos = ip + blockOffset+ rowStartPos;
	fppos = fp + blockOffset+ rowStartPos;
	oppos = op + blockOffset+ rowStartPos;
	kppos = kp + blockOffset+ rowStartPos;

	for (i = 0; i < colsPerThread; i++) {
		fppos[i] = NOTSETLOC; // Not Fixed
		ippos[i] = 50;
		oppos[i] = 50;
	        kppos[i] = 1; // Keep Going
	}
	if(rowStartPos == 0) {
		fppos[0] = SETLOC;
		ippos[0] = 0;
		oppos[0] = 0;
		kppos[0] = 0;
	}
	if(rowStartPos + colsPerThread >= ncols) {
		fppos[colsPerThread-1] = SETLOC;
		ippos[colsPerThread-1] = 0;
		oppos[colsPerThread-1] = 0;
		kppos[colsPerThread-1] = 0;
	}
	if(blockOffset == 0) {
		for(i=0;i < colsPerThread; i++) {
			fppos[i] = SETLOC;
			ippos[i] = 0;
			oppos[i] = 0;
			kppos[i] = 0;
		}
	}
	if(blockOffset == ncols - 1) {
		for(i=0;i < colsPerThread; i++) {
			fppos[i] = SETLOC;
			ippos[i] = 100;
			oppos[i] = 100;
			kppos[i] = 0;
		}
	}
	if(blockOffset == 400 && rowStartPos < 330) {
		if(rowStartPos + colsPerThread > 330) {
			int end = 330 - rowStartPos;
			for(i=0;i<end;i++) {
				fppos[i] = SETLOC;
				ippos[i] = 100;
				oppos[i] = 100;
				kppos[i] = 0;
			}
		}
		else {
			for(i=0;i<colsPerThread;i++) {
				fppos[i] = SETLOC;
				ippos[i] = 100;
				oppos[i] = 100;
				kppos[i] = 0;
			}
		}
	} 
	if(blockOffset == 200 && rowStartPos <= 500 && rowStartPos + colsPerThread >=500) {
		i=500-rowStartPos;
		fppos[i] = SETLOC;
		ippos[i] = 100;
		oppos[i] = 100;
		kppos[i] = 0;
		
	}
        // Insert code to set the rest of the boundary and fixed positions
}
__global__ void doCalc(float *iplate, float *oplate, float *fplate, int ncols)
{
	/* Compute the 5 point stencil for my region */
   	int i;
	int rowStartPos;
	int blockOffset;
	float *ippos, *oppos, *fppos;
	__shared__ float oldRow[TOTCOLS];

	blockOffset = blockIdx.x * ncols;
	rowStartPos = threadIdx.x * (ncols/blockDim.x);
	int colsPerThread = ncols/blockDim.x;

	ippos = iplate + blockOffset + rowStartPos;
	oppos = oplate + blockOffset + rowStartPos;
	fppos = fplate + blockOffset + rowStartPos;
	
	for(i=0;i<colsPerThread;i++) {
		oldRow[rowStartPos+i] = oplate[blockOffset+rowStartPos+i];
	}
	__syncthreads();
		
	for(i=0; i<colsPerThread; i++) {
		if(fppos[i] != SETLOC) {
			ippos[i] = (oldRow[rowStartPos+i-1]+oldRow[rowStartPos+i+1]+oppos[i-ncols] + oppos[i+ncols] + 4*oldRow[rowStartPos+i])/8;
		}
	}		
}

__global__ void doCheck(float *iplate, float *oplate, float *fixed, int *lkeepgoing, int ncols)
{
	int i;
	int rowStartPos;
	int blockOffset;
	float *ippos, *fppos;
	int *kppos;	

	__shared__ float currentRow[TOTCOLS];

	blockOffset = blockIdx.x * ncols;
	rowStartPos = threadIdx.x * (ncols/blockDim.x);
	int colsPerThread = ncols/blockDim.x;

	ippos = iplate + blockOffset + rowStartPos;
	fppos = fixed + blockOffset + rowStartPos;
	kppos = lkeepgoing + blockOffset + rowStartPos;
	for(i=0;i<colsPerThread;i++) {
		currentRow[rowStartPos+i] = iplate[rowStartPos + blockOffset + i];
	}
	__syncthreads();

	for(i=0;i<colsPerThread;i++) {
		if(fppos[i] != SETLOC) {
			if(fabsf(currentRow[rowStartPos+i]-(currentRow[rowStartPos+i-1]+currentRow[rowStartPos+i+1]+ippos[i-ncols]+ippos[i+ncols])/4) > 0.1) {
				kppos[i] = 1;
			}
			else {
				kppos[i] = 0;
			}
		}
	}
}

__global__ void reduceSingle(int *idata, int *single, int nrows)
{
	// Reduce rows to the first element in each row
	int i;
	extern __shared__ int parts[];
	
        // Each block gets a row, each thread will reduce part of a row

        // Calculate our offset into the row
        // The number of cols per thread

	// Sum my part of one dimensional array and put it shared memory
	parts[threadIdx.x] = 0;
	for (i = threadIdx.x; i < nrows; i+=blockDim.x) {
		parts[threadIdx.x] += idata[i];
	}
	int tid = threadIdx.x;
        if (tid < 512) { parts[tid] += parts[tid + 512];}  
        __syncthreads();
        if (tid < 256) { parts[tid] += parts[tid + 256];}
        __syncthreads();
        if (tid < 128) { parts[tid] += parts[tid + 128];}
        __syncthreads();
        if (tid < 64) { parts[tid] += parts[tid + 64];}
        __syncthreads();
        if (tid < 32) { parts[tid] += parts[tid + 32];}
        __syncthreads();
	if(threadIdx.x == 0) {
		*single = 0;
		for(i = 0; i < 32; i++) {
			*single += parts[i];
		}
	}
}

__global__ void iReduceSingle(int *idata, int *single, int ncols) {
	int i;
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	unsigned int startPos = blockDim.x + threadIdx.x;
	int colsPerThread = ncols/blockDim.x;
	int myPart = 0;
	for(i=0;i<colsPerThread;i++) {
		myPart+=idata[startPos+i];
	}
	sdata[tid]=myPart;
	__syncthreads();
	
	unsigned int s;
	for(s=1;s<blockDim.x;s*=2){
		if(tid%(2*s) == 0){
			sdata[tid]+=sdata[tid+s];
		}
		__syncthreads();
	}
	if(tid==0)*single=sdata[0];
}

__global__ void iReduceSingle2(int *idata, int *single, unsigned int ncols) {
	int i;
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	unsigned int startPos = blockDim.x + threadIdx.x;
	int colsPerThread = ncols/blockDim.x;
	int myPart = 0;
	for(i=0;i<colsPerThread;i++) {
		myPart+=idata[startPos+i];
	}
	sdata[tid]=myPart;
	__syncthreads();
	
	unsigned int s;
	for(s=1;s<blockDim.x;s*=2) {
		int index = 2*s*tid;
		if(index<blockDim.x) {
			sdata[index] += sdata[index+s];
		}
		__syncthreads();
	}
	if(tid==0)*single=sdata[0];
}

__global__ void sReduceSingle(int *idata,int *single,unsigned int ncols) {
	int i;
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	unsigned int startPos = blockDim.x + threadIdx.x;
	int colsPerThread = ncols/blockDim.x;
	int myPart = 0;
	for(i=0;i<colsPerThread;i++) {
		myPart+=idata[startPos+i];
	}
	sdata[tid]=myPart;
	__syncthreads();
	
	unsigned int s;
	for(s=blockDim.x/2;s>0;s>>=1) {
		if(tid<s) {
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}
	if(tid==0)*single=sdata[0];

}

__global__ void reduceSum(int *idata, int *odata, unsigned int ncols)
{
	// Reduce rows to the first element in each row
	int i;
        int blockOffset;
        int rowStartPos;
        int colsPerThread;
        int *mypart;
	
        // Each block gets a row, each thread will reduce part of a row

	// Calculate the offset of the row
        blockOffset = blockIdx.x * ncols;
        // Calculate our offset into the row
	rowStartPos = threadIdx.x * (ncols/blockDim.x);
        // The number of cols per thread
        colsPerThread = ncols/blockDim.x;

	mypart = idata + blockOffset + rowStartPos;

	// Sum all of the elements in my thread block and put them 
        // into the first column spot
	for (i = 1; i < colsPerThread; i++) {
		mypart[0] += mypart[i];
	}
	__syncthreads(); // Wait for everyone to complete
        // Now reduce all of the threads in my block into the first spot for my row
	if(threadIdx.x == 0) {
		odata[blockIdx.x] = 0;
		for(i = 0; i < blockDim.x; i++) {
			odata[blockIdx.x] += mypart[i*colsPerThread];
		}
	}
	// We cant synchronize between blocks, so we will have to start another kernel
}

__global__ void iReduceSum(int *idata, int *odata, unsigned int ncols) {
	int i;
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	unsigned int startPos = blockDim.x + threadIdx.x;
	int colsPerThread = ncols/blockDim.x;
	int blockOffset = threadIdx.x *(ncols/blockDim.x);
	int myPart = 0;
	for(i=0;i<colsPerThread;i++) {
		myPart+=idata[blockOffset+startPos+i];
	}
	sdata[tid]=myPart;
	__syncthreads();
	
	unsigned int s;
	for(s=1;s<blockDim.x;s*=2){
		if(tid%(2*s) == 0){
			sdata[tid]+=sdata[tid+s];
		}
		__syncthreads();
	}
	if(tid==0)odata[blockIdx.x]=sdata[0];
} 

__global__ void iReduceSum2(int *idata, int *odata, unsigned int ncols) {
	int i;
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	unsigned int startPos = blockDim.x + threadIdx.x;
	int colsPerThread = ncols/blockDim.x;
	int blockOffset = threadIdx.x *(ncols/blockDim.x);
	int myPart = 0;
	for(i=0;i<colsPerThread;i++) {
		myPart+=idata[blockOffset+startPos+i];
	}
	sdata[tid]=myPart;
	__syncthreads();
	
	unsigned int s;
	for(s=1;s<blockDim.x;s*=2) {
		int index = 2*s*tid;
		if(index<blockDim.x) {
			sdata[index] += sdata[index+s];
		}
		__syncthreads();
	}
	if(tid==0)odata[blockIdx.x]=sdata[0];
}

__global__ void sReduceSum(int *idata,int *odata,unsigned int ncols) {
	int i;
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	unsigned int startPos = blockDim.x + threadIdx.x;
	int colsPerThread = ncols/blockDim.x;
	int blockOffset = threadIdx.x *(ncols/blockDim.x);
	int myPart = 0;
	for(i=0;i<colsPerThread;i++) {
		myPart+=idata[blockOffset+startPos+i];
	}
	sdata[tid]=myPart;
	__syncthreads();
	
	unsigned int s;
	for(s=blockDim.x/2;s>0;s>>=1) {
		if(tid<s) {
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}
	if(tid==0)odata[blockIdx.x]=sdata[0];

}
	
void Compute()
{
	int *keepgoing_single;
	int *keepgoing_sums;
	int keepgoing;
	int blocksize = BLOCKSIZE;
	int iteration;
	double t0, tottime;

//	double start = When();
	ncols = TOTCOLS;
	nrows = TOTROWS;

	// One block per row
	InitArrays<<< nrows, blocksize >>>(iplate, oplate, fixed, lkeepgoing, ncols);
	cudaMalloc((void **)&keepgoing_single, 1 * sizeof(int));
	keepgoing = 1;
	cudaMalloc((void **)&keepgoing_sums, nrows * sizeof(int));
 	int *peek = (int *)malloc(nrows*sizeof(int));
	
	
	for (iteration = 0; (iteration < MAXIT); iteration++)
	{
//		t0 = When();
		doCalc<<< nrows, blocksize >>>(iplate, oplate, fixed, ncols);
//		fprintf(stderr,"calc: %f\n",When()-t0);
//		t0 = When();
		doCheck<<< nrows, blocksize >>>(iplate, oplate, fixed, lkeepgoing, ncols);
//		fprintf(stderr,"check: %f\n",When()-t0);
//		t0 = When();
		iReduceSum2<<< nrows, blocksize, blocksize*sizeof(int)>>>(lkeepgoing, keepgoing_sums, ncols);
//		fprintf(stderr,"reduce: %f\n",When()-t0);
//		cudaMemcpy(peek, keepgoing_sums, nrows*sizeof(int), cudaMemcpyDeviceToHost);
//		fprintf(stderr, "after cudaMemcpy \n");
//		int i;
 //		for(i = 0; i < nrows; i++) {
//			fprintf(stderr, "%d, ",peek[i]);
//		}
		// Now we have the sum for each row in the first column, 
		//  reduce to one value
//		t0 = When();
// 		int timeit;
//		for(timeit = 0; timeit < 10000; timeit++){
//		t0 = When();
		
		iReduceSingle2<<<1, blocksize, blocksize*sizeof(int)>>>(keepgoing_sums, keepgoing_single, nrows);
//		fprintf(stderr,"reduceSingle: %f\n",When()-t0);
//		}
//		tottime = When()-t0;
		
		keepgoing = 0;
		cudaMemcpy(&keepgoing, keepgoing_single, 1 * sizeof(int), cudaMemcpyDeviceToHost);
//		tottime = When() - start;
//		fprintf(stderr, "keepgoing = %d time %f\n", keepgoing, tottime);
		//fprintf(stderr, "keepgoint[100]: %d\n", lkeepgoing[100]);
		/* swap the new value pointer with the old value pointer */
		tmp = oplate;
		oplate = iplate;
		iplate = tmp;
	}
	free(peek);
	cudaFree(keepgoing_single);
	cudaFree(keepgoing_sums);
	fprintf(stderr,"Finished in %d iterations\n", iteration);
}

/* Return the current time in seconds, using a double precision number.       */
double When()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

