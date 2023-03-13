#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;
__global__ void prod_kernell(int *d_matrixA, int *d_matrixB, int *d_matrixE, int p, int q, int r)
{
    __shared__ int A[32][32];
    __shared__ int B[32][32];
    int row = blockIdx.y*32 + threadIdx.y;    //row in the output matrix
    int col = blockIdx.x*32 + threadIdx.x;	//column in the output matrix
    int sum = 0;
    for (int i=0;i<(q+31)/32;i++){      //ith phase will make use of shared memory for all current output indices
        if (row < p && i*32+threadIdx.x < q)    //if row is a valid row in the output matrix
            A[threadIdx.y][threadIdx.x] = d_matrixA[row*q + i*32 + threadIdx.x];    //i*32 skips 32 elements processed in i phases
        else
            A[threadIdx.y][threadIdx.x] = 0;

        if (col < r && i*32+threadIdx.y < q)   //if col is a valid column in the output matrix
            B[threadIdx.y][threadIdx.x] = d_matrixB[(i*32+threadIdx.y)*r + col];   //loads in column major fashion
        else
            B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();    //need to syncronize all the threads before they go on to accumulate the sum
        
        for(int k=0;k<32;k++)
            sum+= A[threadIdx.y][k] * B[k][threadIdx.x];
        __syncthreads();
    }
    if (row < p && col < r)
        d_matrixE[row*r + col]= sum;
}
__global__ void prodt_kernell(int* d_matrixC, int* d_matrixD, int* d_matrixE, int p, int q, int r) {
    __shared__ int C[32][32];
    __shared__ int D[32][32];
    int row = blockIdx.y*32 + threadIdx.y;   //row in the output matrix
    int col = blockIdx.x*32 + threadIdx.x;   //column in the output matrix

    int sum = 0;
    for (int i=0;i<(q+31)/32;i++) {
        if (row<p && i*32 + threadIdx.x < q) {     //if row is a valid row in the output matrix
            C[threadIdx.y][threadIdx.x] = d_matrixC[row* q + i*32 + threadIdx.x];   //Accessed in row major fashion
        } else {
            C[threadIdx.y][threadIdx.x] = 0;
        }

        if (col<r && i*32 + threadIdx.y < q) {    //if col is a valid column in the output matrix
            D[threadIdx.y][threadIdx.x] = d_matrixD[col*q + i*32 + threadIdx.y]; //reads in row major fashion to calculate the C*Dt
        } else {
            D[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();      //need to syncronize all the threads before they go on to accumulate the sum
        for (int k=0;k<32;k++) {
            sum += C[threadIdx.y][k] * D[k][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < p && col < r) {
        d_matrixE[row * r + col]+= sum;
    }
}

void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	dim3 gridDim(ceil(float(r)/32),ceil(float(p)/32),1);    //each thread calculates one element of output matrix total blocks launched r/32 and p/32 
	dim3 blockDim(32,32,1);									//max threads allowed in a block
	prod_kernell<<<gridDim,blockDim>>>(d_matrixA,d_matrixB,d_matrixE,p,q,r);   //initialises d_matrixE with AB product
	prodt_kernell<<<gridDim,blockDim>>>(d_matrixC,d_matrixD,d_matrixE,p,q,r);	//adds CDt product values to d_matrixE
  	cudaDeviceSynchronize();
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
