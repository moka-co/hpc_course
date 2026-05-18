#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void checkCUDAError(const char* msg){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess ){
        fprintf(stderr, "Cuda error: %s %s\n", msg,cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   
}

void initializeMatrix(int* matrix, int M, int N){
    int i,j;
    for(i=0;i<N;i++)
        for(j=0;j<M;j++)
            matrix[i*M+j]=i*M+j;
}

void printMatrix(int*matrix, int M, int N){
    int i,j;
    for(i=0;i<N;i++){
        for(j=0;j<M;j++)
            printf("%d ", matrix[i*M+j]);
        printf("\n");
    }
}

void equalMatrix(int* m1, int*m2, int M, int N){
    int i, j;
    for(i=0;i<N;i++){
        for(j=0;j<M;j++){
            if(m1[i*M+j]!=m2[i*M+j]){
                printf("\n\n!!! The host and device results are different. !!!\n\n");
            return;
            }
        }
    }

    printf("\nThe host and device results are the same.\n");
}

void equalMatrixPitch(int* m1, int*m2, int M, int N){
    int i, j;
    for(i=0;i<N;i++){
        for(j=0;j<M;j++){
            if(m1[i*M+j]!=m2[i*M+j]){
                printf("\n\n!!! The pitched and non-pitched device results are different. !!!\n\n");
            return;
            }
        }
    }

    printf("\n\nThe pitched and non-pitched device results are the same.\n\n");
}

void HadamardProductMatricesCPU(int *in1, int *in2, int *out, int row, int col){
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            out[i*col+j] = in1[i*col+j] * in2[i*col+j];
        }
    }
}

void ScalarProductMatricesCPU(int *in1, int k, int *out, int row, int col){
    for(int i=0; i < row; i++){
        for(int j=0; j<col; j++){
            out[i*col+j] = in1[i*col+j] * k;
        }
    }
}

__global__ void HadamardProductMatricesGPU(int *in1, int *in2, int *out, int row, int col){
    int indexRow=threadIdx.y + blockIdx.y*blockDim.y;
    int indexCol=threadIdx.x + blockIdx.x*blockDim.x;
    if(indexRow<row && indexCol<col){
        out[indexRow*col+indexCol]=in1[indexRow*col+indexCol]*in2[indexRow*col+indexCol];
    }
    
}

__global__ void ScalarProductMatricesGPU(int *in1, int k, int *out, int row, int col){
    int indexRow=threadIdx.y + blockIdx.y*blockDim.y;
    int indexCol=threadIdx.x + blockIdx.x*blockDim.x;
    if(indexRow<row && indexCol<col){
        out[indexRow*col+indexCol]=in1[indexRow*col+indexCol]*k;
    }
    
}

// Kernel variants using pitch
__global__ void HadamardProductMatricesGPU_Pitch(int *in1, int *in2, int *out, 
                                                  int row, int col, 
                                                  size_t pitch1, size_t pitch2, size_t pitchOut){
    int indexRow=threadIdx.y + blockIdx.y*blockDim.y;
    int indexCol=threadIdx.x + blockIdx.x*blockDim.x;
    if(indexRow<row && indexCol<col){
        int* row1 = (int*)((char*)in1 + indexRow*pitch1);
        int* row2 = (int*)((char*)in2 + indexRow*pitch2);
        int* rowOut = (int*)((char*)out + indexRow*pitchOut);
        rowOut[indexCol] = row1[indexCol] * row2[indexCol];
    }
}

__global__ void ScalarProductMatricesGPU_Pitch(int *in1, int k, int *out, 
                                               int row, int col, 
                                               size_t pitch1, size_t pitchOut){
    int indexRow=threadIdx.y + blockIdx.y*blockDim.y;
    int indexCol=threadIdx.x + blockIdx.x*blockDim.x;
    if(indexRow<row && indexCol<col){
        int* row1 = (int*)((char*)in1 + indexRow*pitch1);
        int* rowOut = (int*)((char*)out + indexRow*pitchOut);
        rowOut[indexCol] = row1[indexCol] * k;
    }
}

// Helper function to convert pitched memory to linear
void copyPitchedToLinear(int* pitched, int* linear, int row, int col, size_t pitch){
    for(int i = 0; i < row; i++){
        int* row_ptr = (int*)(pitched + i*pitch);
        for(int j = 0; j < col; j++){
            linear[i*col+j] = row_ptr[j];
        }
    }
}

int main(int argn, char * argv[]){
    srand(time(NULL));

    //Variables declaration
    dim3 nBlocks(1,1,1), nThreadsPerBlock(1,1,1);
    int k=2;
    int M,N;
    int *A_host, *B_host, *C_host, *Ak_host;
    int *A_device, *B_device, *C_device,*copy, *Ak_device, *Ak_copy;
    int *A_device_pitch, *B_device_pitch, *C_device_pitch, *copy_pitch, *Ak_device_pitch, *Ak_copy_pitch;
    size_t pitchA, pitchB, pitchC, pitchAk;
    int size,flag;

    // Cuda Event Declarations for Timing
    cudaEvent_t start_gpu, stop_gpu;
    float elapsed_gpu_nonpitched = 0, elapsed_gpu_pitched = 0;
    
    // CPU timing variables
    clock_t start_cpu, end_cpu;
    double elapsed_cpu_ms = 0;

    // Create CUDA events
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);


    printf("***\t MATRICES MULTIPLICATION \t***\n");

    if(argn < 7){
        printf(" Insufficient number of parameters.\n");
        printf(" Usage: : %s <M> <N> <NumThreadsPerBlock.x> <NumThreadsPerBlock.y> <flag for print> <scalar>\n", argv[0]);
        printf("Default values will be used. \n");
        nThreadsPerBlock.x=4;
        nThreadsPerBlock.y=3;
        M=9000;
        N=12000;
        flag=0;
        k = 2;
    }else{
        M=atoi(argv[1]);
        N=atoi(argv[2]);
        nThreadsPerBlock.x=atoi(argv[3]);
        nThreadsPerBlock.y=atoi(argv[4]);
        flag=atoi(argv[5]);
        k=atoi(argv[6]);
    }

    // Computation of the number of blocks
    nBlocks.y=M/nThreadsPerBlock.y + ((M%nThreadsPerBlock.y)==0?0:1);
    nBlocks.x=N/nThreadsPerBlock.x + ((N%nThreadsPerBlock.x)==0?0:1);
    size=M*N*sizeof(int);

    // print of kernel configuration
    printf("Matrices size = %d * %d\n",M, N);
    printf(" Number of threads per block = %d * %d\n", nThreadsPerBlock.y,nThreadsPerBlock.x);
    printf("Number of blocks = %d * %d\n\n",nBlocks.y,nBlocks.x);

    //Host memory allocation
    A_host=(int*)malloc(size);
    B_host=(int*)malloc(size);
    C_host=(int*)malloc(size);
    Ak_host = (int*)malloc(size);
    copy=(int*)malloc(size);
    Ak_copy = (int*)malloc(size);

    copy_pitch = (int*)malloc(size);
    Ak_copy_pitch = (int*)malloc(size);

    //device memory allocation without pitch
    cudaMalloc((void**)&A_device,size);
    cudaMalloc((void**)&B_device,size);
    cudaMalloc((void**)&C_device,size);
    cudaMalloc((void**)&Ak_device, size);

    checkCUDAError("cudaMalloc");

    //Device memory allocation with pitch
    cudaMallocPitch((void**)&A_device_pitch, &pitchA, N*sizeof(int), M);
    cudaMallocPitch((void**)&B_device_pitch, &pitchB, N*sizeof(int), M);
    cudaMallocPitch((void**)&C_device_pitch, &pitchC, N*sizeof(int), M);
    cudaMallocPitch((void**)&Ak_device_pitch, &pitchAk, N*sizeof(int), M);

    checkCUDAError("cudaMallocPitch");

    // initialize host matrices
    initializeMatrix(A_host,M, N);
    initializeMatrix(B_host,M, N);

    // copy data from host to device
    cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

    checkCUDAError("cudaMemcpy");

    // copy data from host to device using cudaMemcpy2D
    cudaMemcpy2D(A_device_pitch, pitchA, A_host, N*sizeof(int), N*sizeof(int), M, cudaMemcpyHostToDevice);
    cudaMemcpy2D(B_device_pitch, pitchB, B_host, N*sizeof(int), N*sizeof(int), M, cudaMemcpyHostToDevice);

    checkCUDAError("cudaMemcpy2D - cudaMemcpyHostToDevice");

    // --------- Hadamard Multiplication -----------
    // CPU VERSION
    start_cpu = clock();
    HadamardProductMatricesCPU(A_host, B_host, C_host, M, N);
    end_cpu = clock();
    elapsed_cpu_ms = ((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0;
    printf("Hadamard Product CPU time: %.4f ms\n", elapsed_cpu_ms);
    
    // GPU NON-PITCHED VERSION
    cudaEventRecord(start_gpu, 0);
    HadamardProductMatricesGPU<<<nBlocks, nThreadsPerBlock>>>(A_device, B_device, C_device, M, N);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_nonpitched, start_gpu, stop_gpu);
    printf("Hadamard Product GPU (non-pitched) time: %.4f ms\n", elapsed_gpu_nonpitched);
    
    // GPU PITCHED VERSION
    cudaEventRecord(start_gpu, 0);
    HadamardProductMatricesGPU_Pitch<<<nBlocks, nThreadsPerBlock>>>(A_device_pitch, B_device_pitch, C_device_pitch, M, N, pitchA, pitchB, pitchC);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_pitched, start_gpu, stop_gpu);
    printf("Hadamard Product GPU (pitched) time: %.4f ms\n", elapsed_gpu_pitched);

    // copy the results from device to host
    cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(copy_pitch, N*sizeof(int), C_device_pitch, pitchC, N*sizeof(int), M, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy2D - cudaMemcpyDeviceToHost");

    // ------------ Scalar Multiplication ----------------

    // --- CPU VERSION ---
    start_cpu = clock();
    ScalarProductMatricesCPU(A_host, k, Ak_host, M, N);
    end_cpu = clock();
    elapsed_cpu_ms = ((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0;
    printf("Scalar Product CPU time: %.4f ms\n", elapsed_cpu_ms);
    
    // --- GPU NON-PITCHED VERSION ---
    cudaEventRecord(start_gpu, 0);
    ScalarProductMatricesGPU<<<nBlocks, nThreadsPerBlock>>>(A_device, k, Ak_device, M, N);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_nonpitched, start_gpu, stop_gpu);
    printf("Scalar Product GPU (non-pitched) time: %.4f ms\n", elapsed_gpu_nonpitched);
    
    // --- GPU PITCHED VERSION ---
    cudaEventRecord(start_gpu, 0);
    ScalarProductMatricesGPU_Pitch<<<nBlocks, nThreadsPerBlock>>>(A_device_pitch, k, Ak_device_pitch, M, N, pitchA, pitchAk);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_pitched, start_gpu, stop_gpu);
    printf("Scalar Product GPU (pitched) time: %.4f ms\n", elapsed_gpu_pitched);

    //copy the results from device to host
    cudaMemcpy(Ak_copy, Ak_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(Ak_copy_pitch, N*sizeof(int), Ak_device_pitch, pitchC, N*sizeof(int), M, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy2D - cudaMemcpyDeviceToHost");

    /* ----------- Scalar Product ---------------*/

    ScalarProductMatricesCPU(A_host, k, Ak_host, M,N);
    checkCUDAError("cudaMemcpy2D - ScalarProductMatricesCPU");

    //print the matrices and the results
    if(flag==1){
        // Results
        printf("Host Results\n");
        printf("Host results - Hadamard Product\n"); printMatrix(C_host,M, N);
        printf("\nHost results - Scalar Product\n"); printMatrix(Ak_host, M,N);


        printf("\n\nHadamard Product\n\n");
        printf("\nmatrix A\n"); printMatrix(A_host,M,N);
        printf("\nmatrix B\n"); printMatrix(B_host,M,N);
        printf("\nDevice results\n\n"); printMatrix(copy,M,N);

        printf("\n\nHadamard Product - with Pitch\n");
        printf("Device results\n"); printMatrix(copy_pitch,M,N);

        //Check if they are equal
        equalMatrix(copy, C_host,M,N);

        equalMatrixPitch(copy, copy_pitch,M,N);

        printf("\n\nScalar Product\n");
        printf("matrix A\n\n"); printMatrix(A_host, M,N);
        printf("\nDevice results\n\n"); printMatrix(Ak_copy, M,N);
        printf("\nScalar Product - with Pitch\n");
        printf("Device results\n\n"); printMatrix(Ak_copy_pitch, M,N);

        //Check if they are equal
        equalMatrix(Ak_copy, Ak_host, M,N);
        equalMatrixPitch(Ak_copy, Ak_copy_pitch,M,N);

    }

    
    //host memory de-allocation
    free(A_host);
    free(B_host);
    free(C_host);
    free(copy);

    //device memory de-allocation
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    // CUDA timers de-allocation
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    exit(0);

}

