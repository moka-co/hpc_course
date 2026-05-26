/*
 * dot_product.cu
 * Computes the dot product k = sum_{i=0}^{N-1} A[i] * B[i].
 *
 * Two GPU strategies:
 *   - DotProductArrayWoutSMGPU : each thread writes its partial sum to
 *     global memory; the CPU sums L*M values.
 *   - DotProductArrayWithSMGPU : threads in a block accumulate into SM;
 *     thread 0 reduces the block and writes one value to global memory;
 *     the CPU only sums L values (one per block).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void checkCUDAError(const char* msg){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess ){
        fprintf(stderr, "Cuda error: %s %s\n", msg,cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   
}

// CPU reference.
void DotProductCPU(int *a, int *b, int n, int *result)
{
    *result = 0;
    for (int i = 0; i < n; i++)
        *result += a[i] * b[i];
}

// GPU kernel – WITHOUT shared memory.
// Each thread handles a stride-sized chunk of the arrays.
// Partial sums land in PartialSums[L*M] in global memory.
__global__ void DotProductArrayWoutSMGPU(int *in1, int *in2,
                                          int *PartialSums,
                                          int k, int N)
{
    int psIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int inIndex = psIndex * k;

    if (inIndex < N) {
        for (int i = 0; i < k; i++) {
            if (inIndex + i < N)
                PartialSums[psIndex] += in1[inIndex + i] * in2[inIndex + i];
            else
                break;
        }
    }
}

// GPU kernel – WITH shared memory.
// Each thread accumulates its chunk into shMem[threadIdx.x].
// After __syncthreads(), thread 0 reduces the block's SM and writes
// a single value to PartialSums[blockIdx.x].
__global__ void DotProductArrayWithSMGPU(int *in1, int *in2,
                                          int *PartialSums,
                                          int stride, int N)
{
    extern __shared__ int shMem[];

    int psIndex  = blockIdx.x;
    int shIdx    = threadIdx.x;
    int inIndex  = (threadIdx.x + blockIdx.x * blockDim.x) * stride;

    shMem[shIdx] = 0;

    if (inIndex < N) {
        for (int i = 0; i < stride; i++) {
            if (inIndex + i < N)
                shMem[shIdx] += in1[inIndex + i] * in2[inIndex + i];
            else
                break;
        }
    }

    __syncthreads();

    // Thread 0 of each block reduces the shared-memory array.
    if (threadIdx.x == 0) {
        int value = 0;
        for (int i = 0; i < blockDim.x; i++)
            value += shMem[i];
        PartialSums[psIndex] = value;
    }
}

// Kernel configuration wrappers.
static void run_WoutSM(int *in1, int *in2,
                        int nBlocks, int nThreadsPerBlock,
                        int N, int *result)
{
    int *in1_d, *in2_d, *ps_d;
    int *ps_h;
    int total = nBlocks * nThreadsPerBlock;
    int k     = (N + total - 1) / total;   // Chunk per thread.

    cudaMalloc((void **)&in1_d, N * sizeof(int));
    cudaMalloc((void **)&in2_d, N * sizeof(int));
    cudaMalloc((void **)&ps_d,  total * sizeof(int));
    checkCUDAError("cudaMalloc");
    ps_h = (int *)calloc(total, sizeof(int));

    cudaMemcpy(in1_d, in1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in2_d, in2, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(ps_d, 0, total * sizeof(int));
    checkCUDAError("cudaMemcpy/cudaMemset");

    // CUDA timing.
    cudaEvent_t start_gpu, stop_gpu;
    float elapsed_gpu_ms = 0;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu, 0);
    DotProductArrayWoutSMGPU<<<nBlocks, nThreadsPerBlock>>>(
        in1_d, in2_d, ps_d, k, N);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_ms, start_gpu, stop_gpu);
    checkCUDAError("Kernel launch - DotProductArrayWoutSMGPU");
    printf("GPU (no SM) execution time: %.4f ms\n", elapsed_gpu_ms);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    cudaMemcpy(ps_h, ps_d, total * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy DeviceToHost");

    *result = 0;
    for (int i = 0; i < total; i++)
        *result += ps_h[i];

    free(ps_h);
    cudaFree(in1_d); cudaFree(in2_d); cudaFree(ps_d);
}

static void run_WithSM(int *in1, int *in2,
                        int nBlocks, int nThreadsPerBlock,
                        int N, int *result)
{
    int *in1_d, *in2_d, *ps_d;
    int *ps_h;
    int total  = nBlocks * nThreadsPerBlock;
    int stride = (N + total - 1) / total;  // Chunk per thread.

    cudaMalloc((void **)&in1_d, N * sizeof(int));
    cudaMalloc((void **)&in2_d, N * sizeof(int));
    cudaMalloc((void **)&ps_d,  nBlocks * sizeof(int));
    checkCUDAError("cudaMalloc");
    ps_h = (int *)calloc(nBlocks, sizeof(int));

    cudaMemcpy(in1_d, in1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in2_d, in2, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy");

    int shMemSize = nThreadsPerBlock * sizeof(int);

    // CUDA timing.
    cudaEvent_t start_gpu, stop_gpu;
    float elapsed_gpu_ms = 0;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu, 0);
    DotProductArrayWithSMGPU<<<nBlocks, nThreadsPerBlock, shMemSize>>>(
        in1_d, in2_d, ps_d, stride, N);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_ms, start_gpu, stop_gpu);
    checkCUDAError("Kernel launch - DotProductArrayWithSMGPU");
    printf("GPU (SM) execution time: %.4f ms\n", elapsed_gpu_ms);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    cudaMemcpy(ps_h, ps_d, nBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy DeviceToHost");

    *result = 0;
    for (int i = 0; i < nBlocks; i++)
        *result += ps_h[i];

    free(ps_h);
    cudaFree(in1_d); cudaFree(in2_d); cudaFree(ps_d);
}

// Helpers.
static void fill_random(int *arr, int len, int maxval)
{
    for (int i = 0; i < len; i++)
        arr[i] = rand() % maxval + 1;
}

// Main.
int main(int argc, char *argv[])
{
    int N           = (argc > 1) ? atoi(argv[1]) : 1024;
    int nBlocks     = (argc > 2) ? atoi(argv[2]) : 4;
    int nTPB        = (argc > 3) ? atoi(argv[3]) : 32;

    printf("Dot product: N=%d, blocks=%d, threads/block=%d\n",
           N, nBlocks, nTPB);

    int *A = (int *)malloc(N * sizeof(int));
    int *B = (int *)malloc(N * sizeof(int));
    if (!A || !B) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(42);
    fill_random(A, N, 5);
    fill_random(B, N, 5);

    // CPU reference.
    clock_t start_cpu, end_cpu;
    double elapsed_cpu_ms = 0;
    start_cpu = clock();
    int cpu_result = 0;
    DotProductCPU(A, B, N, &cpu_result);
    end_cpu = clock();
    elapsed_cpu_ms = ((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0;
    printf("CPU time: %.4f ms\n", elapsed_cpu_ms);

    // GPU without SM.
    int gpu_nosm = 0;
    run_WoutSM(A, B, nBlocks, nTPB, N, &gpu_nosm);

    // GPU with SM.
    int gpu_sm = 0;
    run_WithSM(A, B, nBlocks, nTPB, N, &gpu_sm);

    printf("CPU result        : %d\n", cpu_result);
    printf("GPU (no SM) result: %d  -> %s\n",
           gpu_nosm, gpu_nosm == cpu_result ? "CORRECT" : "MISMATCH");
    printf("GPU (SM)    result: %d  -> %s\n",
           gpu_sm,   gpu_sm   == cpu_result ? "CORRECT" : "MISMATCH");

    free(A); free(B);
    return 0;
}
