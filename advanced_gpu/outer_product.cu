/*
 * outer_product.cu
 * Computes the outer product C[i,j] = A[i] * B[j]
 * for i in [0, M-1], j in [0, N-1].
 *
 * Uses cudaMallocPitch for coalesced 2D memory access.
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
void OuterProductCPU(int *a, int *b, int *c, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            c[i * n + j] = a[i] * b[j];
}

// GPU kernel.
__global__ void Outer_ProductGPU(int *a, int *b, int *c,
                                  int pitch, int m, int n)
{
    int indexRow = threadIdx.y + blockIdx.y * blockDim.y;
    int indexCol = threadIdx.x + blockIdx.x * blockDim.x;

    if (indexRow < m && indexCol < n)
        c[indexRow * pitch + indexCol] = a[indexRow] * b[indexCol];
}

// Helpers.
static void fill_random(int *arr, int len, int maxval)
{
    for (int i = 0; i < len; i++)
        arr[i] = rand() % maxval + 1;
}

static int verify(int *ref, int *gpu, int m, int n)
{
    for (int i = 0; i < m * n; i++)
        if (ref[i] != gpu[i]) return 0;
    return 1;
}

// Main.
int main(int argc, char *argv[])
{
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;

    printf("Outer product: A[%d] x B[%d] -> C[%d x %d]\n", M, N, M, N);

    // Host allocation.
    int *A_host = (int *)malloc(M * sizeof(int));
    int *B_host = (int *)malloc(N * sizeof(int));
    int *C_host = (int *)malloc(M * N * sizeof(int));  // CPU result.
    int *C_copy = (int *)malloc(M * N * sizeof(int));  // GPU result.

    if (!A_host || !B_host || !C_host || !C_copy) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    fill_random(A_host, M, 10);
    fill_random(B_host, N, 10);
    memset(C_host, 0, M * N * sizeof(int));
    memset(C_copy, 0, M * N * sizeof(int));

    // Device allocation.
    int *A_device, *B_device;
    int *C_device;
    size_t pitch;   // Pitch in bytes returned by cudaMallocPitch.

    cudaMalloc((void **)&A_device, M * sizeof(int));
    cudaMalloc((void **)&B_device, N * sizeof(int));
    cudaMallocPitch((void **)&C_device, &pitch, N * sizeof(int), M);
    checkCUDAError("cudaMalloc/cudaMallocPitch");

    // Host to device.
    cudaMemcpy(A_device, A_host, M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy HostToDevice");

    // Kernel launch.
    dim3 nThreadsPerBlock(16, 16);
    dim3 nBlocks((N + nThreadsPerBlock.x - 1) / nThreadsPerBlock.x,
                 (M + nThreadsPerBlock.y - 1) / nThreadsPerBlock.y);

    // CUDA timers.
    cudaEvent_t start_gpu, stop_gpu;
    float elapsed_gpu_ms = 0;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Pass pitch/sizeof(int) so the kernel can stride rows correctly.
    // pitch is in bytes; dividing by sizeof(int) gives the int-stride.
    cudaEventRecord(start_gpu, 0);
    Outer_ProductGPU<<<nBlocks, nThreadsPerBlock>>>(
        A_device, B_device, C_device,
        (int)(pitch / sizeof(int)), M, N);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_ms, start_gpu, stop_gpu);
    checkCUDAError("Kernel launch - Outer_ProductGPU");
    printf("GPU execution time: %.4f ms\n", elapsed_gpu_ms);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // Device to host (2D pitched copy).
    cudaMemcpy2D(C_copy, N * sizeof(int),
                 C_device, pitch,
                 N * sizeof(int), M,
                 cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy2D DeviceToHost");

    // CPU reference.
    clock_t start_cpu, end_cpu;
    double elapsed_cpu_ms = 0;
    start_cpu = clock();
    OuterProductCPU(A_host, B_host, C_host, M, N);
    end_cpu = clock();
    elapsed_cpu_ms = ((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0;
    printf("CPU execution time: %.4f ms\n", elapsed_cpu_ms);

    // Verify.
    if (verify(C_host, C_copy, M, N))
        printf("Result: CORRECT\n");
    else
        printf("Result: MISMATCH\n");

    // Print (small matrices only).
    if (M <= 8 && N <= 8) {
        printf("\nA: ");
        for (int i = 0; i < M; i++) printf("%3d ", A_host[i]);
        printf("\nB: ");
        for (int j = 0; j < N; j++) printf("%3d ", B_host[j]);
        printf("\nC (GPU):\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++)
                printf("%4d ", C_copy[i * N + j]);
            printf("\n");
        }
    }

    // Cleanup.
    free(A_host); free(B_host); free(C_host); free(C_copy);
    cudaFree(A_device); cudaFree(B_device); cudaFree(C_device);

    return 0;
}
