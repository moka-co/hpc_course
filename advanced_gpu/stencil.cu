/*
 * stencil.cu
 * Computes a 1-D stencil: B[i] = sum_{j=-k}^{k} A[i+j]
 * (out-of-bounds positions contribute 0).
 *
 * Two GPU kernels are provided:
 *   - stencilWoutSMGPU : reads directly from global memory
 *   - stencilWithSMGPU : uses shared memory to reduce global reads
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
void stencilCPU(int *in, int *out, int n, int k)
{
    for (int i = 0; i < n; i++)
        for (int j = -k; j <= k; j++)
            out[i] += ((i + j) < 0 || (i + j) >= n) ? 0 : in[i + j];
}

// GPU kernel – WITHOUT shared memory.
__global__ void stencilWoutSMGPU(int *in, int *out, int n, int radius)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) return;

    int value = 0;
    for (int i = -radius; i <= radius; i++)
        value += ((index + i) < 0 || (index + i) >= n) ? 0 : in[index + i];

    out[index] = value;
}

// GPU kernel – WITH shared memory.
// Shared memory layout (size = blockDim.x + 2*radius):
// [ left halo | central elements | right halo ]
//   radius       blockDim.x        radius
__global__ void stencilWithSMGPU(int *in, int *out, int n, int radius)
{
    extern __shared__ int shMem[];

    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx  = threadIdx.x + radius;        // offset by halo.

    // 1. Every thread loads its central element.
    shMem[local_idx] = (global_idx < n) ? in[global_idx] : 0;

    // 2. First 'radius' threads also load the halo elements.
    if (threadIdx.x < radius) {
        // left halo.
        int left = global_idx - radius;
        shMem[local_idx - radius] = (left < 0) ? 0 : in[left];

        // right halo.
        int right = global_idx + blockDim.x;
        shMem[local_idx + blockDim.x] = (right >= n) ? 0 : in[right];
    }

    // 3. Wait until all threads have finished loading.
    __syncthreads();

    if (global_idx >= n) return;

    // 4. Compute sum over neighborhood from shared memory only.
    int value = 0;
    for (int i = -radius; i <= radius; i++)
        value += shMem[local_idx + i];

    // 5. Write result to global memory.
    out[global_idx] = value;
}

// Helpers.
static void fill_random(int *arr, int len, int maxval)
{
    for (int i = 0; i < len; i++)
        arr[i] = rand() % maxval + 1;
}

static int verify(int *ref, int *gpu, int n, const char *label)
{
    for (int i = 0; i < n; i++)
        if (ref[i] != gpu[i]) {
            printf("%s MISMATCH at i=%d: ref=%d gpu=%d\n",
                   label, i, ref[i], gpu[i]);
            return 0;
        }
    return 1;
}

// Main.
int main(int argc, char *argv[])
{
    int N      = (argc > 1) ? atoi(argv[1]) : 1024;
    int radius = (argc > 2) ? atoi(argv[2]) : 3;

    printf("Stencil: N=%d, radius=%d\n", N, radius);

    // Host allocation.
    size_t bytes = N * sizeof(int);
    int *in_host        = (int *)malloc(bytes);
    int *out_cpu        = (int *)calloc(N, sizeof(int));
    int *out_gpu_nosm   = (int *)calloc(N, sizeof(int));
    int *out_gpu_sm     = (int *)calloc(N, sizeof(int));

    if (!in_host || !out_cpu || !out_gpu_nosm || !out_gpu_sm) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    fill_random(in_host, N, 10);

    // Device allocation.
    int *in_device, *out_device_nosm, *out_device_sm;
    cudaMalloc((void **)&in_device,        bytes);
    cudaMalloc((void **)&out_device_nosm,  bytes);
    cudaMalloc((void **)&out_device_sm,    bytes);
    checkCUDAError("cudaMalloc");
    cudaMemset(out_device_nosm, 0, bytes);
    cudaMemset(out_device_sm,   0, bytes);
    checkCUDAError("cudaMemset");

    cudaMemcpy(in_device, in_host, bytes, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy HostToDevice");

    // Kernel configuration.
    int T = 256;                                      // threads per block.
    int nBlocks = (N + T - 1) / T;
    dim3 nThreadsPerBlock(T);

    // CUDA timing setup.
    cudaEvent_t start_gpu, stop_gpu;
    float elapsed_gpu_ms = 0;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Without SM.
    cudaEventRecord(start_gpu, 0);
    stencilWoutSMGPU<<<nBlocks, nThreadsPerBlock>>>(
        in_device, out_device_nosm, N, radius);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_ms, start_gpu, stop_gpu);
    checkCUDAError("Kernel launch - stencilWoutSMGPU");
    printf("GPU (Without SM) execution time: %.4f ms\n", elapsed_gpu_ms);

    // With SM – shared buffer needs T + 2*radius ints.
    int shMemSize = (T + 2 * radius) * sizeof(int);
    cudaEventRecord(start_gpu, 0);
    stencilWithSMGPU<<<nBlocks, nThreadsPerBlock, shMemSize>>>(
        in_device, out_device_sm, N, radius);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_ms, start_gpu, stop_gpu);
    checkCUDAError("Kernel launch - stencilWithSMGPU");
    printf("GPU (With SM) execution time: %.4f ms\n", elapsed_gpu_ms);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // Copy results back.
    cudaMemcpy(out_gpu_nosm, out_device_nosm, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_gpu_sm,   out_device_sm,   bytes, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy DeviceToHost");

    // CPU reference.
    clock_t start_cpu, end_cpu;
    double elapsed_cpu_ms = 0;
    start_cpu = clock();
    stencilCPU(in_host, out_cpu, N, radius);
    end_cpu = clock();
    elapsed_cpu_ms = ((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0;
    printf("CPU execution time: %.4f ms\n", elapsed_cpu_ms);

    // Verify.
    int ok_nosm = verify(out_cpu, out_gpu_nosm, N, "NoSM");
    int ok_sm   = verify(out_cpu, out_gpu_sm,   N, "SM  ");
    printf("Without SM : %s\n", ok_nosm ? "CORRECT" : "MISMATCH");
    printf("With    SM : %s\n", ok_sm   ? "CORRECT" : "MISMATCH");

    // Print (small arrays only).
    if (N <= 20) {
        printf("\nInput : ");
        for (int i = 0; i < N; i++) printf("%3d ", in_host[i]);
        printf("\nCPU   : ");
        for (int i = 0; i < N; i++) printf("%3d ", out_cpu[i]);
        printf("\nGPU(SM): ");
        for (int i = 0; i < N; i++) printf("%3d ", out_gpu_sm[i]);
        printf("\n");
    }

    // Cleanup.
    free(in_host); free(out_cpu); free(out_gpu_nosm); free(out_gpu_sm);
    cudaFree(in_device); cudaFree(out_device_nosm); cudaFree(out_device_sm);

    return 0;
}
