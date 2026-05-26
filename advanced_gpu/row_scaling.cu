/* row_scaling.cu
 Element-wise matrix-vector multiplication (row scaling):
   result[i][j] = matrix[i][j] * vector[j]

 Input : matrix M×N, vector of size N
 Output: result matrix M×N

 Two GPU strategies:
   - RowScalingGPU    : direct global-memory version (from lecture notes)
   - RowScalingSMGPU  : caches the vector in shared memory to avoid
     repeated global reads when multiple row tiles share the same
     column range.
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
void RowScalingCPU(int *matrix, int *vector, int *result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i * cols + j] = matrix[i * cols + j] * vector[j];
}

// GPU kernel – without shared memory (from lecture).
// Uses pitched 2-D layout for coalesced access.
__global__ void RowScalingGPU(int *mat, int *vec, int *result,
                               int M, int N, int pitch)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;   // indexY in notes.
    int row = threadIdx.y + blockDim.y * blockIdx.y;   // indexX in notes.

    if (row < M && col < N)
        result[row * pitch + col] = mat[row * pitch + col] * vec[col];
}

// GPU kernel – WITH shared memory.
// Each thread block loads its tile of the vector into SM so that all
// rows in the tile share a single set of global reads for vec[].
// SM layout: shVec[blockDim.x] (one entry per column in the tile).
__global__ void RowScalingSMGPU(int *mat, int *vec, int *result,
                                 int M, int N, int pitch)
{
    extern __shared__ int shVec[];   // size = blockDim.x * sizeof(int).

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    // 1. First row of threads in this block loads the vector tile into SM.
    if (threadIdx.y == 0 && col < N)
        shVec[threadIdx.x] = vec[col];

    __syncthreads();

    // 2. All threads multiply using the cached vector value.
    if (row < M && col < N)
        result[row * pitch + col] =
            mat[row * pitch + col] * shVec[threadIdx.x];
}

// Helpers.
static void fill_random(int *arr, int len, int maxval)
{
    for (int i = 0; i < len; i++)
        arr[i] = rand() % maxval + 1;
}

static int verify(int *ref, int *gpu, int rows, int cols, int pitch,
                  const char *label)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            int rval = ref[i * cols + j];
            int gval = gpu[i * pitch + j];
            if (rval != gval) {
                printf("%s MISMATCH at [%d][%d]: ref=%d gpu=%d\n",
                       label, i, j, rval, gval);
                return 0;
            }
        }
    return 1;
}

// Main.
int main(int argc, char *argv[])
{
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;

    printf("Row scaling: matrix [%d x %d], vector [%d]\n", M, N, N);

    // Host allocation.
    int *mat_h    = (int *)malloc(M * N * sizeof(int));
    int *vec_h    = (int *)malloc(N * sizeof(int));
    int *res_cpu  = (int *)malloc(M * N * sizeof(int));

    if (!mat_h || !vec_h || !res_cpu) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    srand(42);
    fill_random(mat_h, M * N, 10);
    fill_random(vec_h, N,     10);
    memset(res_cpu, 0, M * N * sizeof(int));

    // Device allocation (pitched).
    int *mat_d, *vec_d;
    int *res_nosm_d, *res_sm_d;
    size_t pitch_bytes;

    cudaMalloc((void **)&vec_d,       N * sizeof(int));
    cudaMallocPitch((void **)&mat_d,       &pitch_bytes, N * sizeof(int), M);
    cudaMallocPitch((void **)&res_nosm_d,  &pitch_bytes, N * sizeof(int), M);
    cudaMallocPitch((void **)&res_sm_d,    &pitch_bytes, N * sizeof(int), M);
    checkCUDAError("cudaMalloc/cudaMallocPitch");

    int pitch = (int)(pitch_bytes / sizeof(int));  // int-stride.

    // Upload flat host matrix row by row into the pitched device matrix.
    for (int i = 0; i < M; i++) {
        cudaMemcpy((char *)mat_d + i * pitch_bytes,
                   mat_h + i * N,
                   N * sizeof(int),
                   cudaMemcpyHostToDevice);
    }
    cudaMemcpy(vec_d, vec_h, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy HostToDevice");

    // Kernel launch.
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);

    // CUDA timing setup.
    cudaEvent_t start_gpu, stop_gpu;
    float elapsed_gpu_ms = 0;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Without SM.
    cudaEventRecord(start_gpu, 0);
    RowScalingGPU<<<blocks, threads>>>(
        mat_d, vec_d, res_nosm_d, M, N, pitch);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_ms, start_gpu, stop_gpu);
    checkCUDAError("Kernel launch - RowScalingGPU");
    printf("GPU (Without SM) execution time: %.4f ms\n", elapsed_gpu_ms);

    // With SM – shared buffer holds one row of vector tiles (blockDim.x ints).
    int shMemSize = threads.x * sizeof(int);
    cudaEventRecord(start_gpu, 0);
    RowScalingSMGPU<<<blocks, threads, shMemSize>>>(
        mat_d, vec_d, res_sm_d, M, N, pitch);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_gpu_ms, start_gpu, stop_gpu);
    checkCUDAError("Kernel launch - RowScalingSMGPU");
    printf("GPU (With SM) execution time: %.4f ms\n", elapsed_gpu_ms);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // Copy results back (pitched).
    int *res_nosm_h = (int *)malloc(M * pitch * sizeof(int));
    int *res_sm_h   = (int *)malloc(M * pitch * sizeof(int));

    cudaMemcpy(res_nosm_h, res_nosm_d,
               M * pitch_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(res_sm_h,   res_sm_d,
               M * pitch_bytes, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy DeviceToHost");

    // CPU reference.
    clock_t start_cpu, end_cpu;
    double elapsed_cpu_ms = 0;
    start_cpu = clock();
    RowScalingCPU(mat_h, vec_h, res_cpu, M, N);
    end_cpu = clock();
    elapsed_cpu_ms = ((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0;
    printf("CPU execution time: %.4f ms\n", elapsed_cpu_ms);

    // Verify.
    int ok_nosm = verify(res_cpu, res_nosm_h, M, N, pitch, "NoSM");
    int ok_sm   = verify(res_cpu, res_sm_h,   M, N, pitch, "SM  ");
    printf("Without SM : %s\n", ok_nosm ? "CORRECT" : "MISMATCH");
    printf("With    SM : %s\n", ok_sm   ? "CORRECT" : "MISMATCH");

    // Print (small matrices only).
    if (M <= 8 && N <= 8) {
        printf("\nVector:  ");
        for (int j = 0; j < N; j++) printf("%4d ", vec_h[j]);
        printf("\n\nMatrix (GPU SM result):\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++)
                printf("%6d ", res_sm_h[i * pitch + j]);
            printf("\n");
        }
    }

    // Cleanup.
    free(mat_h); free(vec_h); free(res_cpu);
    free(res_nosm_h); free(res_sm_h);
    cudaFree(mat_d); cudaFree(vec_d);
    cudaFree(res_nosm_d); cudaFree(res_sm_d);

    return 0;
}
