#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <errno.h>
#include <limits.h>
#include <string.h>

#define RAND_MAX_VAL 100

/* --- Input Parsing --- */

// Define a struct to manage input
typedef struct {
    int N, NTROW, NTCOL, db;
} Config;

// Print help message when specifying "-h"
void print_usage(char *prog_name) {
    printf("Usage: %s [N] [NTROW] [NTCOL] [db]\n", prog_name);
    printf("    Defaults: N=1024 NTROW=2 NTCOL=2 db=256\n");
}

// Parse integer arguments
int parse_int(const char *arg, int default_val) {
    if(!arg){
        return default_val;
    }
    char *end;
    long val = strtol(arg, &end, 10);
    if (*end != '\0' || val <=0 || val > INT_MAX) {
        fprintf(stderr, "Invalid argument '%s', using default: %d\n", arg, default_val);
        return default_val;
    }
    return (int)val;
}

/* --- Matrix Management --- */
//Allocate memory for matrices
double **alloc_matrix(int rows, int cols) {
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (!mat) { perror("malloc rows"); exit(EXIT_FAILURE); }
    for (int i = 0; i < rows; i++) {
        mat[i] = (double *)calloc(cols, sizeof(double));
        if (!mat[i]) { perror("malloc cols"); exit(EXIT_FAILURE); }
    }
    return mat;
}

//Free matrices
void free_matrix(double **mat, int rows) {
    for (int i = 0; i < rows; i++) free(mat[i]);
    free(mat);
}

//Fill matrices with random numbers
void fill_random_matrix(double **mat, int rows, int cols, unsigned int master_seed) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = master_seed + (unsigned int)tid;
        #pragma omp for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = (rand_r(&seed) % RAND_MAX_VAL) + (rand_r(&seed) % 100) / 100.0;
            }
        }
    }
}

//Utility function for visualizing matrices
void print_matrix(double **mat, int rows, int cols, const char *label) {
    printf("\n%s (%d x %d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) printf("%6.2f ", mat[i][j]);
        printf("\n");
    }
}

//Reset matrix to zero
void reset_matrix_to_zero(double **mat, int rows, int cols){
    for (int i=0; i < rows; i++){
        for (int j=0; j < cols; j++){
            mat[i][j]=0.0;
        }
    }
}

/* --- Multiplication routines --- */

//1. matmat with ikj index order
void matmat_ikj(double **A, double **B, double **C, int i0, int k0, int j0, int ni, int nk, int nj) {
    for (int i = i0; i < i0 + ni; i++) {
        for (int k = k0; k < k0 + nk; k++) {
            for (int j = j0; j < j0 + nj; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//2. matmat with ijk index order
void matmat_ijk(double **A, double **B, double **C, int i0, int k0, int j0, int ni, int nk, int nj) {
    for (int i = i0; i < i0 + ni; i++) {
        for (int j = j0; j < j0 + nj; j++) {
            for (int k = k0; k < k0 + nk; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//3. matmat with jik index order
void matmat_jik(double **A, double **B, double **C, int i0, int k0, int j0, int ni, int nk, int nj) {
    for (int j = j0; j < j0 + nj; j++) {
        for (int i = i0; i < i0 + ni; i++) {
            for (int k = k0; k < k0 + nk; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//4. matmat with jki index order
void matmat_jki(double **A, double **B, double **C, int i0, int k0, int j0, int ni, int nk, int nj) {
    for (int j = j0; j < j0 + nj; j++) {
        for (int k = k0; k < k0 + nk; k++) {
            for (int i = i0; i < i0 + ni; i++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//5. matmat with kij order
void matmat_kij(double **A, double **B, double **C, int i0, int k0, int j0, int ni, int nk, int nj) {
    for (int k = k0; k < k0 + nk; k++) {
        for (int i = i0; i < i0 + ni; i++) {
            for (int j = j0; j < j0 + nj; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//6. matmat with kji order
void matmat_kji(double **A, double **B, double **C, int i0, int k0, int j0, int ni, int nk, int nj) {
    for (int k = k0; k < k0 + nk; k++) {
        for (int j = j0; j < j0 + nj; j++) {
            for (int i = i0; i < i0 + ni; i++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Blocked multiplication - breaks a range into smaller sub-tiles
void matmatblock(double **A, double **B, double **C,
                 int i_start, int k_start, int j_start,
                 int N1, int N2, int N3,
                 int db1, int db2, int db3) {
    for (int i0 = i_start; i0 < i_start + N1; i0 += db1) {
        int ib = (i0 + db1 <= i_start + N1) ? db1 : (i_start + N1) - i0;

        for (int k0 = k_start; k0 < k_start + N2; k0 += db2) {
            int kb = (k0 + db2 <= k_start + N2) ? db2 : (k_start + N2) - k0;

            for (int j0 = j_start; j0 < j_start + N3; j0 += db3) {
                int jb = (j0 + db3 <= j_start + N3) ? db3 : (j_start + N3) - j0;

                matmat_ikj(A, B, C, i0, k0, j0, ib, kb, jb);
            }
        }
    }
}

// Parallel thread handler
void matmatthread(double **A, double **B, double **C,
                  int N1, int N2, int N3,
                  int db1, int db2, int db3,
                  int NTROW, int NTCOL) {
    int trow = N1 / NTROW;
    int tcol = N3 / NTCOL;

    #pragma omp parallel for collapse(2) num_threads(NTROW * NTCOL) \
        shared(A, B, C, N1, N2, N3, trow, tcol, db1, db2, db3)
    for (int ti = 0; ti < NTROW; ti++) {
        for (int tj = 0; tj < NTCOL; tj++) {
            int i0 = ti * trow;
            int ni = (ti == NTROW - 1) ? N1 - i0 : trow;

            int j0 = tj * tcol;
            int nj = (tj == NTCOL - 1) ? N3 - j0 : tcol;

            // Each thread computes its assigned slice of C using blocking
            matmatblock(A, B, C, i0, 0, j0, ni, N2, nj, db1, db2, db3);
        }
    }
}

/* --- Main --- */

/* Arguments accepted:
- argv[1] : N, numbers of rows/columns
- argv[2] : NTROW, number of thread assigned to the row of the matrix
- argv[3] : NTCOL, same as the previous one but for columns
- argv[4] : db1,db2,db3 tile sizes for each matrix, for simplicity all with the same value
*/
int main(int argc, char *argv[]) {

    //Parse "-h" and "--help" arguments
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)){
        print_usage(argv[0]);
        return 0;
    }
    
    Config conf = {
        .N = parse_int(argc > 1 ? argv[1] : NULL, 1024),
        .NTROW = parse_int(argc > 2 ? argv[2] : NULL, 2),
        .NTCOL = parse_int(argc > 3 ? argv[3] : NULL, 2),
        .db = parse_int(argc > 4 ? argv[4] : NULL, 256)
    };
    const int N = conf.N;
    const int NTROW = conf.NTROW;
    const int NTCOL = conf.NTCOL;
    const int db1 = conf.db;
    const int db2 = db1;
    const int db3= db1;

    printf("--------------------------------------------------\n");
    printf("Configuration:\n");
    printf("  N (Size)          : %d\n", N);
    printf("  NTROW (Threads R) : %d\n", NTROW);
    printf("  NTCOL (Threads C) : %d\n", NTCOL);
    printf("  Tile Sizes (db)   : %d, %d, %d\n", db1, db2, db3);
    printf("--------------------------------------------------\n\n");


    // Variables and matrix initialization
    double t0 = 0.0;
    double **A = alloc_matrix(N, N);
    double **B = alloc_matrix(N, N);
    double **Cbasic = alloc_matrix(N, N);
    double **Cblock = alloc_matrix(N, N);
    double **Cthread = alloc_matrix(N, N);

    fill_random_matrix(A, N, N, 42u);
    fill_random_matrix(B, N, N, 43u);

    // Basic Sequential Matrix-Matrix multiplication, all 6 variants
    // 1. ijk order
    t0 = omp_get_wtime();
    matmat_ijk(A, B, Cbasic, 0, 0, 0, N, N, N);
    printf("matmat ijk index time: %.6f s\n", omp_get_wtime() - t0);
    reset_matrix_to_zero(Cbasic, N, N);

    // 2. ikj order
    t0 = omp_get_wtime();
    matmat_ikj(A, B, Cbasic, 0, 0, 0, N, N, N);
    printf("matmat ikj index time: %.6f s\n", omp_get_wtime() - t0);
    reset_matrix_to_zero(Cbasic, N, N);

    // 3. jik order
    t0 = omp_get_wtime();
    matmat_jik(A, B, Cbasic, 0, 0, 0, N, N, N);
    printf("matmat jik index time: %.6f s\n", omp_get_wtime() - t0);
    reset_matrix_to_zero(Cbasic, N, N);

    // 4. jki order
    t0 = omp_get_wtime();
    matmat_jki(A, B, Cbasic, 0, 0, 0, N, N, N);
    printf("matmat jki index time: %.6f s\n", omp_get_wtime() - t0);
    reset_matrix_to_zero(Cbasic, N, N);

    // 5. kij order
    t0 = omp_get_wtime();
    matmat_kij(A, B, Cbasic, 0, 0, 0, N, N, N);
    printf("matmat kij index time: %.6f s\n", omp_get_wtime() - t0);
    reset_matrix_to_zero(Cbasic, N, N);

    // 6. kji order
    t0 = omp_get_wtime();
    matmat_kji(A, B, Cbasic, 0, 0, 0, N, N, N);
    printf("matmat kji index time: %.6f s\n", omp_get_wtime() - t0);

    /* -------------------------- */

    // Blocked Sequential
    t0 = omp_get_wtime();
    matmatblock(A, B, Cblock, 0, 0, 0, N, N, N, db1, db2, db3);
    printf("matmatblock time : %.6f s\n", omp_get_wtime() - t0);

    // Parallel Blocked
    t0 = omp_get_wtime();
    matmatthread(A, B, Cthread, N, N, N, db1, db2, db3, NTROW, NTCOL);
    printf("matmatthread time: %.6f s\n", omp_get_wtime() - t0);

    // Verification, should be close to zero if the logic is correct
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double diff = (Cbasic[i][j] - Cthread[i][j]);
            if (diff < 0) diff = -diff;
            if (diff > max_err) max_err = diff;
        }
    }
    printf("\nMax Error: %.2e %s\n", max_err, max_err < 1e-9 ? "(PASS)" : "(FAIL)");

    // Free memory 
    free_matrix(A, N); free_matrix(B, N);
    free_matrix(Cbasic, N); free_matrix(Cblock, N); free_matrix(Cthread, N);

    return 0;
}