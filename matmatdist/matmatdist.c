#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <mpi.h>

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

// matmat with ikj index order
void matmat_ikj(double **A, double **B, double **C, int i0, int k0, int j0, int ni, int nk, int nj) {
    for (int i = i0; i < i0 + ni; i++) {
        for (int k = k0; k < k0 + nk; k++) {
            for (int j = j0; j < j0 + nj; j++) {
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

static double **flat_to_2d(double *buf, int rows, int cols) {
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (!mat) { perror("flat_to_2d"); exit(EXIT_FAILURE); }
    for (int i = 0; i < rows; i++)
        mat[i] = buf + i * cols;
    return mat;
}

void matmatdist(
    MPI_Comm grid_comm, // // grid process communicator
    int LDA, int LDB, int LDC, // //Leading dimension
    double **A, double **B, double **C,
    int N1, int N2, int N3, // dim matrices
    int DB1, int DB2, int DB3, //dim blocks
    int NTrow, int NTcol //thread config
) {
    int grid_rank;
    int dims[2], periods[2], coords[2];
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_get(grid_comm, 2, dims, periods, coords);

    int row_idx = coords[0];
    int col_idx = coords[1];
    //printf("Rank %d is located at grid coordinates (%d, %d)\n", grid_rank, row_idx, col_idx);


    // Copy slice of A into local buffer local_a
    
    // 1. Compute offsets
    int rows_per_proc = N1 / dims[0];
    int cols_per_proc = N2 / dims[1];
    double **local_a = alloc_matrix(rows_per_proc, cols_per_proc);
    int offset_i = row_idx * rows_per_proc;
    int offset_j = col_idx * cols_per_proc;

    // 2. Copy
    for (int i = 0; i < rows_per_proc; i++) {
        int global_i = offset_i + i;
        for (int j = 0; j < cols_per_proc; j++) {
            int global_j = offset_j + j;
            local_a[i][j] = A[global_i][global_j];
        }
    }

    // Copy slize of B into local buffer local_b

    // 1. Compute offsets
    rows_per_proc = N2 / dims[0];
    cols_per_proc = N3 / dims[1];
    double **local_b = alloc_matrix(rows_per_proc, cols_per_proc);
    offset_i = row_idx * rows_per_proc;
    offset_j = col_idx * cols_per_proc;

    // 2. Copy slice of B
    for (int i = 0; i < rows_per_proc; i++) {
        int global_i = offset_i + i;
        for (int j = 0; j < cols_per_proc; j++) {
            int global_j = offset_j + j;
            local_b[i][j] = B[global_i][global_j];
        }
    }

    // Create row and col communicators
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    int remain_row[2] = {0, 1};
    MPI_Cart_sub(grid_comm, remain_row, &row_comm);
    int remain_col[2]= {1,0};
    MPI_Cart_sub(grid_comm, remain_col, &col_comm);

    /*
    //Test row brodcast
    int row_test_val = grid_rank;
    printf("Row_test_val, Process with rank: %d\n", row_test_val);
    //Process with rank 0 will broadcast its global rank to its row mates
    MPI_Bcast(&row_test_val, 1, MPI_INT, 0, row_comm);
    printf("[Row Test] Global Rank %d (Coord %d,%d) received %d from Row Rank 0\n",
       grid_rank, coords[0], coords[1], row_test_val);

    //Test col broadcast
    int col_test_val = grid_rank;
    // Process with col_rank 0 will broadcast its global rank to its column mates
    MPI_Bcast(&col_test_val, 1, MPI_INT, 0, col_comm);
    printf("[Col Test] Global Rank %d (Coord %d,%d) received %d from Col Rank 0\n",
       grid_rank, coords[0], coords[1], col_test_val);

    printf("----------------------------------\n");
    */
    // SUMMA stub — broadcast and accumulate per column panel
    // (full SUMMA implementation would go here)

    int rows_per_proc_a = N1 / dims[0];
    int cols_per_proc_a = N2 / dims[1];
    int rows_per_proc_b = N2/dims[0];
    int cols_per_proc_b = N3/dims[1];
    double* panel_a_buf = (double *) malloc(rows_per_proc_a*cols_per_proc_a* sizeof(double *));
    double* panel_b_buf = (double *) malloc(rows_per_proc_b*cols_per_proc_b* sizeof(double *));

    //Allocate local c
    int rows_per_proc_c = N1 / dims[0];
    int cols_per_proc_c = N3 / dims[1];

    // Allocate the local matrix
    double **local_c = alloc_matrix(rows_per_proc_c, cols_per_proc_c);
    reset_matrix_to_zero(local_c, rows_per_proc_c, cols_per_proc_c);

    for (int k = 0; k < dims[1]; k++) {
        // TODO: MPI_Bcast local_a row panel from rank with col_idx==k
        if (k == col_idx){
            for (int i=0; i<rows_per_proc_a; i++){
                for(int j=0; j<cols_per_proc_a; j++){
                    panel_a_buf[i*cols_per_proc_a +j] = local_a[i][j];
                }
            }

        }
        // Broadcast across the row
        MPI_Bcast(panel_a_buf, rows_per_proc_a * cols_per_proc_a, MPI_DOUBLE, k, row_comm);
        
        if (k==row_idx) {
            for (int i=0; i<rows_per_proc_b; i++){
                for(int j=0; j<cols_per_proc_b; j++){
                    panel_b_buf[i*cols_per_proc_b + j] = local_b[i][j];
                }
            }

        }
        MPI_Bcast(panel_b_buf, rows_per_proc_b * cols_per_proc_b, MPI_DOUBLE, k, col_comm);

        // Wrap flat buffers into double** views (zero-copy).
        double **pa = flat_to_2d(panel_a_buf, rows_per_proc_a, cols_per_proc_a);
        double **pb = flat_to_2d(panel_b_buf, rows_per_proc_b, cols_per_proc_b);

        matmatthread(pa, pb, local_c,
                     rows_per_proc_a,   /* N1 for this local multiply      */
                     cols_per_proc_a,   /* N2 (shared / inner dimension)   */
                     cols_per_proc_b,   /* N3 for this local multiply      */
                     DB1, DB2, DB3,
                     NTrow, NTcol);
        free(pa);
        free(pb);
    }

        //Gather local_c back to global C on rank 0
        int local_c_size = rows_per_proc_c * cols_per_proc_c;

        /* Pack local_c into a flat send buffer */
        double *send_buf = (double *)malloc(local_c_size * sizeof(double));
        for (int i = 0; i < rows_per_proc_c; i++){
            for (int j = 0; j < cols_per_proc_c; j++){
            send_buf[i * cols_per_proc_c + j] = local_c[i][j];
            }
        }

        int world_size;
        MPI_Comm_size(grid_comm, &world_size);

        double *recv_buf = NULL;
        if (grid_rank == 0)
            recv_buf = (double *)malloc(world_size * local_c_size * sizeof(double));

        /* All processes contribute equally sized blocks, so Gather is enough. */
        MPI_Gather(send_buf, local_c_size, MPI_DOUBLE, recv_buf, local_c_size, MPI_DOUBLE, 0, grid_comm);

        if (grid_rank == 0) {
            for (int r = 0; r < world_size; r++) {
                int r_coords[2];
                MPI_Cart_coords(grid_comm, r, 2, r_coords);
    
                int base_i = r_coords[0] * rows_per_proc_c;
                int base_j = r_coords[1] * cols_per_proc_c;
                double *block = recv_buf + r * local_c_size;
    
                for (int i = 0; i < rows_per_proc_c; i++)
                    for (int j = 0; j < cols_per_proc_c; j++)
                        C[base_i + i][base_j + j] = block[i * cols_per_proc_c + j];
            }
            free(recv_buf);
        }


    free(send_buf);
    free(panel_a_buf);
    free(panel_b_buf);
    free_matrix(local_a, rows_per_proc_a);
    free_matrix(local_b, rows_per_proc_b);
    free_matrix(local_c, rows_per_proc_c);
 
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

/* --- Main --- */

/* Arguments accepted:
- argv[1] : N, numbers of rows/columns
- argv[2] : NTROW, number of thread assigned to the row of the matrix
- argv[3] : NTCOL, same as the previous one but for columns
- argv[4] : db1,db2,db3 tile sizes for each matrix, for simplicity all with the same value
*/
int main(int argc, char *argv[]) {

    // Initial MPI Configuration
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Rank %d of %d Reporting\n", world_rank, world_size);
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);

    int periods[2] = {0, 0}; // Set to 1 for logical wrap-around (torus), 0 for flat grid
    int reorder = 1;         // Allow MPI to reorder ranks for optimization
    
    //Create the Cartesian communicator
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);

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

    if (world_rank==0){
        printf("--------------------------------------------------\n");
        printf("Configuration:\n");
        printf("  N (Size)          : %d\n", N);
        printf("  NTROW (Threads R) : %d\n", NTROW);
        printf("  NTCOL (Threads C) : %d\n", NTCOL);
        printf("  Tile Sizes (db)   : %d, %d, %d\n", db1, db2, db3);
        printf("--------------------------------------------------\n\n");
    }

    // Variables and matrix initialization
    double t_thread = 0.0;
    double t_dist = 0.0;
    double t0 = 0.0;
    double **A = alloc_matrix(N, N);
    double **B = alloc_matrix(N, N);
    double **Cbasic = alloc_matrix(N, N);
    double **Cblock = alloc_matrix(N, N);
    double **Cthread = alloc_matrix(N, N);
    double **Cdist = alloc_matrix(N,N);

    fill_random_matrix(A, N, N, 42u);
    fill_random_matrix(B, N, N, 43u);

    // Parallel Blocked - run only on rank 0
    if (world_rank == 0){
        reset_matrix_to_zero(Cthread, N, N);
        t0 = MPI_Wtime();
        matmatthread(A, B, Cthread,
                     N, N, N,
                     db1, db2, db3,
                     NTROW, NTCOL);
        t_thread = MPI_Wtime() - t0;
        printf("matmatthread time: %.6f s\n", t_thread);
    }

    MPI_Barrier(grid_comm);

    // Distributed Parallel
    t0 = MPI_Wtime();
    matmatdist(
        grid_comm, //grid process communicator
        256, 256, 256, //leadin dimension
        A,B,Cdist, //Matrices
        N,N,N, //dimension matrices
        db1, db2, db3, // Dmension blocks
        NTROW, NTCOL
    );
    MPI_Barrier(grid_comm);
    t_dist = MPI_Wtime() - t0;
    if(world_rank == 0){
        printf("matmatdist time: %.6f s\n", t_dist);
    }

    /* Correctness check */
        if (world_rank == 0) {
        double max_err = 0.0;
        double max_rel = 0.0;
        int    err_i = -1, err_j = -1;
 
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double diff = Cthread[i][j] - Cdist[i][j];
                if (diff < 0.0) diff = -diff;
                double ref  = Cthread[i][j] < 0.0 ? -Cthread[i][j] : Cthread[i][j];
                double rel  = (ref > 1e-15) ? diff / ref : diff;
                if (diff > max_err) { max_err = diff; err_i = i; err_j = j; }
                if (rel  > max_rel)   max_rel  = rel;
            }
        }
 
        printf("\n--- Correctness check (Cthread vs Cdist) ---\n");
        printf("  Max absolute error : %.6e", max_err);
        if (err_i >= 0)
            printf("  at [%d][%d]  (thread=%.6f  dist=%.6f)",
                   err_i, err_j, Cthread[err_i][err_j], Cdist[err_i][err_j]);
        printf("\n");
        printf("  Max relative error : %.6e\n", max_rel);
        if (max_rel < 1e-9)
            printf("  Result: PASS (errors within floating-point tolerance)\n");
        else
            printf("  Result: FAIL (errors exceed 1e-9 threshold)\n");
                /* Summary table */
        printf("\n%-20s %12s %12s %12s\n",
               "Method", "Time (s)", "Speedup", "GFlop/s");
        printf("%-20s %12s %12s %12s\n",
               "--------------------", "------------", "------------", "------------");
        double gflops = 2.0 * N * (double)N * N * 1e-9;
        printf("%-20s %12.4f %12s %12.3f\n",
               "matmatthread", t_thread, "1.00x (ref)", gflops / t_thread);
        printf("%-20s %12.4f %12.3fx %12.3f\n",
               "matmatdist", t_dist,
               t_thread / t_dist,
               gflops / t_dist);
    }


    // Free memory 
    free_matrix(A, N); free_matrix(B, N);
    free_matrix(Cbasic, N); free_matrix(Cblock, N); free_matrix(Cthread, N);

    MPI_Finalize();
    return 0;
}