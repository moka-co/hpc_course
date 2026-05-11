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

typedef struct {
    int N, NTROW, NTCOL, db;
    int PROW, PCOL;
} Config;

void print_usage(char *prog_name) {
    printf("Usage: %s [N] [NTROW] [NTCOL] [db] [PROW] [PCOL]\n", prog_name);
    printf("    Defaults: N=1024 NTROW=2 NTCOL=2 db=256 PROW=0 PCOL=0\n");
}

int parse_int(const char *arg, int default_val) {
    if(!arg) return default_val;
    char *end;
    long val = strtol(arg, &end, 10);
    if (*end != '\0' || val <= 0 || val > INT_MAX) return default_val;
    return (int)val;
}

/* --- Matrix Management --- */

double **alloc_matrix(int rows, int cols) {
    double *data = (double *)calloc(rows * cols, sizeof(double));
    if (!data) { perror("calloc"); exit(EXIT_FAILURE); }
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (!mat) { perror("malloc pointers"); free(data); exit(EXIT_FAILURE); }
    for (int i = 0; i < rows; i++) mat[i] = &data[i * cols];
    return mat;
}

void free_matrix(double **mat, int rows) {
    if (mat) {
        if (mat[0]) free(mat[0]);
        free(mat);
    }
}

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

void reset_matrix_to_zero(double **mat, int rows, int cols) {
    if (!mat || !mat[0]) return;
    memset(mat[0], 0, rows * cols * sizeof(double));
}

/* --- Multiplication routines --- */

void matmat_ikj(double **A, double **B, double **C, int i0, int k0, int j0, int ni, int nk, int nj) {
    for (int i = i0; i < i0 + ni; i++) {
        for (int k = k0; k < k0 + nk; k++) {
            for (int j = j0; j < j0 + nj; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmatblock(double **A, double **B, double **C,
                 int i_start, int k_start, int j_start,
                 int N1, int N2, int N3,
                 int db1, int db2, int db3) {
    for (int i0 = i_start; i0 < i_start + N1; i0 += db1) {
        int ib = (i0 + db1 <= i_start + N1) ? db1 : (i_start + N1) - i0;
        for (int k0 = k_start; k0 < k_start + N2; k0 += db2) {
            int kb = (k0 + db2 <= k_start + N2) ? db2 : (k_start + N2) - k0;
            for (int j0 = j_start; j0 < j_start + N3; j0 += db3) {
                int jb = (j0 + db3 <= j_start + N3) ? db3 : (int)(j_start + N3 - j0);
                matmat_ikj(A, B, C, i0, k0, j0, ib, kb, (int)jb);
            }
        }
    }
}

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
            matmatblock(A, B, C, i0, 0, j0, ni, N2, nj, db1, db2, db3);
        }
    }
}

static double **flat_to_2d(double *buf, int rows, int cols) {
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (!mat) { perror("flat_to_2d"); exit(EXIT_FAILURE); }
    for (int i = 0; i < rows; i++) mat[i] = buf + i * cols;
    return mat;
}

void matmatdist(
    MPI_Comm grid_comm,
    int LDA, int LDB, int LDC,
    double **A, double **B, double **C,
    int N1, int N2, int N3,
    int DB1, int DB2, int DB3,
    int NTrow, int NTcol
) {
    int grid_rank;
    int dims[2], periods[2], coords[2];
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_get(grid_comm, 2, dims, periods, coords);

    int row_idx = coords[0];
    int col_idx = coords[1];

    MPI_Comm row_comm;
    MPI_Comm col_comm;
    int remain_row[2] = {0, 1};
    int remain_col[2] = {1, 0};
    MPI_Cart_sub(grid_comm, remain_row, &row_comm);
    MPI_Cart_sub(grid_comm, remain_col, &col_comm);

    int local_m = N1 / dims[0];
    int local_k = N2 / dims[0];
    int local_n = N3 / dims[1];

    double *panel_a_buf = (double *)malloc(local_m * local_k * sizeof(double));
    double *panel_b_buf = (double *)malloc(local_k * local_n * sizeof(double));
    
    double **local_c = alloc_matrix(local_m, local_n);
    reset_matrix_to_zero(local_c, local_m, local_n);

    for (int k = 0; k < dims[0]; k++) {
        int root_col = (k * dims[1]) / dims[0];
        
        if (root_col == col_idx) {
            for (int i = 0; i < local_m; i++) {
                for (int j = 0; j < local_k; j++) {
                    panel_a_buf[i * local_k + j] = A[row_idx * local_m + i][k * local_k + j];
                }
            }
        }
        MPI_Bcast(panel_a_buf, local_m * local_k, MPI_DOUBLE, root_col, row_comm);
        
        if (row_idx == k) {
            for (int i = 0; i < local_k; i++) {
                for (int j = 0; j < local_n; j++) {
                    panel_b_buf[i * local_n + j] = B[k * local_k + i][col_idx * local_n + j];
                }
            }
        }
        MPI_Bcast(panel_b_buf, local_k * local_n, MPI_DOUBLE, k, col_comm);

        double **pa = flat_to_2d(panel_a_buf, local_m, local_k);
        double **pb = flat_to_2d(panel_b_buf, local_k, local_n);

        matmatthread(pa, pb, local_c, local_m, local_k, local_n, DB1, DB2, DB3, NTrow, NTcol);
        free(pa);
        free(pb);
    }

    int local_c_size = local_m * local_n;
    double *send_buf = (double *)malloc(local_c_size * sizeof(double));
    for (int i = 0; i < local_m; i++)
        for (int j = 0; j < local_n; j++)
            send_buf[i * local_n + j] = local_c[i][j];

    int world_size;
    MPI_Comm_size(grid_comm, &world_size);
    double *recv_buf = NULL;
    if (grid_rank == 0) recv_buf = (double *)malloc(world_size * local_c_size * sizeof(double));

    MPI_Gather(send_buf, local_c_size, MPI_DOUBLE, recv_buf, local_c_size, MPI_DOUBLE, 0, grid_comm);

    if (grid_rank == 0) {
        for (int r = 0; r < world_size; r++) {
            int r_coords[2];
            MPI_Cart_coords(grid_comm, r, 2, r_coords);
            int base_i = r_coords[0] * local_m;
            int base_j = r_coords[1] * local_n;
            double *block = recv_buf + r * local_c_size;
            for (int i = 0; i < local_m; i++)
                for (int j = 0; j < local_n; j++)
                    C[base_i + i][base_j + j] = block[i * local_n + j];
        }
        free(recv_buf);
    }

    free(send_buf);
    free(panel_a_buf);
    free(panel_b_buf);
    free_matrix(local_c, local_m);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        if (world_rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    Config conf = {
        .N = parse_int(argc > 1 ? argv[1] : NULL, 1024),
        .NTROW = parse_int(argc > 2 ? argv[2] : NULL, 2),
        .NTCOL = parse_int(argc > 3 ? argv[3] : NULL, 2),
        .db = parse_int(argc > 4 ? argv[4] : NULL, 256),
        .PROW = parse_int(argc > 5 ? argv[5] : NULL, 0),
        .PCOL = parse_int(argc > 6 ? argv[6] : NULL, 0)
    };

    const int N = conf.N;
    int dims[2] = {conf.PROW, conf.PCOL};
    if (dims[0] == 0 || dims[1] == 0) MPI_Dims_create(world_size, 2, dims);
    
    if (world_rank == 0) {
        if (dims[0] * dims[1] != world_size) {
            fprintf(stderr, "ERROR: PROW * PCOL (%d*%d) != world_size (%d)\n", dims[0], dims[1], world_size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (N % dims[0] != 0 || N % dims[1] != 0) {
            fprintf(stderr, "ERROR: N=%d not divisible by grid %dx%d\n", N, dims[0], dims[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Config: N=%d, Threads=%dx%d, Tile=%d, Grid=%dx%d\n", N, conf.NTROW, conf.NTCOL, conf.db, dims[0], dims[1]);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int periods[2] = {0, 0};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    double **A = NULL, **B = NULL, **Cdist = NULL;
    double **Cbasic = NULL, **Cblock = NULL, **Cthread = NULL;

    if (world_rank == 0) {
        A = alloc_matrix(N, N);
        B = alloc_matrix(N, N);
        Cbasic = alloc_matrix(N, N);
        Cblock = alloc_matrix(N, N);
        Cthread = alloc_matrix(N, N);
        Cdist = alloc_matrix(N, N);
        fill_random_matrix(A, N, N, 42u);
        fill_random_matrix(B, N, N, 43u);
    } else {
        A = alloc_matrix(N, N);
        B = alloc_matrix(N, N);
        Cdist = alloc_matrix(N, N);
    }

    MPI_Bcast(A[0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B[0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t_ikj = 0, t_block = 0, t_thread = 0, t_dist = 0, t0;

    if (world_rank == 0) {
        t0 = MPI_Wtime(); matmat_ikj(A, B, Cbasic, 0, 0, 0, N, N, N); t_ikj = MPI_Wtime() - t0;
        printf("matmat_ikj time:    %.6f s\n", t_ikj);
        t0 = MPI_Wtime(); matmatblock(A, B, Cblock, 0, 0, 0, N, N, N, conf.db, conf.db, conf.db); t_block = MPI_Wtime() - t0;
        printf("matmatblock time:   %.6f s\n", t_block);
        t0 = MPI_Wtime(); matmatthread(A, B, Cthread, N, N, N, conf.db, conf.db, conf.db, conf.NTROW, conf.NTCOL); t_thread = MPI_Wtime() - t0;
        printf("matmatthread time:  %.6f s\n", t_thread);
    }

    MPI_Barrier(grid_comm);
    t0 = MPI_Wtime();
    matmatdist(grid_comm, N, N, N, A, B, Cdist, N, N, N, conf.db, conf.db, conf.db, conf.NTROW, conf.NTCOL);
    MPI_Barrier(grid_comm);
    t_dist = MPI_Wtime() - t0;

    if (world_rank == 0) {
        printf("matmatdist time:    %.6f s\n", t_dist);
        double max_err = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double diff = Cbasic[i][j] - Cdist[i][j];
                if (diff < 0) diff = -diff;
                if (diff > max_err) max_err = diff;
            }
        }
        printf("\nMax Error: %.6e (%s)\n", max_err, max_err < 1e-9 ? "PASS" : "FAIL");
        printf("\n%-20s %12s %12s %12s\n", "Method", "Time (s)", "Speedup", "GFlop/s");
        double gflops = 2.0 * N * N * N * 1e-9;
        printf("%-20s %12.4f %12s %12.3f\n", "matmat_ikj", t_ikj, "1.00x", gflops / t_ikj);
        printf("%-20s %12.4f %12.3fx %12.3f\n", "matmatblock", t_block, t_ikj / t_block, gflops / t_block);
        printf("%-20s %12.4f %12.3fx %12.3f\n", "matmatthread", t_thread, t_ikj / t_thread, gflops / t_thread);
        printf("%-20s %12.4f %12.3fx %12.3f\n", "matmatdist", t_dist, t_ikj / t_dist, gflops / t_dist);
    }

    free_matrix(A, N); free_matrix(B, N); free_matrix(Cdist, N);
    if (world_rank == 0) {
        free_matrix(Cbasic, N); free_matrix(Cblock, N); free_matrix(Cthread, N);
    }
    MPI_Finalize();
    return 0;
}
