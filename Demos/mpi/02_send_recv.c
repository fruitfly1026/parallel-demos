/* Point-to-point: rank 0 sends one int to rank 1. Run with >= 2 ranks. */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) { if (!rank) printf("Run with >= 2 ranks\n"); MPI_Finalize(); return 0; }

    int tag = 0;
    if (rank == 0) {
        int msg = 42;
        MPI_Send(&msg, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
        printf("rank 0 sent %d to rank 1\n", msg);
    } else if (rank == 1) {
        int msg;
        MPI_Recv(&msg, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank 1 received %d from rank 0\n", msg);
    }
    MPI_Finalize();
    return 0;
}
