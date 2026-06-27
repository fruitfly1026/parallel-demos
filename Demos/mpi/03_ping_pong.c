/* Ping-pong latency between rank 0 and rank 1 for several message sizes.
   Mirrors HW1: use gettimeofday for precision at small sizes. Run with 2 ranks. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double now_us(void) {                 /* microseconds */
    struct timeval t; gettimeofday(&t, NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) { if (!rank) printf("Run with 2 ranks\n"); MPI_Finalize(); return 0; }

    const int iters = 1000;
    size_t sizes[] = { 32, 1024, 32*1024, 1024*1024 };
    for (int s = 0; s < 4; s++) {
        size_t n = sizes[s];
        char *buf = malloc(n); memset(buf, 0, n);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = now_us();
        for (int i = 0; i < iters; i++) {
            if (rank == 0) {
                MPI_Send(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }
        double rtt = (now_us() - t0) / iters;
        if (rank == 0)
            printf("%8zu bytes:  rtt = %8.2f us   one-way = %8.2f us\n", n, rtt, rtt / 2);
        free(buf);
    }
    MPI_Finalize();
    return 0;
}
