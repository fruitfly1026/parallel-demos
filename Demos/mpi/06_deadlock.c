/* Classic MPI deadlock and how to fix it (ring exchange with neighbors).
   With BLOCKING MPI_Send of a large message, if every rank Sends first,
   no one reaches Recv -> deadlock. Two safe fixes are shown. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000000;                               /* big enough to deadlock */
    int *snd = malloc(n * sizeof(int));
    int *rcv = malloc(n * sizeof(int));
    int next = (rank + 1) % size;
    int prev = (rank + size - 1) % size;

    /* --- DEADLOCK (everyone Sends first): uncomment to see it hang ---
    MPI_Send(snd, n, MPI_INT, next, 0, MPI_COMM_WORLD);
    MPI_Recv(rcv, n, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    */

    /* --- FIX 1: MPI_Sendrecv — one call, the runtime orders it safely --- */
    MPI_Sendrecv(snd, n, MPI_INT, next, 0,
                 rcv, n, MPI_INT, prev, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* --- FIX 2: nonblocking Irecv + Isend + Waitall (alternative) ---
    MPI_Request req[2];
    MPI_Irecv(rcv, n, MPI_INT, prev, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Isend(snd, n, MPI_INT, next, 0, MPI_COMM_WORLD, &req[1]);
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    */

    if (rank == 0)
        printf("Exchanged %d ints around a ring of %d ranks — no deadlock.\n", n, size);
    free(snd); free(rcv);
    MPI_Finalize();
    return 0;
}
