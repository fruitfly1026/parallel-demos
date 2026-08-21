/* Classic MPI deadlock and how to fix it (ring exchange with neighbors).
   This program has TWO modes selected by argv[1]:
       ./06_deadlock deadlock   -> every rank MPI_Send(large) BEFORE MPI_Recv
       ./06_deadlock fix        -> MPI_Sendrecv (default if no arg)

   WHY IT DEADLOCKS: a large MPI_Send cannot complete until the matching
   MPI_Recv is posted (the message is too big for the eager buffer, so the
   runtime uses the rendezvous protocol and MPI_Send blocks). If EVERY rank
   calls MPI_Send first, no rank ever reaches its MPI_Recv, so no send can
   complete -> all ranks block forever.

   THE FIX: MPI_Sendrecv issues the send and the receive together and lets the
   runtime order them safely (equivalently: nonblocking Irecv+Isend+Waitall,
   or an odd/even send-then-recv / recv-then-send schedule).

   Run the deadlock mode under a timeout so it demonstrably hangs:
       timeout 8 mpirun --oversubscribe -np 4 ./06_deadlock deadlock   # hangs
       mpirun --oversubscribe -np 4 ./06_deadlock fix                  # completes
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *mode = (argc > 1) ? argv[1] : "fix";
    int n = 1000000;                               /* big enough to force rendezvous */
    int *snd = malloc(n * sizeof(int));
    int *rcv = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) snd[i] = rank;     /* tag every element with my rank */
    int next = (rank + 1) % size;
    int prev = (rank + size - 1) % size;

    if (strcmp(mode, "deadlock") == 0) {
        /* BROKEN: all ranks Send first. With a large message this blocks and
           never returns -> the program hangs (the timeout will kill it). */
        if (rank == 0) printf("[deadlock mode] every rank Sends %d ints before Recv...\n", n);
        fflush(stdout);
        MPI_Send(snd, n, MPI_INT, next, 0, MPI_COMM_WORLD);   /* <-- hangs here */
        MPI_Recv(rcv, n, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank %d: (unreachable when it deadlocks)\n", rank);
    } else {
        /* FIXED: MPI_Sendrecv — one call, the runtime orders send+recv safely. */
        MPI_Sendrecv(snd, n, MPI_INT, next, 0,
                     rcv, n, MPI_INT, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* CORRECTNESS CHECK (fix path): each rank must receive its predecessor's
       rank in every element. Verify locally, then reduce to a global verdict. */
    int local_ok = 1;
    for (int i = 0; i < n; i++) if (rcv[i] != prev) { local_ok = 0; break; }
    int all_ok = 0;
    MPI_Reduce(&local_ok, &all_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Exchanged %d ints around a ring of %d ranks — no deadlock. verify -> %s\n",
               n, size, all_ok ? "PASS" : "FAIL");

    free(snd); free(rcv);
    MPI_Finalize();
    return 0;
}

/* Reference run — OpenMPI 4.1.6, 48-core Xeon w7-2495X:
   The deadlock mode really hangs (killed by the 8 s timeout, exit 124); the
   fixed mode completes and verifies at every P.

   $ timeout 8 mpirun --oversubscribe -np 4 ./06_deadlock deadlock
   [deadlock mode] every rank Sends 1000000 ints before Recv...
   (no further output — all ranks block in MPI_Send; timeout kills it, rc=124)

   $ mpirun --oversubscribe -np 4 ./06_deadlock fix
   Exchanged 1000000 ints around a ring of 4 ranks — no deadlock. verify -> PASS

   $ mpirun --oversubscribe -np 8 ./06_deadlock fix
   Exchanged 1000000 ints around a ring of 8 ranks — no deadlock. verify -> PASS
*/
