/* Point-to-point: rank 0 sends a message to rank 1, which VERIFIES every
   element it received against the value the sender promised (PASS/FAIL).
   We vary the message size to show the same pattern scales. Run with >= 2 ranks. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) { if (!rank) printf("Run with >= 2 ranks\n"); MPI_Finalize(); return 0; }

    const int tag = 0;
    /* PARAMETER-VARYING: send buffers of growing length; each element i is
       seeded to a known formula so the receiver can check it independently. */
    int counts[] = { 1, 16, 1024, 65536 };
    for (int c = 0; c < 4; c++) {
        int n = counts[c];
        int *buf = malloc(n * sizeof(int));
        if (rank == 0) {
            for (int i = 0; i < n; i++) buf[i] = 42 + i;      /* known payload */
            MPI_Send(buf, n, MPI_INT, 1, tag, MPI_COMM_WORLD);
            printf("rank 0 sent %d ints (buf[0]=%d, buf[%d]=%d) to rank 1\n",
                   n, buf[0], n - 1, buf[n - 1]);
        } else if (rank == 1) {
            MPI_Recv(buf, n, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* CORRECTNESS CHECK: every received element must equal the serial
               reference value 42+i. */
            int ok = 1;
            for (int i = 0; i < n; i++) if (buf[i] != 42 + i) { ok = 0; break; }
            printf("rank 1 received %d ints, verified vs expected -> %s\n",
                   n, ok ? "PASS" : "FAIL");
        }
        free(buf);
    }
    MPI_Finalize();
    return 0;
}

/* Reference run — OpenMPI 4.1.6, 48-core Xeon w7-2495X:
   $ mpirun --oversubscribe -np 2 ./02_send_recv
   rank 0 sent 1 ints (buf[0]=42, buf[0]=42) to rank 1
   rank 0 sent 16 ints (buf[0]=42, buf[15]=57) to rank 1
   rank 0 sent 1024 ints (buf[0]=42, buf[1023]=1065) to rank 1
   rank 0 sent 65536 ints (buf[0]=42, buf[65535]=65577) to rank 1
   rank 1 received 1 ints, verified vs expected -> PASS
   rank 1 received 16 ints, verified vs expected -> PASS
   rank 1 received 1024 ints, verified vs expected -> PASS
   rank 1 received 65536 ints, verified vs expected -> PASS
*/
