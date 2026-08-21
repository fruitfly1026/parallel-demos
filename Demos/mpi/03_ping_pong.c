/* Ping-pong between rank 0 and rank 1.  Three things are demonstrated:
     (a) CORRECTNESS  — rank 0 sends a known pattern, rank 1 echoes it, rank 0
         verifies every byte came back unchanged (PASS/FAIL).
     (b) PARAMETER-VARYING — the round-trip is timed with MPI_Wtime across a
         range of message sizes; latency is flat for tiny messages and grows
         with bandwidth for large ones.
     (c) INEFFICIENT vs OPTIMIZED — moving N ints as N separate one-int
         messages (latency-bound, "chatty") versus one aggregated message of N
         ints (bandwidth-bound). Each MPI_Send pays the ~1.5 us message latency
         once; sending element-by-element multiplies that by N. Aggregating the
         payload into a single message is the fix. Both are timed with
         MPI_Wtime and their received data verified.
   Run with 2 ranks. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) { if (!rank) printf("Run with 2 ranks\n"); MPI_Finalize(); return 0; }

    /* ---------- (a)+(b) latency / correctness sweep ---------- */
    const int iters = 1000;
    size_t sizes[] = { 32, 1024, 32*1024, 1024*1024 };
    for (int s = 0; s < 4; s++) {
        size_t n = sizes[s];
        unsigned char *buf = malloc(n);
        for (size_t i = 0; i < n; i++) buf[i] = (unsigned char)((i * 31 + 7) & 0xFF);
        int ok = 1;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int i = 0; i < iters; i++) {
            if (rank == 0) {
                MPI_Send(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }
        double rtt_us = (MPI_Wtime() - t0) * 1e6 / iters;
        if (rank == 0) {                       /* verify the echo is intact */
            for (size_t i = 0; i < n; i++)
                if (buf[i] != (unsigned char)((i * 31 + 7) & 0xFF)) { ok = 0; break; }
            printf("%8zu bytes:  rtt = %8.2f us   one-way = %8.2f us   echo -> %s\n",
                   n, rtt_us, rtt_us / 2, ok ? "PASS" : "FAIL");
        }
        free(buf);
    }

    /* ---------- (c) chatty (N messages) vs aggregated (1 message) ---------- */
    const int N = 50000;                         /* ints to move rank0 -> rank1 */
    int *src = malloc(N * sizeof(int));
    int *dst = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) src[i] = 7 * i + 3;   /* known payload */

    /* INEFFICIENCY: one MPI_Send per element. N round trips of the message-
       passing machinery; runtime is N * per-message latency, independent of
       how little data each message carries. */
    memset(dst, 0, N * sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);
    double tc = MPI_Wtime();
    if (rank == 0)
        for (int i = 0; i < N; i++) MPI_Send(&src[i], 1, MPI_INT, 1, 3, MPI_COMM_WORLD);
    else if (rank == 1)
        for (int i = 0; i < N; i++) MPI_Recv(&dst[i], 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double t_chatty = MPI_Wtime() - tc;
    int chatty_ok = 1;
    if (rank == 1) for (int i = 0; i < N; i++) if (dst[i] != 7 * i + 3) { chatty_ok = 0; break; }

    /* FIX: aggregate the whole payload into ONE message. Pay the latency once;
       the rest is bandwidth. */
    memset(dst, 0, N * sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);
    double ta = MPI_Wtime();
    if (rank == 0)      MPI_Send(src, N, MPI_INT, 1, 4, MPI_COMM_WORLD);
    else if (rank == 1) MPI_Recv(dst, N, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double t_agg = MPI_Wtime() - ta;
    int agg_ok = 1;
    if (rank == 1) for (int i = 0; i < N; i++) if (dst[i] != 7 * i + 3) { agg_ok = 0; break; }

    /* rank 1 did the verification; report from there (it holds the results). */
    if (rank == 1) {
        printf("\nMove %d ints rank0->rank1:\n", N);
        printf("  chatty  (%d one-int msgs): %8.2f ms   verify -> %s\n",
               N, t_chatty * 1e3, chatty_ok ? "PASS" : "FAIL");
        printf("  aggregated (1 msg)       : %8.2f ms   verify -> %s\n",
               t_agg * 1e3, agg_ok ? "PASS" : "FAIL");
        printf("  speedup: %.1fx\n", t_chatty / t_agg);
    }
    free(src); free(dst);
    MPI_Finalize();
    return 0;
}

/* Reference run — OpenMPI 4.1.6, 48-core Xeon w7-2495X:
   $ mpirun --oversubscribe -np 2 ./03_ping_pong
         32 bytes:  rtt =     1.61 us   one-way =     0.80 us   echo -> PASS
       1024 bytes:  rtt =     2.83 us   one-way =     1.41 us   echo -> PASS
      32768 bytes:  rtt =    24.32 us   one-way =    12.16 us   echo -> PASS
    1048576 bytes:  rtt =   369.18 us   one-way =   184.59 us   echo -> PASS

   Move 50000 ints rank0->rank1:
     chatty  (50000 one-int msgs):     4.21 ms   verify -> PASS
     aggregated (1 msg)       :     0.02 ms   verify -> PASS
     speedup: 172.2x
*/
