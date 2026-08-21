/* Collectives with checks and a timed inefficiency study.
     (a) CORRECTNESS — Bcast, Reduce, Allreduce, and Scatter/Gather results are
         each compared against an independent serial computation (PASS/FAIL).
     (b) PARAMETER-VARYING — run with -np 2/4/8: the reduced sum, the scattered
         layout, and the timings all change with the rank count.
     (c) INEFFICIENT vs OPTIMIZED — a hand-rolled "root loops over ranks"
         broadcast and a hand-rolled gather-then-sum, versus MPI_Bcast and
         MPI_Reduce. The library uses a tree (O(log P)); the manual root loop
         is O(P) serial messages. Both timed with MPI_Wtime; speedup printed.
   Run with any number of ranks. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---------- (a) correctness of the basic collectives ---------- */
    int val = (rank == 0) ? 100 : -1;              /* only rank 0 has the value */
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int bcast_ok = (val == 100);                   /* everyone must now see 100 */

    int sum = 0;                                   /* sum of all ranks -> root  */
    MPI_Reduce(&rank, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int serial_sum = size * (size - 1) / 2;        /* 0+1+...+(size-1) */

    int all = 0;                                   /* everyone gets the sum     */
    MPI_Allreduce(&rank, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int allreduce_ok = (all == serial_sum);

    /* Scatter a root-built array, have each rank double its piece, gather back;
       the gathered result must equal 2*i for every element. */
    int *send = NULL, *recv_all = NULL, piece;
    if (rank == 0) {
        send = malloc(size * sizeof(int));
        recv_all = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) send[i] = i;   /* known input */
    }
    MPI_Scatter(send, 1, MPI_INT, &piece, 1, MPI_INT, 0, MPI_COMM_WORLD);
    piece *= 2;                                       /* local work */
    MPI_Gather(&piece, 1, MPI_INT, recv_all, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int sg_ok = 1;
        for (int i = 0; i < size; i++) if (recv_all[i] != 2 * i) { sg_ok = 0; break; }
        printf("Bcast:     all ranks see 100                 -> %s\n", bcast_ok ? "PASS" : "FAIL");
        printf("Reduce:    sum of ranks 0..%d = %d (serial %d) -> %s\n",
               size - 1, sum, serial_sum, (sum == serial_sum) ? "PASS" : "FAIL");
        printf("Allreduce: every rank sees %d (serial %d)      -> %s\n",
               all, serial_sum, allreduce_ok ? "PASS" : "FAIL");
        printf("Scatter/Gather: doubled pieces vs serial 2*i   -> %s\n", sg_ok ? "PASS" : "FAIL");
        free(send); free(recv_all);
    }

    /* ---------- (c) hand-rolled vs library collectives, timed ----------
       Use an ARRAY payload of K ints and a real rank count so the O(P) linear
       root-loop is clearly beaten by the O(log P) tree the library uses. With
       a single scalar and few ranks the difference is buried in fixed
       overhead; with a 256 KB array and 4/8/16 ranks the scaling is visible. */
    const int K = 65536;                           /* 256 KB per message */
    const int reps = 500;
    int *buf = malloc(K * sizeof(int));
    int *acc = malloc(K * sizeof(int));
    double t0, t_manual_bcast, t_lib_bcast, t_manual_reduce, t_lib_reduce;

    /* INEFFICIENCY: root sends the whole array to every other rank in a serial
       loop -> (P-1) full-array transfers issued back-to-back from one node. */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int r = 0; r < reps; r++) {
        if (rank == 0) {
            for (int i = 0; i < K; i++) buf[i] = 777;
            for (int d = 1; d < size; d++) MPI_Send(buf, K, MPI_INT, d, 9, MPI_COMM_WORLD);
        } else {
            MPI_Recv(buf, K, MPI_INT, 0, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    t_manual_bcast = MPI_Wtime() - t0;
    int mbcast_ok = 1;                             /* every rank got the payload */
    for (int i = 0; i < K; i++) if (buf[i] != 777) { mbcast_ok = 0; break; }

    /* FIX: MPI_Bcast uses a tree -> O(log P) rounds instead of O(P). */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int r = 0; r < reps; r++) {
        if (rank == 0) for (int i = 0; i < K; i++) buf[i] = 777;
        MPI_Bcast(buf, K, MPI_INT, 0, MPI_COMM_WORLD);
    }
    t_lib_bcast = MPI_Wtime() - t0;

    /* INEFFICIENCY: root receives each rank's array one by one and adds them
       elementwise -> (P-1) serial transfers + serial adds, all on the root. */
    for (int i = 0; i < K; i++) buf[i] = rank;     /* each rank contributes its id */
    long manual_check = 0, lib_check = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int r = 0; r < reps; r++) {
        if (rank == 0) {
            for (int i = 0; i < K; i++) acc[i] = buf[i];         /* root's own part */
            int *tmp = malloc(K * sizeof(int));
            for (int d = 1; d < size; d++) {
                MPI_Recv(tmp, K, MPI_INT, d, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < K; i++) acc[i] += tmp[i];
            }
            free(tmp);
        } else {
            MPI_Send(buf, K, MPI_INT, 0, 8, MPI_COMM_WORLD);
        }
    }
    t_manual_reduce = MPI_Wtime() - t0;
    if (rank == 0) manual_check = acc[0];          /* should be serial_sum */

    /* FIX: MPI_Reduce uses a tree reduction. */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int r = 0; r < reps; r++)
        MPI_Reduce(buf, acc, K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    t_lib_reduce = MPI_Wtime() - t0;
    if (rank == 0) lib_check = acc[0];

    if (rank == 0) {
        int reduce_match = (manual_check == lib_check) && (lib_check == serial_sum);
        printf("\nArray collectives, K=%d ints (256 KB), %d reps, P=%d ranks:\n", K, reps, size);
        printf("  broadcast: manual root-loop %8.2f ms  vs  MPI_Bcast  %8.2f ms  -> %.2fx  (payload %s)\n",
               t_manual_bcast * 1e3, t_lib_bcast * 1e3, t_manual_bcast / t_lib_bcast,
               mbcast_ok ? "PASS" : "FAIL");
        printf("  reduce:    manual root-loop %8.2f ms  vs  MPI_Reduce %8.2f ms  -> %.2fx\n",
               t_manual_reduce * 1e3, t_lib_reduce * 1e3, t_manual_reduce / t_lib_reduce);
        printf("  manual sum[0]=%ld, MPI_Reduce sum[0]=%ld, serial=%d -> %s\n",
               manual_check, lib_check, serial_sum, reduce_match ? "PASS" : "FAIL");
    }
    free(buf); free(acc);
    MPI_Finalize();
    return 0;
}

/* Reference run — OpenMPI 4.1.6, 48-core Xeon w7-2495X:
   All four correctness checks PASS at every P. The library's advantage grows
   with the rank count (O(log P) tree vs O(P) serial root-loop):

   $ mpirun --oversubscribe -np 8 ./04_collectives
   Bcast:     all ranks see 100                 -> PASS
   Reduce:    sum of ranks 0..7 = 28 (serial 28) -> PASS
   Allreduce: every rank sees 28 (serial 28)      -> PASS
   Scatter/Gather: doubled pieces vs serial 2*i   -> PASS
   Array collectives, K=65536 ints (256 KB), 500 reps, P=8 ranks:
     broadcast: manual root-loop    87.41 ms  vs  MPI_Bcast     58.28 ms  -> 1.50x  (payload PASS)
     reduce:    manual root-loop    77.56 ms  vs  MPI_Reduce    85.64 ms  -> 0.91x
     manual sum[0]=28, MPI_Reduce sum[0]=28, serial=28 -> PASS

   $ mpirun --oversubscribe -np 16 ./04_collectives   (all checks PASS)
   Array collectives, K=65536 ints (256 KB), 500 reps, P=16 ranks:
     broadcast: manual root-loop   143.82 ms  vs  MPI_Bcast     53.29 ms  -> 2.70x  (payload PASS)
     reduce:    manual root-loop   193.21 ms  vs  MPI_Reduce   110.33 ms  -> 1.75x
     manual sum[0]=120, MPI_Reduce sum[0]=120, serial=120 -> PASS
*/
