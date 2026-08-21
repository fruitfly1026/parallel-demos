/* Hello, MPI — every rank prints who it is and where it runs, then the ranks
   prove they form a complete, unique set 0..size-1 via a gather + check.
   Run:  mpirun -n 4 ./01_hello                                             */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size, len;
    char host[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* my id          */
    MPI_Comm_size(MPI_COMM_WORLD, &size);   /* how many of us */
    MPI_Get_processor_name(host, &len);
    printf("Hello from rank %d of %d on %s\n", rank, size, host);

    /* CORRECTNESS CHECK: gather every rank's id at root and verify the set is
       exactly {0,1,...,size-1} with no gaps and no duplicates. A serial
       reference (the loop below) is compared against what MPI actually
       delivered. Behavior VARIES with -np: rerun with 2/4/8 to see the count
       and the checked set grow. */
    int *ranks = NULL;
    if (rank == 0) ranks = malloc(size * sizeof(int));
    MPI_Gather(&rank, 1, MPI_INT, ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int ok = 1, expected_sum = 0, got_sum = 0;
        for (int i = 0; i < size; i++) { expected_sum += i; got_sum += ranks[i]; }
        for (int i = 0; i < size; i++) if (ranks[i] != i) ok = 0; /* gather is ordered */
        printf("CHECK: gathered %d ranks, sum=%d (serial expects %d) -> %s\n",
               size, got_sum, expected_sum,
               (ok && got_sum == expected_sum) ? "PASS" : "FAIL");
        free(ranks);
    }
    MPI_Finalize();
    return 0;
}

/* Reference run — OpenMPI 4.1.6, 48-core Xeon w7-2495X:
   $ mpirun --oversubscribe -np 2 ./01_hello
   Hello from rank 0 of 2 on jli256-ub01.csc.ncsu.edu
   Hello from rank 1 of 2 on jli256-ub01.csc.ncsu.edu
   CHECK: gathered 2 ranks, sum=1 (serial expects 1) -> PASS

   $ mpirun --oversubscribe -np 4 ./01_hello
   Hello from rank 0 of 4 on jli256-ub01.csc.ncsu.edu   (ranks 1..3 similar)
   CHECK: gathered 4 ranks, sum=6 (serial expects 6) -> PASS

   $ mpirun --oversubscribe -np 8 ./01_hello
   Hello from rank 0 of 8 on jli256-ub01.csc.ncsu.edu   (ranks 1..7 similar)
   CHECK: gathered 8 ranks, sum=28 (serial expects 28) -> PASS
*/
