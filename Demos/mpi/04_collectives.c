/* Collectives: broadcast, reduce, allreduce. Run with any number of ranks. */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int val = (rank == 0) ? 100 : -1;             /* only rank 0 has the value */
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("rank %d after Bcast sees %d\n", rank, val);

    int sum = 0;                                   /* sum of all ranks -> root  */
    MPI_Reduce(&rank, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Reduce: sum of ranks 0..%d = %d\n", size - 1, sum);

    int all = 0;                                   /* everyone gets the sum     */
    MPI_Allreduce(&rank, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("rank %d after Allreduce sees %d\n", rank, all);

    MPI_Finalize();
    return 0;
}
