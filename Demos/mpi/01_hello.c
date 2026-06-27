/* Hello, MPI — every rank prints who it is and where it runs.
   Run:  mpirun -n 4 ./01_hello                                   */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size, len;
    char host[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* my id        */
    MPI_Comm_size(MPI_COMM_WORLD, &size);   /* how many of us */
    MPI_Get_processor_name(host, &len);
    printf("Hello from rank %d of %d on %s\n", rank, size, host);
    MPI_Finalize();
    return 0;
}
