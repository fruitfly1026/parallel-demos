/* Estimate pi by numerical integration of 4/(1+x^2), split across ranks,
   combined with MPI_Reduce. The classic "one reduction" parallel pattern. */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = 100000000L;
    double h = 1.0 / n, local = 0.0;
    for (long i = rank; i < n; i += size) {        /* cyclic split */
        double x = (i + 0.5) * h;
        local += 4.0 / (1.0 + x * x);
    }
    local *= h;

    double pi = 0.0;
    MPI_Reduce(&local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("pi ~= %.15f   (error %.2e)\n", pi, pi - 3.141592653589793);
    MPI_Finalize();
    return 0;
}
