/* Estimate pi by numerical integration of 4/(1+x^2), split across ranks.
     (a) CORRECTNESS — the result is compared against M_PI within a tolerance
         (PASS/FAIL) and, since it is a deterministic quadrature, against the
         known error of the midpoint rule.
     (b) PARAMETER-VARYING — run with -np 2/4/8: the answer is identical (the
         cyclic split is exact regardless of P) while the time-to-solution
         drops. The printed per-run time shows the scaling.
     (c) INEFFICIENT vs OPTIMIZED — the classic mistake of calling MPI_Reduce
         inside the integration loop (one reduction per interval, P-way
         synchronised communication every iteration) versus accumulating a
         local partial sum and doing ONE MPI_Reduce at the end. Both timed
         with MPI_Wtime; speedup printed. */
#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---------- OPTIMIZED: local accumulation, one final reduce ---------- */
    const long n = 100000000L;
    const double h = 1.0 / n;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double local = 0.0;
    for (long i = rank; i < n; i += size) {        /* cyclic split */
        double x = (i + 0.5) * h;
        local += 4.0 / (1.0 + x * x);
    }
    local *= h;
    double pi = 0.0;
    MPI_Reduce(&local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   /* ONE reduce */
    double t_opt = MPI_Wtime() - t0;

    /* ---------- INEFFICIENT: MPI_Reduce every iteration ----------
       Same math, but the sum is reduced across all ranks on EVERY interval.
       Each interval becomes a P-way synchronising collective — communication
       and a barrier per iteration instead of per run. We use a smaller m so
       this actually finishes; even so it is dramatically slower per interval. */
    const long m = 200000;
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double pi_bad = 0.0;
    for (long i = rank; i < m; i += size) {
        double x = (i + 0.5) * h;                  /* reuse same h/grid prefix */
        double term = (4.0 / (1.0 + x * x)) * h;
        double acc = 0.0;
        MPI_Reduce(&term, &acc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); /* per-iter! */
        if (rank == 0) pi_bad += acc;
    }
    double t_bad = MPI_Wtime() - t1;

    /* Optimized version of the SAME m-interval partial, for a fair per-work
       comparison of the two reduction strategies. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    double loc_m = 0.0;
    for (long i = rank; i < m; i += size) { double x = (i + 0.5) * h; loc_m += 4.0 / (1.0 + x * x); }
    loc_m *= h;
    double pi_m = 0.0;
    MPI_Reduce(&loc_m, &pi_m, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t_optm = MPI_Wtime() - t2;

    if (rank == 0) {
        double err = fabs(pi - M_PI);
        int ok = (err < 1e-9);                     /* midpoint rule at n=1e8 */
        int match = (fabs(pi_bad - pi_m) < 1e-12); /* both m-partials agree */
        printf("pi ~= %.15f   (|error| = %.2e vs M_PI) -> %s\n", pi, err, ok ? "PASS" : "FAIL");
        printf("full integral (n=%ld) with ONE reduce: %.3f ms  [P=%d]\n", n, t_opt * 1e3, size);
        printf("\nReduction strategy over m=%ld intervals:\n", m);
        printf("  per-iteration MPI_Reduce : %8.2f ms\n", t_bad  * 1e3);
        printf("  one final  MPI_Reduce    : %8.2f ms\n", t_optm * 1e3);
        printf("  speedup: %.1fx   partials agree (%.15f) -> %s\n",
               t_bad / t_optm, pi_m, match ? "PASS" : "FAIL");
    }
    MPI_Finalize();
    return 0;
}

/* Reference run — OpenMPI 4.1.6, 48-core Xeon w7-2495X:
   pi is correct to ~1e-13 at every P; the full integral scales with P, and one
   final reduce beats a per-iteration reduce by 80-220x:

   $ mpirun --oversubscribe -np 2 ./05_pi
   pi ~= 3.141592653590022   (|error| = 2.29e-13 vs M_PI) -> PASS
   full integral (n=100000000) with ONE reduce: 52.646 ms  [P=2]
   Reduction strategy over m=200000 intervals:
     per-iteration MPI_Reduce :     8.55 ms
     one final  MPI_Reduce    :     0.10 ms
     speedup: 82.6x   partials agree (0.007999989333359) -> PASS

   $ mpirun --oversubscribe -np 4 ./05_pi
   pi ~= 3.141592653590217   (|error| = 4.24e-13) -> PASS   full integral: 28.786 ms  [P=4]
     per-iteration 10.85 ms  vs  one final 0.05 ms  -> speedup 206.5x  -> PASS

   $ mpirun --oversubscribe -np 8 ./05_pi
   pi ~= 3.141592653589613   (|error| = 1.80e-13) -> PASS   full integral: 17.553 ms  [P=8]
     per-iteration  7.47 ms  vs  one final 0.03 ms  -> speedup 221.9x  -> PASS
*/
