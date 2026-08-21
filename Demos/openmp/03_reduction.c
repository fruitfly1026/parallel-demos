/* Reduction — sum and dot-product done three ways to show WHY you need
   'reduction(+:...)' and not a bare shared accumulator.

   Build:  gcc -O3 -fopenmp 03_reduction.c -o 03_reduction
   Run:    OMP_NUM_THREADS=48 ./03_reduction

     (1) serial            -> the reference answer
     (2) shared accumulator, NO protection -> data RACE: many threads do
         sum += x concurrently (read-modify-write is not atomic), so
         updates are lost. The result is WRONG; how wrong is not defined by
         the standard and can differ across runs, machines, and compilers.
     (3) reduction(+:sum)  -> each thread accumulates a PRIVATE partial,
         the runtime combines the partials at the end. Correct and fast.

   The race in (2) is the inefficiency/bug; (3) is the fix. We also TIME
   (3) to show the correct version is not just right but cheap.           */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (100*1000*1000)   /* 100M elements */

int main(void) {
    double *x = malloc((size_t)N * sizeof *x);
    double *y = malloc((size_t)N * sizeof *y);
    if (!x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (long i = 0; i < N; i++) { x[i] = 1.0; y[i] = 2.0; }
    /* sum(x) should be exactly N; dot(x,y) should be exactly 2*N. */
    const double sum_ref = (double)N;
    const double dot_ref = 2.0 * (double)N;

    /* (1) serial reference ------------------------------------------------ */
    double t0 = omp_get_wtime();
    double sum_serial = 0.0;
    for (long i = 0; i < N; i++) sum_serial += x[i];
    double t_serial = omp_get_wtime() - t0;

    /* (2) WRONG: shared accumulator with a race --------------------------- */
    double sum_race = 0.0;
    t0 = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; i++)
        sum_race += x[i];          /* RACE: unsynchronized read-modify-write */
    double t_race = omp_get_wtime() - t0;

    /* (3) CORRECT: reduction ---------------------------------------------- */
    double sum_red = 0.0, dot_red = 0.0;
    t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_red, dot_red)
    for (long i = 0; i < N; i++) {
        sum_red += x[i];
        dot_red += x[i] * y[i];
    }
    double t_red = omp_get_wtime() - t0;

    printf("threads (max)        : %d\n", omp_get_max_threads());
    printf("reference  sum       : %.1f\n", sum_ref);
    printf("(1) serial sum       : %.1f   time %.4f s\n", sum_serial, t_serial);
    printf("(2) race   sum       : %.1f   time %.4f s   <- WRONG (lost updates)\n",
           sum_race, t_race);
    printf("(3) reduce sum       : %.1f   time %.4f s   speedup %.2fx\n",
           sum_red, t_red, t_serial / t_red);
    printf("(3) reduce dot       : %.1f   (ref %.1f)\n", dot_red, dot_ref);
    printf("correctness (race)   : %s\n", sum_race == sum_ref ? "PASS" : "FAIL");
    printf("correctness (reduce) : %s\n",
           (sum_red == sum_ref && dot_red == dot_ref) ? "PASS" : "FAIL");
    printf("note: the race sum is WRONG (lost updates) and not guaranteed\n"
           "      across runs/machines; the reduction is always exact.\n");

    free(x); free(y);
    return 0;
}

/* Reference run — 48-core Xeon w7-2495X, gcc -O3 -fopenmp, OMP_NUM_THREADS=48:

$ OMP_NUM_THREADS=48 ./03_reduction        # run 1
threads (max)        : 48
reference  sum       : 100000000.0
(1) serial sum       : 100000000.0   time 0.1091 s
(2) race   sum       : 2083333.0   time 0.0089 s   <- WRONG (lost updates)
(3) reduce sum       : 100000000.0   time 0.0144 s   speedup 7.60x
(3) reduce dot       : 200000000.0   (ref 200000000.0)
correctness (race)   : FAIL
correctness (reduce) : PASS

$ OMP_NUM_THREADS=48 ./03_reduction        # run 2
...
(2) race   sum       : 2083333.0   time 0.0099 s   <- WRONG (lost updates)
(3) reduce sum       : 100000000.0   time 0.0145 s   speedup 12.52x
correctness (race)   : FAIL
correctness (reduce) : PASS

The racing accumulator drops ~98% of the updates (100M expected, ~2M kept):
threads overwrite each other's read-modify-write. On this build it happened
to land on the same wrong value both runs, but that is not guaranteed — the
standard makes it undefined. The reduction keeps a private partial per
thread and combines them, so it is both correct AND ~8-13x faster than
serial. (The race "time" is meaningless — it is fast only because it is
computing the wrong answer.)                                               */