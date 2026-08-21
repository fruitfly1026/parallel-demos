/* Loop scheduling — the SAME load-imbalanced loop under three schedules.

   Build:  gcc -O3 -fopenmp 04_schedule.c -o 04_schedule
   Run:    OMP_NUM_THREADS=48 ./04_schedule

   The work per iteration grows with i (inner cost ~ i), so early iterations
   are cheap and late ones are expensive — a triangular load imbalance.

     schedule(static)        : contiguous equal-sized blocks assigned up
         front. The thread that gets the high-i block does most of the work
         while others sit idle -> slowest. This is the inefficiency.
     schedule(dynamic,chunk) : threads grab chunks from a shared queue as
         they finish, so fast threads keep pulling more work -> balanced.
     schedule(guided)        : like dynamic but chunk size starts large and
         shrinks, cutting scheduling overhead while still balancing.

   Fix for imbalance = dynamic/guided. All schedules compute the same sum,
   which we check against a serial reference (PASS/FAIL).                    */
#include <omp.h>
#include <stdio.h>
#include <math.h>

#define N 40000            /* outer iterations */

/* Cost grows with i: inner loop runs i times. Returns a value so the
   compiler cannot optimize the work away. */
static double work(long i) {
    double s = 0.0;
    for (long k = 0; k < i; k++) s += sin((double)k) * 1e-9;
    return s;
}

static double run_static(void) {
    double sum = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (long i = 0; i < N; i++) sum += work(i);
    return sum;
}
static double run_dynamic(int chunk, double *sum_out) {
    double sum = 0.0, t0 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:sum)
    for (long i = 0; i < N; i++) sum += work(i);
    (void)chunk; *sum_out = sum; return omp_get_wtime() - t0;
}
static double run_guided(double *sum_out) {
    double sum = 0.0, t0 = omp_get_wtime();
    #pragma omp parallel for schedule(guided) reduction(+:sum)
    for (long i = 0; i < N; i++) sum += work(i);
    *sum_out = sum; return omp_get_wtime() - t0;
}

int main(void) {
    /* serial reference */
    double t0 = omp_get_wtime(), ref = 0.0;
    for (long i = 0; i < N; i++) ref += work(i);
    double t_ser = omp_get_wtime() - t0;

    double s_stat, s_dyn, s_gui;
    t0 = omp_get_wtime(); s_stat = run_static(); double t_stat = omp_get_wtime() - t0;
    double t_dyn = run_dynamic(64, &s_dyn);
    double t_gui = run_guided(&s_gui);

    printf("threads (max)        : %d\n", omp_get_max_threads());
    printf("serial               : %.4f s   (reference sum %.6e)\n", t_ser, ref);
    printf("schedule(static)     : %.4f s   speedup %.2fx   %s\n",
           t_stat, t_ser / t_stat, fabs(s_stat-ref) < 1e-6*fabs(ref) ? "PASS":"FAIL");
    printf("schedule(dynamic,64) : %.4f s   speedup %.2fx   %s\n",
           t_dyn,  t_ser / t_dyn,  fabs(s_dyn -ref) < 1e-6*fabs(ref) ? "PASS":"FAIL");
    printf("schedule(guided)     : %.4f s   speedup %.2fx   %s\n",
           t_gui,  t_ser / t_gui,  fabs(s_gui -ref) < 1e-6*fabs(ref) ? "PASS":"FAIL");
    printf("note: dynamic/guided beat static because they rebalance the\n"
           "      triangular load; static leaves the high-i thread a bottleneck.\n");
    return 0;
}

/* Reference run — 48-core Xeon w7-2495X, gcc -O3 -fopenmp, OMP_NUM_THREADS=48:

$ OMP_NUM_THREADS=48 ./04_schedule
threads (max)        : 48
serial               : 4.4101 s   (reference sum 3.660858e-05)
schedule(static)     : 0.3403 s   speedup 12.96x   PASS
schedule(dynamic,64) : 0.2648 s   speedup 16.65x   PASS
schedule(guided)     : 0.2449 s   speedup 18.01x   PASS

All three schedules compute the same sum (PASS vs serial). static hands out
equal-sized contiguous blocks up front, so the thread owning the high-i
block does far more work while the rest idle -> slowest. dynamic pulls
64-iteration chunks from a shared queue on demand, so idle threads keep
grabbing work -> better balance. guided starts with big chunks and shrinks
them, keeping balance while cutting queue overhead -> fastest here.         */