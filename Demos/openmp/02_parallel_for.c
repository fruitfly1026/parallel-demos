/* Parallel-for — split a vector operation c = a + b*s across threads and
   check the result against a serial reference; time serial vs parallel.

   Build:  gcc -O3 -fopenmp 02_parallel_for.c -o 02_parallel_for
   Run:    OMP_NUM_THREADS=48 ./02_parallel_for

   Concept: '#pragma omp parallel for' divides the loop's iteration space
   among the team. The iterations here are independent (each writes its own
   c[i]), so the parallel result is bit-identical to the serial one — that
   is why the correctness check passes. Speedup is capped by memory
   bandwidth for this streaming kernel, so it stays well under the ideal.  */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (200*1000*1000)   /* 200M elements */

static void axpy_serial(const float *a, const float *b, float *c, float s) {
    for (long i = 0; i < N; i++) c[i] = a[i] + b[i] * s;
}

static void axpy_parallel(const float *a, const float *b, float *c, float s) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) c[i] = a[i] + b[i] * s;
}

int main(void) {
    float *a = malloc((size_t)N * sizeof *a);
    float *b = malloc((size_t)N * sizeof *b);
    float *c = malloc((size_t)N * sizeof *c);   /* serial reference */
    float *d = malloc((size_t)N * sizeof *d);   /* parallel result  */
    if (!a || !b || !c || !d) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (long i = 0; i < N; i++) { a[i] = (float)(i % 7); b[i] = (float)(i % 13); }
    const float s = 2.5f;

    double t0 = omp_get_wtime();
    axpy_serial(a, b, c, s);
    double t_serial = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    axpy_parallel(a, b, d, s);
    double t_par = omp_get_wtime() - t0;

    /* Correctness check: parallel vs serial reference, element by element. */
    long mism = 0;
    for (long i = 0; i < N; i++) if (c[i] != d[i]) mism++;

    printf("threads (max)  : %d\n", omp_get_max_threads());
    printf("serial   time  : %.4f s\n", t_serial);
    printf("parallel time  : %.4f s\n", t_par);
    printf("speedup        : %.2fx\n", t_serial / t_par);
    printf("correctness    : %s (%ld mismatches vs serial)\n",
           mism == 0 ? "PASS" : "FAIL", mism);

    free(a); free(b); free(c); free(d);
    return 0;
}

/* Reference run — 48-core Xeon w7-2495X, gcc -O3 -fopenmp:

$ OMP_NUM_THREADS=1 ./02_parallel_for
threads (max)  : 1
serial   time  : 0.4210 s
parallel time  : 0.3671 s
speedup        : 1.15x
correctness    : PASS (0 mismatches vs serial)

$ OMP_NUM_THREADS=4 ./02_parallel_for
threads (max)  : 4
serial   time  : 0.3762 s
parallel time  : 0.1155 s
speedup        : 3.26x
correctness    : PASS (0 mismatches vs serial)

$ OMP_NUM_THREADS=48 ./02_parallel_for
threads (max)  : 48
serial   time  : 0.3662 s
parallel time  : 0.0314 s
speedup        : 11.68x
correctness    : PASS (0 mismatches vs serial)

Speedup grows with threads but saturates well below 48x: this AXPY kernel
is memory-bandwidth-bound (3 streams, ~1 flop/element), so once the DRAM
channels are full, more threads cannot go faster. Correctness is exact
because the iterations are independent.                                    */